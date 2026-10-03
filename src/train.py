import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, TensorDataset

DATASETS = ["figtxt", "wildjailbreak_vanilla", "wildjailbreak_adversarial", "jbb_behaviors"]
SPLIT_SIZES = {
    "figtxt": (177, 250),
    "wildjailbreak_vanilla": (177, 500),
    "wildjailbreak_adversarial": (177, 500),
    "jbb_behaviors": (75, 150),
}
SEEDS = [42, 123, 789]
NUM_LABELED = 80
NUM_UNLABELED = 2000
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
WARMUP_EPOCHS = 20
SURE_EPOCHS = 30
TAU = 0.95
MIN_TAU = 0.8
LAMBDA_U = 1.0
MAX_COVERAGE = 0.8
NOISE_STD = 0.02
# Worker processes change the order in which DataLoader shuffling draws from the
# global RNG; 4 is required to reproduce the paper's numbers exactly.
NUM_WORKERS = 4


def load_data():
    data = {}
    for name in DATASETS:
        with open(f"data/{name}.json", encoding="utf-8") as f:
            labels = [s["label"] for s in json.load(f)]
        feats = load_file(f"features/{name}.safetensors")
        data[name] = {
            "x": feats["hidden_states"],
            "safe": torch.tensor([l == "safe" for l in labels], dtype=torch.float32),
            "llm_unsafe": feats["llm_unsafe"].bool(),
            "nll": feats["llm_nll"].double(),
        }
    nll = torch.cat([d["nll"] for d in data.values()])
    for d in data.values():
        d["u"] = ((d.pop("nll") - nll.min()) / (nll.max() - nll.min())).float()
    return data


def make_splits(data, seed):
    labeled, unlabeled, val = [], [], []
    for name in DATASETS:
        n = len(data[name]["safe"])
        n_labeled, n_val = SPLIT_SIZES[name]
        ids = list(range(n))
        random.Random(seed).shuffle(ids)
        pool = ids[:n_labeled]
        safe = [i for i in pool if data[name]["safe"][i]]
        unsafe = [i for i in pool if not data[name]["safe"][i]]
        if safe:
            k = min(len(safe), len(unsafe))
            pool = safe[:k] + unsafe[:k]
        labeled.append([(name, i) for i in pool])
        unlabeled += [(name, i) for i in ids[n_labeled:n - n_val]]
        val += [(name, i) for i in ids[n - n_val:]]
    smallest = min(len(p) for p in labeled)
    labeled = [s for p in labeled for s in p[:smallest]][:NUM_LABELED]
    random.Random(seed).shuffle(unlabeled)
    return labeled, unlabeled[:NUM_UNLABELED], val


def to_dataset(data, samples):
    cols = [torch.stack([data[name][k][i] for name, i in samples]) for k in ("x", "safe", "llm_unsafe", "u")]
    ds_idx = torch.tensor([DATASETS.index(name) for name, _ in samples])
    return TensorDataset(cols[0], cols[1], ds_idx, cols[2], cols[3])


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers, dims = [], [input_dim, 2048, 512, 128]
        for d_in, d_out in zip(dims, dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(), nn.Dropout(DROPOUT)]
        self.net = nn.Sequential(*layers, nn.Linear(dims[-1], 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = [model(x.to(device)).squeeze(1).cpu() for x, *_ in loader]
    return torch.cat(probs)


def accuracy(model, loader, device):
    labels = loader.dataset.tensors[1]
    return ((predict(model, loader, device) >= 0.5).float() == labels).sum().item() / len(labels)


def warmup(model, labeled_loader, val_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=WARMUP_EPOCHS)
    best_acc, best_state = 0.0, None
    for epoch in range(1, WARMUP_EPOCHS + 1):
        model.train()
        for x, y, *_ in labeled_loader:
            optimizer.zero_grad()
            F.binary_cross_entropy(model(x.to(device)), y.to(device).unsqueeze(1)).backward()
            optimizer.step()
        acc = accuracy(model, val_loader, device)
        scheduler.step()
        if acc > best_acc and epoch > 1:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)


def flexmatch_tau(n_unsafe, n_safe):
    betas = [n / max(n_unsafe, n_safe) for n in (n_unsafe, n_safe)]
    return [max(TAU * b / (2.0 - b), MIN_TAU) for b in betas]


@torch.no_grad()
def pseudo_label(model, unlabeled, thresholds, device):
    model.eval()
    xs, labels, weights, counts = [], [], [], {}
    for x, _, ds_idx, llm_unsafe, u in DataLoader(unlabeled, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        x = x.to(device)
        x_weak = x + torch.randn_like(x) * (NOISE_STD * x.std(dim=0).clamp(min=1e-6))
        p = model(x_weak).squeeze(1)
        tau = torch.tensor([thresholds[d] for d in ds_idx.tolist()], device=device)
        keep = ((p < 0.5) & (1 - p >= tau[:, 0])) | ((p >= 0.5) & (p >= tau[:, 1]))
        pred_safe = p >= 0.5
        for d, s, k in zip(ds_idx.tolist(), pred_safe.tolist(), keep.tolist()):
            counts.setdefault(d, [0, 0])[int(s)] += int(k)
        if not keep.any():
            continue
        p, pred_safe, u = p[keep], pred_safe[keep], u.to(device)[keep]
        conf = torch.where(pred_safe, p, 1 - p)
        agree = pred_safe == ~llm_unsafe.to(device)[keep]
        weights.append(torch.where(agree, conf * (1 - u), conf * u).cpu())
        xs.append(x[keep].cpu())
        labels.append(pred_safe.float().cpu())
    if not xs:
        return None, counts
    return TensorDataset(torch.cat(xs), torch.cat(labels), torch.cat(weights)), counts


def sure_epoch(model, labeled_loader, pseudo_loader, optimizer, device):
    model.train()
    pseudo_first = pseudo_loader is not None and len(pseudo_loader) > len(labeled_loader)
    primary, secondary = (pseudo_loader, labeled_loader) if pseudo_first else (labeled_loader, pseudo_loader)
    secondary_iter = iter(secondary) if secondary is not None else None

    def next_secondary():
        nonlocal secondary_iter
        try:
            return next(secondary_iter)
        except StopIteration:
            secondary_iter = iter(secondary)
            return next(secondary_iter)

    for batch in primary:
        if pseudo_first:
            (x_u, y_u, w_u), (x_l, y_l, *_) = batch, next_secondary()
        else:
            x_l, y_l, *_ = batch
            x_u, y_u, w_u = next_secondary() if secondary is not None else (None,) * 3
        optimizer.zero_grad()
        loss = F.binary_cross_entropy(model(x_l.to(device)), y_l.to(device).unsqueeze(1))
        if x_u is not None and x_u.size(0) >= 2:
            per_sample = F.binary_cross_entropy(model(x_u.to(device)), y_u.to(device).unsqueeze(1),
                                                reduction="none")
            loss = loss + LAMBDA_U * (per_sample * w_u.to(device).unsqueeze(1)).mean()
        loss.backward()
        optimizer.step()


def train_sure(model, labeled_loader, unlabeled, val_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SURE_EPOCHS)
    counts = {d: [1, 1] for d in range(len(DATASETS))}
    pseudo, frozen = None, False
    best_acc, best_state = 0.0, None
    for _ in range(SURE_EPOCHS):
        if not frozen:
            thresholds = {d: flexmatch_tau(*c) for d, c in counts.items()}
            pseudo, new_counts = pseudo_label(model, unlabeled, thresholds, device)
            counts.update({d: [max(n, 1) for n in c] for d, c in new_counts.items()})
            frozen = pseudo is not None and len(pseudo) / len(unlabeled) >= MAX_COVERAGE
        pseudo_loader = None
        if pseudo is not None:
            pseudo_loader = DataLoader(pseudo, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        sure_epoch(model, labeled_loader, pseudo_loader, optimizer, device)
        acc = accuracy(model, val_loader, device)
        scheduler.step()
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)


def evaluate(probs, val, data):
    results = {}
    for name in DATASETS:
        r = {}
        for safe, key in ((True, "benign"), (False, "harmful")):
            refused = [p < 0.5 for (ds, i), p in zip(val, probs.tolist())
                       if ds == name and bool(data[ds]["safe"][i]) == safe]
            if refused:
                r[key] = 100 * sum(refused) / len(refused)
        if "benign" in r:
            r["acc"] = (100 - r["benign"] + r["harmful"]) / 2
            r["hm"] = 2 * (100 - r["benign"]) * r["harmful"] / (100 - r["benign"] + r["harmful"])
        results[name] = r
    with_benign = [r for r in results.values() if "benign" in r]
    results["average"] = {
        "benign": np.mean([r["benign"] for r in with_benign]),
        "harmful": np.mean([r["harmful"] for r in results.values()]),
        "acc": np.mean([r["acc"] for r in with_benign]),
        "hm": np.mean([r["hm"] for r in with_benign]),
    }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data()
    results = []
    for seed in SEEDS:
        labeled, unlabeled, val = make_splits(data, seed)
        torch.manual_seed(seed)
        model = MLP(data[DATASETS[0]]["x"].shape[1]).to(device)
        labeled_loader = DataLoader(to_dataset(data, labeled), batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=NUM_WORKERS)
        val_loader = DataLoader(to_dataset(data, val), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        warmup(model, labeled_loader, val_loader, device)
        train_sure(model, labeled_loader, to_dataset(data, unlabeled), val_loader, device)
        results.append(evaluate(predict(model, val_loader, device), val, data))
        print(f"seed {seed}: HM {results[-1]['average']['hm']:.2f}")

    print(f"\n{'':<28}{'benign↓':>14}{'harmful↑':>14}{'acc↑':>14}{'HM↑':>14}")
    for name in DATASETS + ["average"]:
        row = f"{name:<28}"
        for key in ("benign", "harmful", "acc", "hm"):
            vals = [r[name][key] for r in results if key in r[name]]
            row += f"{np.mean(vals):>8.1f}±{np.std(vals):<5.1f}" if vals else f"{'--':>14}"
        print(row)


if __name__ == "__main__":
    main()

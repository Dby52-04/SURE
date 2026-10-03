import csv
import io
import json
import os
import urllib.request

from datasets import load_dataset

FIGTXT_URL = (
    "https://raw.githubusercontent.com/ThuCCSLab/FigStep/"
    "0861b17b3d67887c06ee3534ec65b3012f9becb7/data/question/safebench.csv"
)
FIGTXT_SAFE_CATEGORIES = {"Legal Opinion", "Financial Advice", "Health Consultation"}
WILDJAILBREAK_REVISION = "5ddc12a7894f842b0619b8e1c7ee496b198af009"
JBB_REVISION = "886acc352a31533ffbcf4ef22c744658688086fc"


def figtxt():
    with urllib.request.urlopen(FIGTXT_URL) as resp:
        rows = csv.DictReader(io.StringIO(resp.read().decode("utf-8")))
        return [(r["question"], r["category_name"] in FIGTXT_SAFE_CATEGORIES) for r in rows]


def wildjailbreak(rows, kind):
    harmful = [r[kind] for r in rows if r["data_type"] == f"{kind}_harmful"][:2000]
    benign = [r[kind] for r in rows if r["data_type"] == f"{kind}_benign"][:2000]
    return [(p, False) for p in harmful] + [(p, True) for p in benign]


def jbb():
    rows = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison", split="test",
                        revision=JBB_REVISION)
    return [(r["prompt"], False) for r in rows]


def save(name, samples):
    data = [{"id": i, "prompt": p, "label": "safe" if safe else "unsafe"}
            for i, (p, safe) in enumerate(samples)]
    with open(f"data/{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    os.makedirs("data", exist_ok=True)
    save("figtxt", figtxt())
    rows = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False,
                        revision=WILDJAILBREAK_REVISION)["train"]
    save("wildjailbreak_vanilla", wildjailbreak(rows, "vanilla"))
    save("wildjailbreak_adversarial", wildjailbreak(rows, "adversarial"))
    save("jbb_behaviors", jbb())


if __name__ == "__main__":
    main()

# SURE: Data-Efficient Safety Guardrailing via Internal Representations and Uncertainty-Weighted Pseudo-Labels (npj Artificial Intelligence)

Hanwen Li, Jinhao Duan, Chenxi Yuan, James Diffenderfer, Sandeep Madireddy, Bhavya Kailkhura, Kaidi Xu

This is the official implementation of the npj Artificial Intelligence paper
[*SURE: Data-Efficient Safety Guardrailing via Internal Representations and Uncertainty-Weighted
Pseudo-Labels*](https://www.nature.com/articles/s44387-026-00141-y).

![SURE overview](assets/overview.png)

## Setup

Requires Python 3.11 and one GPU. Request access to the gated
[WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) dataset and
[Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model first.

```bash
pip install -r requirements.txt
huggingface-cli login
```

## 1. Build the datasets

```bash
python build_dataset.py    # writes data/*.json
```

| File | Source | Safe / Unsafe |
|---|---|---|
| `data/figtxt.json` | [FigStep SafeBench](https://github.com/ThuCCSLab/FigStep) questions; Legal Opinion / Financial Advice / Health Consultation are labeled `safe`, the other 7 categories `unsafe` | 150 / 350 |
| `data/wildjailbreak_vanilla.json` | [allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) `train`, first 2000 `vanilla_harmful` + first 2000 `vanilla_benign` | 2000 / 2000 |
| `data/wildjailbreak_adversarial.json` | same, `adversarial_harmful` / `adversarial_benign` | 2000 / 2000 |
| `data/jbb_behaviors.json` | [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) `judge_comparison` test prompts | 0 / 300 |

Every sample is `{"id", "prompt", "label"}`. All sources are pinned to fixed revisions, so ids are reproducible.

## 2. Build the features

```bash
python build_features.py   # reads data/*.json, writes features/*.safetensors
```

Each `features/<dataset>.safetensors` holds, for sample id `i` in row `i`:

| Tensor | Shape | Meaning |
|---|---|---|
| `hidden_states` | (N, 4096) | last-token hidden state of the prompt at layer 17 of Llama-3-8B-Instruct — the classifier input |
| `llm_unsafe` | (N,) | 1 if the LLM, prompted as a safety classifier, answers UNSAFE |
| `llm_nll` | (N,) | negative log-probability of the first token of that answer — the LLM's uncertainty |

`llm_unsafe` and `llm_nll` weight the pseudo-labels during semi-supervised training.

## 3. Train and evaluate

```bash
python train.py            # seeds 42, 123, 789
```

For each seed the prompts are split into 80 labeled, 2000 unlabeled and 1400 validation prompts. An MLP on the hidden states is warmed up on the labeled set and then trained with SURE: confident predictions on the unlabeled pool become pseudo-labels (per-dataset, per-class adaptive thresholds), and each pseudo-label is weighted by how strongly the LLM's own verdict and uncertainty support it. The script prints, per dataset and averaged over seeds, the refusal rate on benign prompts (↓), on harmful prompts (↑), their accuracy and harmonic mean (HM).

## Citation

```bibtex
@article{li2026sure,
  title   = {{SURE}: Data-Efficient Safety Guardrailing via Internal Representations and Uncertainty-Weighted Pseudo-Labels},
  author  = {Li, Hanwen and Duan, Jinhao and Yuan, Chenxi and Diffenderfer, James and Madireddy, Sandeep and Kailkhura, Bhavya and Xu, Kaidi},
  journal = {npj Artificial Intelligence},
  year    = {2026},
  doi     = {10.1038/s44387-026-00141-y}
}
```

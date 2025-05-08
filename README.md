# Exploring Multi‑Head Latent Attention in DeepSeek‑Coder
*A systematic benchmark of LoRA, QLoRA & baseline fine‑tuning across multiple attention kernels*

---

## Project Motivation
Large‑language‑model fine‑tuning is often constrained by **GPU memory**, **training speed**, and **engineering complexity**. This project delivers a framework to **measure the real‑world trade‑offs** between adapter techniques (Baseline, LoRA, QLoRA) under a family of attention implementations:

* **Eager / SDPA** (PyTorch standard)
* **FlashAttention‑2**
* **Multi‑Head Latent Attention (MLA)**

All experiments use **DeepSeek‑Coder‑1.3B** and a slice of **OpenOrca** for fast iteration during the mid‑point review. We are *not* re‑implementing kernels—just benchmarking.

---

## Repo Outline
```text
.
├── attentions/                 # Custom kernels (mla.py)
├── configs/                    # JSON experiment configs (seq‑len, kernels, …)
├── deepseek‑models/            # Git sub‑module with patched DeepSeek‑Coder‑1.3B
├── benchmark_attn_qlora.py     # QLoRA       benchmark script
├── benchmark_wandb.py          # Thin wrapper that streams metrics to Weights & Biases
├── finetune_deepseek_attn.py   # One‑off fine‑tune helper (no benchmarking)
├── finetune_wandb.py           # Same as above but logs to wandb
├── requirements.txt            # Python dependencies
└── README.md                   # ← this file
```

---

## Setup & Requirements
```bash
# 1 · Install deps (T4 GPU, Deep Learning VM with CUDA 12.1+, M126, Debian 11, Python 3.10)
pip install -r requirements.txt
```

---

## How to Run
```bash
python benchmark_attn_qlora.py deepseek-ai/deepseek-coder-1.3b-base configs/test.json runs/bench
```

###  Stream metrics to Weights & Biases
```bash
export WANDB_API_KEY=<your‑key>

pip install wandb
wandb login

python finetune_wandb.py deepseek-ai/deepseek-coder-1.3b-base configs/test.json runs/bench
```
Our results can be found in the [public report](https://wandb.ai/louiszh-columbia-university/finetune_with_tables/reports/HPML-Adapter-Benchmarking-Results--VmlldzoxMjY4MTQzMA).

---

## Results
| Mode      | Kernel            | Train t ↓ | Sample/s ↑ | Peak MiB ↓ | 
|-----------|-------------------|-----------|------------|------------|
| **LoRA**  | eager             | 24.5 s    | 4.116      | 7226       |
| **LoRA**  | sdpa              | 24.9 s    | 4.063      | 7226       |
| **LoRA**  | flash‑attn‑2      | 24.6 s    | 4.100      | 7226       |
| **LoRA**  | MLA               | 20.7 s    | 4.910      | 7226       |
| **QLoRA** | eager             | 88.0 s    | 1.139      | 7226       |
| **QLoRA** | sdpa              | 87.8 s    | 1.143      | 7226       |
| **QLoRA** | flash‑attn‑2      | 87.8 s    | 1.141      | 7226       |
| **QLoRA** | MLA               | 77.6 s    | 1.293      | 8294       |

---

## Observations
* LoRA + MLA is fastest: 4.91 samples/sec, 20.7s runtime — due to low-rank updates + efficient MLA attention.
* LoRA is ~4× faster than QLoRA: LoRA avoids quantization overhead; QLoRA requires dequantization during training.
* QLoRA + MLA is fastest QLoRA setup: Best among QLoRA variants, but ~15% higher memory usage (8294 MiB).
* Tradeoff: QLoRA saves memory via quantization, but training is significantly slower than LoRA.

---

## Reproduction & Extend
1. **Change kernels** – edit `configs/*.json` → "attn_impls" : ["eager", "flash_attention_2", …].
2. **Bigger data**    – bump `TRAIN_SIZE`, `VAL_SIZE`, or point to a HF dataset split.
3. **More epochs**    – adjust `num_train_epochs` in the config or script.
4. **Distributed**    – pass `--deepspeed ds_config.json`

---

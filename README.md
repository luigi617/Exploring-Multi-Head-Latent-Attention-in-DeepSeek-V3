# Exploring Multi‑Head Latent Attention in DeepSeek‑Coder
*A systematic benchmark of LoRA, QLoRA & baseline fine‑tuning across multiple attention kernels*

---

## Project Motivation
Large‑language‑model fine‑tuning is often constrained by **GPU memory**, **training speed**, and **engineering complexity**. This project delivers a framework to **measure the real‑world trade‑offs** between adapter techniques (Baseline, LoRA, QLoRA) under a family of attention implementations:

* **Eager / SDPA** (PyTorch standard)
* **FlashAttention‑2**
* **Multi‑Head Latent Attention (MLA)**

All experiments use **DeepSeek‑Coder‑1.3B** and a slice of **OpenOrca** for fast iteration. We are *not* re‑implementing kernels, just benchmarking.

---

## Repo Outline
```text
.
├── attentions/                 # Custom kernels (mla.py)
├── configs/                    # JSON experiment configs
├── deepseek‑models/            # Git sub‑module with patched DeepSeek‑Coder‑1.3B
├── benchmark_attn_qlora.py     # QLoRA       benchmark script
├── benchmark_wandb.py          # Thin wrapper that streams metrics to Weights & Biases
├── finetune_deepseek_attn.py   # One‑off fine‑tune helper (no benchmarking)
├── finetune_wandb.py           # Same as above but logs to wandb
├── requirements.txt            # Python dependencies
└── README.md                   
```

---

## Setup & Requirements
```bash
# Install deps
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
Our results can be found in the [public report](https://api.wandb.ai/links/louiszh-columbia-university/5mknyy8s).

---

## Results
| Mode      | Kernel            | Train t ↓ | Sample/s ↑ | Peak MiB ↓ | 
|-----------|-------------------|-----------|------------|------------|
| **LoRA**  | eager             | 237.9 s    | 4.207      | 8480       |
| **LoRA**  | sdpa              | 238.0 s    | 4.208      | 8480       |
| **LoRA**  | flash‑attn‑2      | 238.1 s    | 4.206      | 8480       |
| **LoRA**  | MLA               | 184.2 s    | 5.442      | 8480       |
| **QLoRA** | eager             | 827.5 s    | 1.209      | 7226       |
| **QLoRA** | sdpa              | 827.5 s    | 1.209      | 7226       |
| **QLoRA** | flash‑attn‑2      | 827.7 s    | 1.209      | 7226       |
| **QLoRA** | MLA               | 481.4 s    | 2.709      | 8294       |

---

## Observations
* LoRA + MLA is fastest: 5.442 samples/sec, 184.2s runtime — due to low-rank updates + efficient MLA attention.
* LoRA is ~4× faster than QLoRA: LoRA avoids quantization overhead; QLoRA requires dequantization during training.
* QLoRA + MLA is fastest QLoRA setup: Best among QLoRA variants, but ~15% higher memory usage.
* Tradeoff: QLoRA saves memory via quantization, but training is significantly slower than LoRA.

---

## Reproduction & Extend
1. **Change kernels** – edit `configs/*.json` → "attn_impls" : ["eager", "flash_attention_2", …].
2. **Changer data**    – bump `TRAIN_SIZE`, `VAL_SIZE`, or point to a HF dataset split.
3. **Change epochs**    – adjust `num_train_epochs` in the config or script.
4. **Distributed**    – pass `--deepspeed ds_config.json`

---

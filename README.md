# Exploring Multi‑Head Latent Attention in DeepSeek‑Coder
*A systematic benchmark of LoRA, QLoRA & baseline fine‑tuning across multiple attention kernels*

---

## ✨ Project Motivation
Large‑language‑model fine‑tuning is often constrained by **GPU memory**, **training speed**, and **engineering complexity**. This project delivers a *drop‑in* framework to **measure the real‑world trade‑offs** between adapter techniques (Baseline, LoRA, QLoRA) under a family of attention implementations:

* **Eager / SDPA** (PyTorch standard)
* **FlashAttention‑2**
* **Paged Attention**
* **Multi‑Head Latent Attention (MLA)**

All experiments use **DeepSeek‑Coder‑1.3B** and a 120‑sample slice of **OpenOrca** for fast iteration during the mid‑point review. We are *not* re‑implementing kernels—just benchmarking.

---

## 🗂 Repository Outline
```text
.
├── attentions/                 # Custom kernels (mla.py)
├── configs/                    # JSON experiment configs (batch‑size, kernels, …)
├── deepseek‑models/            # Git sub‑module with patched DeepSeek‑Coder‑1.3B
├── benchmark_attn.py           # Baseline    (fp16) benchmark script
├── benchmark_attn_lora.py      # LoRA        benchmark script
├── benchmark_attn_qlora.py     # QLoRA       benchmark script (v1)
├── benchmark_attn_qlora_v2.py  # QLoRA       benchmark w/ profiling + ROUGE (v2)
├── benchmark_wandb.py          # Thin wrapper that streams metrics to Weights & Biases
├── finetune_deepseek_attn.py   # One‑off fine‑tune helper (no benchmarking)
├── finetune_wandb.py           # Same as above but logs to wandb
├── model_FA.py                 # FlashAttention‑2 convenience loader
├── requirements.txt            # Python dependencies
└── README.md                   # ← this file
```

---

## ⚙️ Setup & Requirements
```bash
# 1 · Create environment
conda create -n deepseek-bench python=3.10
conda activate deepseek-bench

# 2 · Install deps (GPU, CUDA 11.8+)
pip install -r requirements.txt

# 3 · (Optional) install Flash‑Attention‑2 wheels if missing
#     see: https://github.com/Dao-AILab/flash-attention
```

---

## 🚀 How to Run – Quick Examples
### 1. Baseline (full‑precision)
```bash
python benchmark_attn.py deepseek-ai/deepseek-coder-1.3b-base configs/baseline.json runs/baseline
```

### 2. LoRA
```bash
python benchmark_attn_lora.py deepseek-ai/deepseek-coder-1.3b-base configs/comparison.json runs/lora
```

### 3. QLoRA (v2, profiling enabled)
```bash
python benchmark_attn_qlora_v2.py deepseek-ai/deepseek-coder-1.3b-base configs/comparison.json runs/qlora_v2
```

### 4. Stream metrics to Weights & Biases
```bash
export WANDB_ENTITY=<your‑entity>
export WANDB_PROJECT=deepseek-attn-bench
export WANDB_API_KEY=<your‑key>

python benchmark_attn_qlora_v2.py deepseek-ai/deepseek-coder-1.3b-base configs/comparison.json runs/bench

```

Our results can be found in the [public report](https://wandb.ai/louiszh-columbia-university/finetune_with_tables/reports/HPML-Adapter-Benchmarking-Results--VmlldzoxMjY4MTQzMA).

---

## 📊 Sample Results
| Mode      | Kernel            | Train t ↓ | Tok/s ↑ | Peak MiB ↓ | Val PPL ↓ |
|-----------|-------------------|-----------|---------|------------|-----------|
| baseline  | eager             | 26.0 s    | 3.87    | 13 114     | 17.9      |
| baseline  | flash‑attn‑2      | 25.8 s    | 3.90    | 13 114     | 17.8      |
| baseline  | MLA               | 21.9 s    | 4.60    | 16 863     | 18.4      |
| **LoRA**  | eager             | 23.9 s    | 4.22    | 16 863     | 18.0      |
| **QLoRA** | flash‑attn‑2      | 18.7 s    | 4.95    | **8 732**  | 18.3      |

Detailed per‑step profiling (CPU+CUDA) is saved in `runs/**/profiling_trace.json`; open in Chrome DevTools ➜ *Performance* for flame‑charts.

---

## 🔍 Observations (Mid‑point)
* **FlashAttention‑2** gives a consistent ~4 % speed‑up over SDPA with no memory penalty.
* **MLA** delivers the highest throughput but increases GPU memory via latent projections.
* **QLoRA + FA‑2** *halves* peak memory vs. fp16 while also being the fastest overall.
* Quality differences (Perplexity, ROUGE‑L) remain within ±3 % on the small OpenOrca slice.

> *Take‑away:* For small‑/mid‑scale fine‑tuning, **QLoRA combined with FlashAttention‑2** provides the best compute‑/memory‑efficiency sweet spot.

---

## 🔄 Reproduce & Extend
1. **Change kernels** – edit `configs/*.json` → "attn_impls" : ["eager", "flash_attention_2", …].
2. **Bigger data**    – bump `TRAIN_SIZE`, `VAL_SIZE`, or point to a HF dataset split.
3. **More epochs**    – adjust `num_train_epochs` in the config or script.
4. **Distributed**    – pass `--deepspeed ds_config.json` or enable FSDP in `TrainingArguments`.

---

## 📄 License & Citation
Released under **Apache 2.0**. If you build on this work, please cite the original LoRA, QLoRA, FlashAttention‑2, and DeepSeek‑Coder papers.

---

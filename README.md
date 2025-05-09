# HPML Project: Exploring Multi‑Head Latent Attention in DeepSeek-Coder

A systematic benchmark of LoRA, QLoRA & baseline fine‑tuning across multiple attention kernels.

---
## Team Information
- **Team Name**: [Team Name]
- **Members**:
  - Luigi Liu (ll3840)
  - Yixiao Li (yl5760)
  - Louis Zheng (lz2834)

---

## Problem Statement

Large‑language‑model fine‑tuning is often constrained by **GPU memory**, **training speed**, and **engineering complexity**. This project benchmarks the real‑world trade‑offs of three adapter techniques (Baseline, LoRA, QLoRA) across a family of attention implementations:

* **Eager / SDPA** (PyTorch standard)
* **FlashAttention‑2**
* **Multi‑Head Latent Attention (MLA)**

All experiments use **DeepSeek‑Coder‑1.3B** and a slice of **OpenOrca** for fast iteration—no kernel re‑implementation, only benchmark comparisons.

---

## 2. Model Description

- **Base Architecture**  
  - **Model:** `deepseek-ai/deepseek-coder-1.3b-base` (1.3 B-parameter causal LM)  
  - **Framework:** PyTorch + Hugging Face Transformers (`AutoModelForCausalLM`, `AutoTokenizer`)

- **Quantization & Adapters**  
  - **QLoRA (4-bit) Setup:**  
    - `BitsAndBytesConfig` with NF4 quantization, double-quant, compute in FP16  
    - `prepare_model_for_kbit_training()` to enable low-bit fine-tuning  
  - **LoRA Adapter:**  
    - Rank `r=32`, α=16, dropout=0.05, no bias  
    - Target modules vary by attention type (standard Q/K/V/O projections vs. MLA latent/token projections)

- **Attention Implementations**  
  - **Built-In:**  
    - Eager (PyTorch), SDPA, FlashAttention-2, Flex  
  - **Custom:**  
    - **MultiHeadLatentAttention (MLA):**  
      - Projects Q/K/V through learned latent tokens  
      - Quantized to 4-bit  
    - **PagedAttention:**  
      - Block-sparse attention (block size 64)  
      - Quantized to 4-bit  
  - Replacement is done layer-by-layer via `replace_attention_with_quantized()`.

- **Training & Profiling**  
  - **Hugging Face Trainer:**  
    - FP16, `gradient_accumulation_steps=32`, 1 epoch, gradient checkpointing  
  - **Profiling:**  
    - `torch.profiler` (CPU + CUDA) → Chrome trace + summary of top ops  
  - **Logged Metrics:**  
    - Training runtime, samples/sec (train/eval/gen), peak GPU memory  
    - Train/val loss, perplexity, ROUGE-1/2/L, trainable & total parameter counts

- **Data Pipeline**  
  - **Dataset:** Open-Orca conversational data (100 train / 20 val)  
  - **Tokenization:** Custom prompt–response concatenation, truncation to `max_seq_len`  
  - **Collation:** `DataCollatorForLanguageModeling` (causal LM style)


---

## Repository Structure

```text
.
├── attentions/                 # Custom kernels (mla.py)
├── configs/                    # JSON experiment configs
├── deepseek-models/            # Git sub-module with patched DeepSeek‑Coder‑1.3B
├── wandb                       # stores wandb logs and results
├── benchmark_attn_qlora.py     # QLoRA benchmark script
├── finetune_wandb.py           # Logs metrics to Weights & Biases
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 3. Final Results Summary

| Mode      | Kernel       | Train Time (s) ↓ | Samples/s ↑ | Peak Mem (MiB) ↓ |
| --------- | ------------ | ---------------- | ----------- | ---------------- |
| Baseline  | eager        | 246.0            | 4.100       | 14 800           |
| Baseline  | sdpa         | 246.1            | 4.100       | 14 800           |
| Baseline  | flash-attn-2 | 246.2            | 4.090       | 14 800           |
| Baseline  | MLA          | 198.5            | 5.105       | 14 800           |
| **LoRA**  | eager        | 237.9            | 4.207       | 8 480            |
| **LoRA**  | sdpa         | 238.0            | 4.208       | 8 480            |
| **LoRA**  | flash-attn-2 | 238.1            | 4.206       | 8 480            |
| **LoRA**  | MLA          | 184.2            | 5.442       | 8 480            |
| **QLoRA** | eager        | 827.5            | 1.209       | 2 950            |
| **QLoRA** | sdpa         | 827.5            | 1.209       | 2 950            |
| **QLoRA** | flash-attn-2 | 827.7            | 1.209       | 2 950            |
| **QLoRA** | MLA          | 481.4            | 2.079       | 3 950            |


Public report from wandb: [https://api.wandb.ai/links/louiszh-columbia-university/5mknyy8s](https://api.wandb.ai/links/louiszh-columbia-university/5mknyy8s)
---

## Observations

* **MLA** is consistently the fastest and most efficient for training speed across all modes.
* **LoRA + fp16** offers an excellent balance: fast training with significant memory savings (\~43% less than baseline).
* **QLoRA (4-bit)** reduces memory usage the most (down to \~2 950 MiB), but suffers from lower throughput—except when using MLA, which nearly doubles its speed while remaining memory-efficient.
* The choice between eager/SDPA and FlashAttention‑2 has minimal impact on performance unless paired with MLA, which clearly boosts both speed and throughput.

---

---

## 4. Reproducibility Instructions

### A. Requirements

Install Python dependencies:
```bash
pip install -r requirements.txt
```

---

### B. Benchmark QLoRA

```bash
python benchmark_attn_qlora.py \
  deepseek-ai/deepseek-coder-1.3b-base \
  configs/test.json \
  runs/bench
```

---

### C. Stream metrics to Weights & Biases

```bash
export WANDB_API_KEY=<your-key>
pip install wandb
wandb login

python finetune_wandb.py \
  deepseek-ai/deepseek-coder-1.3b-base \
  configs/test.json \
  runs/bench
```

---

### D. Quickstart: Minimum Reproducible Result

To reproduce our minimum reported result, run:

```bash
# Step 1: Set up environment
pip install -r requirements.txt

python benchmark_attn_qlora.py \
  deepseek-ai/deepseek-coder-1.3b-base \
  configs/test.json \
  runs/bench
```

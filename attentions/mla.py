#!/usr/bin/env python
"""
attentions/mla.py
Implementation of **Multi-Head Latent Attention (MLA)** – a two-stage
cross-attention block in which a small set of learnable latent vectors
first attends over the full input sequence and is then attended back by
the tokens.  The layer keeps the same public interface as the Hugging
Face `SelfAttention` module so it can be swapped-in transparently.

The layer exposes eight projections so that LoRA/QLoRA can be applied
independently to every leg of the bi-directional attention:

    ── latents ← tokens  :  to_q_latent · to_k_token · to_v_token · out_latent
    ── tokens  ← latents :  to_q_token · to_k_latent · to_v_latent · out_token
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------------
# small helper copied from the earlier snippet so the file is self-contained
def get_linear(in_features: int,
               out_features: int,
               bias: bool,
               quantization: Optional[str] = None) -> nn.Module:
    """
    Returns an nn.Module implementing y = Wx (+b) under an optional
    `quantization` spec.  If bitsandbytes is **not** available the call
    simply falls back to a regular `nn.Linear`.
    """
    if quantization is None or quantization.lower() in {"none", "fp32", ""}:
        return nn.Linear(in_features, out_features, bias=bias)

    try:
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
    except (ImportError, RuntimeError):
        return nn.Linear(in_features, out_features, bias=bias)

    q = quantization.lower()
    if q in {"8bit", "int8"}:
        return Linear8bitLt(
            in_features, out_features, bias=bias, activation_fn=None
        )
    if q in {"4bit", "int4"}:
        # nf4 with fp16 accumulation is a sensible default
        return Linear4bit(
            in_features, out_features, bias=bias,
            quant_type="nf4", compute_dtype=torch.float16
        )
    raise ValueError(f"Unsupported quantization mode: {quantization}")
# --------------------------------------------------------------------------------------------------


class MultiHeadLatentAttention(nn.Module):
    """
    A *minimal* yet fully-functional implementation of the latent attention
    block described in Perceiver-style models, adapted so it can replace the
    standard self-attention used in decoder-only LLMs.

    The class purposefully keeps the Hugging-Face causal-LM API:

        hidden_states, present_kv = mla(hidden_states,
                                        attention_mask      = ...,
                                        past_key_value       = ...,
                                        output_attentions    = ...,
                                        use_cache            = ...)

    Only the first tensor return is used by the benchmark script, but the
    extra outputs make the layer a drop-in replacement for inference paths
    that expect KV caching or attention maps.
    """

    def __init__(
        self,
        cfg,                         # a `transformers.PretrainedConfig`
        num_latents: Optional[int] = None,
        num_heads_latent: Optional[int] = None,
        num_heads_token: Optional[int] = None,
        dropout: float = 0.0,
        quantization: Optional[str] = "4bit",
    ):
        super().__init__()

        self.hidden_size: int = getattr(cfg, "hidden_size", cfg.hidden_size)
        self.num_heads_latent = num_heads_latent or getattr(
            cfg, "num_attention_heads", 8
        )
        self.num_heads_token = num_heads_token or self.num_heads_latent
        self.head_dim = self.hidden_size // self.num_heads_token
        if self.head_dim * self.num_heads_token != self.hidden_size:
            raise ValueError(
                "`hidden_size` must be divisible by `num_heads_token`."
            )

        # learnable latent vectors -------------------------------------------------
        self.num_latents = num_latents or getattr(cfg, "num_latents", 64)
        self.latents = nn.Parameter(
            torch.randn(self.num_latents, self.hidden_size) * 0.02
        )

        # 1) latents query the tokens ---------------------------------------------
        self.to_q_latent = get_linear(
            self.hidden_size, self.hidden_size, bias=False, quantization=quantization
        )
        self.to_k_token = get_linear(
            self.hidden_size, self.hidden_size, bias=False, quantization=quantization
        )
        self.to_v_token = get_linear(
            self.hidden_size, self.hidden_size, bias=False, quantization=quantization
        )
        self.out_latent = get_linear(
            self.hidden_size, self.hidden_size, bias=False, quantization=quantization
        )

        # 2) tokens query the (updated) latents -----------------------------------
        self.to_q_token = get_linear(
            self.hidden_size, self.hidden_size, bias=False, quantization=quantization
        )
        self.to_k_latent = get_linear(
            self.hidden_size, self.hidden_size, bias=False, quantization=quantization
        )
        self.to_v_latent = get_linear(
            self.hidden_size, self.hidden_size, bias=False, quantization=quantization
        )
        self.out_token = get_linear(
            self.hidden_size, self.hidden_size, bias=False, quantization=quantization
        )

        self.dropout = nn.Dropout(dropout)

    # ──────────────────────────────────────────────────────────────────────────
    # internal utilities
    def _shape(self, x: torch.Tensor, n_head: int) -> torch.Tensor:
        """
        (B, L, D) → (B, n_head, L, head_dim)
        """
        B, L, _ = x.shape
        x = x.view(B, L, n_head, self.head_dim).permute(0, 2, 1, 3)
        return x

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, n_head, L, head_dim) → (B, L, D)
        """
        B, H, L, Hd = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, L, H * Hd)

    # ──────────────────────────────────────────────────────────────────────────
    # forward pass
    def forward(
        self,
        hidden_states: torch.Tensor,                      # (B, S, D)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **unused,                                        # <- NEW: absorbs cache_position and others
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        """
        Return shape & semantics match HF `LlamaAttention`:
            hidden_states           : (B, S, D)
            attention_weights / None: (B, S, L)  or  None
            present_kv / None       : ((k_lat, v_lat),)  or  None
        """
        B, S, _ = hidden_states.shape
        device = hidden_states.device

        # ------ 1) Latents attend over tokens ---------------------------------
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)

        q_lat = self._shape(self.to_q_latent(latents), self.num_heads_latent)
        k_tok = self._shape(self.to_k_token(hidden_states), self.num_heads_latent)
        v_tok = self._shape(self.to_v_token(hidden_states), self.num_heads_latent)

        attn_scores = torch.matmul(q_lat, k_tok.transpose(-2, -1))
        attn_scores *= 1 / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 4:               # (B,1,S,S)
                key_mask = attention_mask[:, :, -1:, :] # take first row
            elif attention_mask.dim() == 2:             # (B,S)
                key_mask = attention_mask[:, None, None, :]
            else:                                       # already (B,1,1,S)
                key_mask = attention_mask
            attn_scores = attn_scores + key_mask

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(v_tok.dtype)
        attn_probs = self.dropout(attn_probs)

        latents_out = torch.matmul(attn_probs, v_tok)          # (B, H_lat, L, Hd)
        latents_out = self._merge(latents_out)                 # (B, L, D)
        latents_out = self.out_latent(latents_out)             # (B, L, D)

        # ------ 2) Tokens attend over (updated) latents -----------------------
        q_tok = self._shape(self.to_q_token(hidden_states), self.num_heads_token)
        k_lat = self._shape(self.to_k_latent(latents_out), self.num_heads_token)
        v_lat = self._shape(self.to_v_latent(latents_out), self.num_heads_token)

        attn_scores2 = torch.matmul(q_tok, k_lat.transpose(-2, -1))
        attn_scores2 *= 1 / math.sqrt(self.head_dim)

        attn_probs2 = F.softmax(attn_scores2, dim=-1, dtype=torch.float32).to(v_lat.dtype)
        attn_probs2 = self.dropout(attn_probs2)

        token_out = torch.matmul(attn_probs2, v_lat)           # (B, H_tok, S, Hd)
        token_out = self._merge(token_out)                     # (B, S, D)
        token_out = self.out_token(token_out)                  # (B, S, D)

        # ---- assemble HF-compatible return tuple -----------------------------
        if output_attentions:
            # average heads: (B, S, L)
            attn_w = attn_probs2.mean(dim=1).detach()
        else:
            attn_w = None

        present = None

        return token_out, attn_w, present
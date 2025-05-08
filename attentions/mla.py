#!/usr/bin/env python
"""
attentions/mla.py
Implementation of **Multi‑Head Latent Attention (MLA)** – a bidirectional
token‑latent block that can be swapped into HF decoder layers.

Latents ← Tokens :  to_q_latent · to_k_token · to_v_token · out_latent  
Tokens ← Latents :  to_q_token · to_k_latent · to_v_latent · out_token
"""
from __future__ import annotations
from typing import Tuple, Optional
import math, torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        config,
        num_latents: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size  = config.hidden_size
        self.num_heads    = config.num_attention_heads
        self.head_dim     = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.scale        = self.head_dim ** -0.5
        self.num_latents  = num_latents
        self.dropout_p    = dropout

        # ---------- projections (LoRA‑friendly: each direction owns its matrix)
        linear = lambda: nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # latents ← tokens
        self.to_q_latent = linear()
        self.to_k_token  = linear()
        self.to_v_token  = linear()
        self.out_latent  = linear()

        # tokens ← latents
        self.to_q_token  = linear()
        self.to_k_latent = linear()
        self.to_v_latent = linear()
        self.out_token   = linear()

        # learnable latent vectors (L × D)
        self.latents = nn.Parameter(torch.randn(num_latents, self.hidden_size))

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    # -------------------------------------------------------------------- helpers
    def _shape(self, x: torch.Tensor, b: int) -> torch.Tensor:
        # (B, S, H*D) → (B, H, S, D)
        return x.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def _attention(self, q, k, v, mask=None):
        # q,k,v : (B, H, S_q, D)
        attn = (q @ k.transpose(-2, -1)) * self.scale            # (B, H, S_q, S_k)

        if mask is not None:
            # ---- sanitize mask to (B, 1, 1, S_k) ----
            if mask.dim() == 4:                  # (B, 1, S_q, S_k) from HF
                mask = mask[..., -1:, :]         # keep only padding row
            if mask.dim() == 3:                  # (B, 1, S_k)
                mask = mask[:, :, None, :]
            if mask.dim() == 2:                  # (B, S_k)
                mask = mask[:, None, None, :]
            # mask is now broadcast‑compatible with attn
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn @ v                          # (B, H, S_q, D)

    # -------------------------------------------------------------------- forward
    def forward(
        self,
        hidden_states: torch.Tensor,                 # (B, S, D)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,   # accepted, unused
        past_key_value: Optional[Tuple] = None,      # cache not yet implemented
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,                               # catches layer_head_mask, etc.
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:

        b, s, _ = hidden_states.size()
        target_dtype = self.to_q_latent.weight.dtype           # usually fp16 or fp32
   
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)
   
        latent_tokens = self.latents                         \
            .to(target_dtype)                                \
            .unsqueeze(0).expand(b, -1, -1)                  # (B, L, D)

        # --------------- Stage 1 – latents attend over **tokens**
        q_l = self._shape(self.to_q_latent(latent_tokens), b)  # (B, H, L, D)
        k_t = self._shape(self.to_k_token(hidden_states), b)   # (B, H, S, D)
        v_t = self._shape(self.to_v_token(hidden_states), b)

        latents_enriched = self._attention(q_l, k_t, v_t, attention_mask)
        latents_enriched = (
            latents_enriched.transpose(1, 2)
            .contiguous()
            .view(b, self.num_latents, self.hidden_size)
        )
        latents_enriched = self.out_latent(latents_enriched)   # (B, L, D)

        # --------------- Stage 2 – tokens attend over **enriched latents**
        q_t = self._shape(self.to_q_token(hidden_states), b)   # (B, H, S, D)
        k_l = self._shape(self.to_k_latent(latents_enriched), b)
        v_l = self._shape(self.to_v_latent(latents_enriched), b)

        tok_out = self._attention(q_t, k_l, v_l)               # (B, H, S, D)
        tok_out = (
            tok_out.transpose(1, 2)
            .contiguous()
            .view(b, s, self.hidden_size)
        )
        tok_out = self.out_token(tok_out)
        tok_out = self.proj_drop(tok_out)

        # --- HF return signature
        present = None  # no cache yet
        attn_w  = None
        if output_attentions:
            # return token‑side attention weights (average over heads)
            attn_w = torch.mean(
                (q_t @ k_l.transpose(-2, -1)) * self.scale, dim=1
            )  # (B, S, L)

        return tok_out, present, attn_w

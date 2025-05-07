#!/usr/bin/env python
"""
Paged Attention – light‑weight local attention that keeps only the
`page_size` most‑recent keys/values per head.  During generation this
acts like a size‑N cache window; during training the full sequence is
chunked into non‑overlapping pages.
"""
from __future__ import annotations
import torch, math
import torch.nn as nn
from typing import Optional, Tuple

class PagedAttention(nn.Module):
    def __init__(self, config, page_size: int = 256):
        super().__init__()
        self.hidden_size  = config.hidden_size
        self.num_heads    = config.num_attention_heads
        self.page_size    = page_size
        self.head_dim     = self.hidden_size // self.num_heads
        self.scale        = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

    # helpers ------------------------------------------------------------
    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,S,D)

    # forward ------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        b, s, _ = hidden_states.shape
        q = self._reshape(self.q_proj(hidden_states))                   # (B,H,S,D)

        if past_key_value is not None:
            k_prev, v_prev = past_key_value                             # each (B,H,L,D)
            k_new = self._reshape(self.k_proj(hidden_states))
            v_new = self._reshape(self.v_proj(hidden_states))
            k = torch.cat([k_prev, k_new], dim=2)[:, :, -self.page_size :, :]
            v = torch.cat([v_prev, v_new], dim=2)[:, :, -self.page_size :, :]
        else:
            k = self._reshape(self.k_proj(hidden_states))
            v = self._reshape(self.v_proj(hidden_states))
            # training mode: chunk S into pages (non‑overlap window)
            if not use_cache:
                k = k.unfold(dimension=2, size=self.page_size, step=self.page_size)  # (B,H,Pg,L,D)
                v = v.unfold(dimension=2, size=self.page_size, step=self.page_size)
                q = q.unfold(dimension=2, size=self.page_size, step=self.page_size)
                k, v, q = [x.reshape(-1, self.num_heads, self.page_size, self.head_dim) for x in (k, v, q)]

        scores = (q @ k.transpose(-2, -1)) * self.scale                 # (B*,H,S_q,S_k)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs  = scores.softmax(dim=-1)
        ctx    = probs @ v                                              # (B*,H,S_q,D)

        ctx = ctx.transpose(1,2).contiguous().view(b, -1, self.hidden_size)
        out = self.o_proj(ctx)

        present = (k, v) if use_cache else None
        attn = probs if output_attentions else None
        return out, present, attn

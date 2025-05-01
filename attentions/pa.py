# Lightweight paged-KV attention

from __future__ import annotations
from typing import Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit


class PagedAttention(torch.nn.Module):
    def __init__(
        self,
        config,
        block_size: int = 64,
        *,
        use_bias: bool = False,
    ):
        super().__init__()

        self.hidden_size: int = config.hidden_size
        self.num_heads: int = getattr(
            config, "num_attention_heads", getattr(config, "n_heads", None)
        )
        if self.num_heads is None:
            raise ValueError("Could not infer number of heads from config")

        # Some configs (Mistral, Llama-2/3) have separate kv-heads
        self.num_key_value_heads: int = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        self.head_dim: int = self.hidden_size // self.num_heads
        self.scale: float = self.head_dim ** -0.5

        self.block_size: int = block_size
        self.max_position_embeddings: int = getattr(
            config, "max_position_embeddings", 8192
        )

        proj_cls = Linear4bit if hasattr(config, "quantization_config") else torch.nn.Linear

        self.q_proj = proj_cls(self.hidden_size, self.hidden_size, bias=use_bias)
        self.k_proj = proj_cls(self.hidden_size, self.hidden_size, bias=use_bias)
        self.v_proj = proj_cls(self.hidden_size, self.hidden_size, bias=use_bias)
        self.o_proj = proj_cls(self.hidden_size, self.hidden_size, bias=use_bias)

        # Run-time KV cache (allocated lazily on first decode step)
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.register_buffer("cache_index", torch.zeros(1, dtype=torch.long), persistent=False)

    # keep pretrained weights when we swap modules
    @torch.no_grad()
    def copy_from(self, other_attn: torch.nn.Module) -> None:
        """
        Shallow-copies the projection sub-modules from an existing
        attention layer (useful when we replace HF’s stock module
        with PagedAttention but want to preserve the weights).
        """
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, getattr(other_attn, name))
        # remove the original module from the parameter list so that
        # optimizer / LoRA see only one copy
        other_attn.__dict__.clear()

    def forward(
        self,
        hidden_states: torch.FloatTensor,                      # (B, T, D)
        attention_mask: Optional[torch.Tensor] = None,         # (B, 1, T, S) in HF
        position_ids: Optional[torch.LongTensor] = None,       # ignore
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:

        bsz, tgt_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # (B, T, H, Hd)  ->  (B, H, T, Hd)
        def _reshape(x, n_heads):
            return x.view(bsz, tgt_len, n_heads, self.head_dim).transpose(1, 2)

        q = _reshape(q, self.num_heads)
        k = _reshape(k, self.num_key_value_heads)
        v = _reshape(v, self.num_key_value_heads)

        # append to / concatenate KV cache if provided
        has_past = (
            past_key_value is not None
            and len(past_key_value) == 2
            and all(t is not None for t in past_key_value)
        )
        if has_past:
            past_k, past_v = past_key_value          # (B, n_kv, S, Hd)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        kv_seq_len = k.size(2)

        # save cache for decoding
        new_past_kv = None
        if use_cache:
            new_past_kv = (k, v)

        # build mask
        attn_mask = None
        if attention_mask is not None:
            # (B, S)  -> (B, 1, 1, S)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]

            # keep / pad so the last dim == kv_seq_len
            s = attention_mask.size(-1)
            if s < kv_seq_len:                                   # pad with 0 (no mask)
                pad = torch.zeros(
                    attention_mask.shape[:-1] + (kv_seq_len - s,),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, pad], dim=-1)
            elif s > kv_seq_len:
                attention_mask = attention_mask[..., :kv_seq_len]

            # broadcast over heads & queries. H/Q singleton dims
            if attention_mask.size(1) == 1 and self.num_heads > 1:
                attention_mask = attention_mask.expand(bsz, 1, tgt_len, kv_seq_len)

            attn_mask = attention_mask                       # final (B,1,Q,K) tensor

        # scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scale,
        )                                       # (B, H, T, Hd)

        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, H, Hd)
        attn_output = attn_output.view(bsz, tgt_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        attn_probs = None
        if output_attentions:
            score = (q * self.scale) @ k.transpose(2, 3)        # (B, H, T, S)
            if attention_mask is not None:
                score = score + attention_mask
            attn_probs = score.softmax(dim=-1)

        if output_attentions and use_cache:
            return (attn_output, attn_probs, (k, v))
        elif output_attentions:
            return (attn_output, attn_probs)
        elif use_cache:
            return (attn_output, (k, v))
        else:
            # HF still expects a second dummy element when both flags are False
            return (attn_output, None)

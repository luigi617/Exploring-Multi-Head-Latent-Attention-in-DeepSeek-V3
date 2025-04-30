import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ----------  optional bitsandbytes ----------
try:
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
except ImportError:
    Linear4bit = None
    Linear8bitLt = Nones
# --------------------------------------------

# ---------- helper ------------------------------------------------------------
def get_linear(in_features: int,
               out_features: int,
               bias: bool,
               quantization: Optional[str] = None) -> nn.Module:
    """
    Returns an nn.Module implementing y = Wx (+b) under the requested quantization.
    quantization ∈ {"none", None, "8bit", "4bit", "dynamic"}.
    """
    if quantization is None or quantization.lower() in {"none", "fp32", ""}:
        return nn.Linear(in_features, out_features, bias=bias)

    if quantization.lower() in {"8bit", "int8"}:
        if Linear8bitLt is None:
            raise ImportError("bitsandbytes not available – cannot build 8-bit Linear")
        # By default Linear8bitLt keeps 16-bit accumulators; change dtype if desired
        return Linear8bitLt(in_features, out_features, bias=bias)

    if quantization.lower() in {"4bit", "int4"}:
        if Linear4bit is None:
            raise ImportError("bitsandbytes not available – cannot build 4-bit Linear")
        return Linear4bit(in_features, out_features, bias=bias)

    raise ValueError(f"Unrecognised quantization mode: {quantization!r}")
# ------------------------------------------------------------------------------

class MultiHeadLatentAttention(nn.Module):
    """
    Two-step latent attention (Perceiver-style) **with optional weight quantization**.
    """

    def __init__(self, config, *, quantization: Optional[str] = None):
        super().__init__()
        self.embed_dim  = config.hidden_size
        self.num_heads  = config.num_attention_heads
        self.head_dim   = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, \
            f"embed_dim ({self.embed_dim}) not divisible by num_heads ({self.num_heads})"

        # latent parameters
        self.num_latents = getattr(config, "num_latents", 64)
        self.latent_dim  = getattr(config, "latent_dim", self.embed_dim)
        self.latents     = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))

        # ---------- projections: latents ←→ tokens ----------
        # latents attend to tokens
        self.to_q_latent = get_linear(self.latent_dim, self.embed_dim, bias=False, quantization=quantization)
        self.to_k_token  = get_linear(self.embed_dim,  self.embed_dim, bias=False, quantization=quantization)
        self.to_v_token  = get_linear(self.embed_dim,  self.embed_dim, bias=False, quantization=quantization)
        self.out_latent  = get_linear(self.embed_dim,  self.latent_dim, bias=False, quantization=quantization)
        # tokens attend to *updated* latents
        self.to_q_token  = get_linear(self.embed_dim,  self.embed_dim, bias=False, quantization=quantization)
        self.to_k_latent = get_linear(self.latent_dim, self.embed_dim, bias=False, quantization=quantization)
        self.to_v_latent = get_linear(self.latent_dim, self.embed_dim, bias=False, quantization=quantization)
        self.out_token   = get_linear(self.embed_dim,  self.embed_dim, bias=False, quantization=quantization)

        # dropout
        p = getattr(config, "attention_probs_dropout_prob", 0.0)
        self.dropout = nn.Dropout(p)

    # -------- convenience ------------------------------------------------------
    def to_dynamic_int8(self):
        """
        Returns an **INT-8 dynamically-quantised** copy using PyTorch AO.
        Only nn.Linear layers are quantised; everything else stays FP-32.
        """
        import torch.ao.quantization as tq
        return tq.quantize_dynamic(
            self, {nn.Linear}, dtype=torch.qint8, inplace=False
        )
    # --------------------------------------------------------------------------

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                *args, **kwargs) -> torch.Tensor:
        """
        hidden_states : [B, S, D]
        attention_mask: broadcastable to [B, 1, S, S] (optional)
        """
        x = hidden_states
        B, S, _ = x.shape

        # ---------- step 1: latents attend to tokens ----------
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)          # [B, L, latent_dim]
        Ql = self.to_q_latent(lat).view(B, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        Kt = self.to_k_token(x).view(B, S,              self.num_heads, self.head_dim).transpose(1, 2)
        Vt = self.to_v_token(x).view(B, S,              self.num_heads, self.head_dim).transpose(1, 2)

        scores  = (Ql @ Kt.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn    = self.dropout(F.softmax(scores, dim=-1))

        lat_ctx = (attn @ Vt).transpose(1, 2).reshape(B, self.num_latents, self.embed_dim)
        lat_ctx = self.out_latent(lat_ctx)

        # ---------- step 2: tokens attend to updated latents ----------
        Qt = self.to_q_token(x   ).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        Kl = self.to_k_latent(lat_ctx).view(B, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        Vl = self.to_v_latent(lat_ctx).view(B, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)

        scores2 = (Qt @ Kl.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn2   = self.dropout(F.softmax(scores2, dim=-1))

        tok_ctx = (attn2 @ Vl).transpose(1, 2).reshape(B, S, self.embed_dim)
        return self.out_token(tok_ctx)

    # -------- utility for weight-tying from an existing module -----------------
    @classmethod
    def from_existing(cls,
                      orig_attn: nn.Module,
                      *,
                      quantization: Optional[str] = None,
                      **kw) -> "MultiHeadLatentAttention":
        """
        Clone keys/values/output from a pretrained self-attention layer.
        """
        config  = getattr(orig_attn, "config", orig_attn)
        inst    = cls(config, quantization=quantization, **kw)
        with torch.no_grad():
            inst.to_k_token.weight.copy_(orig_attn.k_proj.weight)
            inst.to_v_token.weight.copy_(orig_attn.v_proj.weight)
            inst.out_token.weight.copy_(orig_attn.out_proj.weight)
        return inst

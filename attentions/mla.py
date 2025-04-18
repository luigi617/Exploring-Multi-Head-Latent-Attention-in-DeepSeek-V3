import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MultiHeadLatentAttention(nn.Module):
    """
    A two-step latent attention mechanism:
      1. Latent queries cross-attend over token keys/values.
      2. Tokens query updated latents to produce token outputs.

    Args:
        config: Transformer config with attributes:
          - hidden_size (int)
          - num_attention_heads (int)
          - attention_probs_dropout_prob (float, optional)
          - num_latents (int, optional)
          - latent_dim (int, optional)
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, (
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        )
        # latent setup
        self.num_latents = getattr(config, "num_latents", 64)
        self.latent_dim = getattr(config, "latent_dim", self.embed_dim)
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
        
        # projections: latents -> tokens
        self.to_q_latent = nn.Linear(self.latent_dim, self.embed_dim, bias=False)
        self.to_k_token  = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.to_v_token  = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_latent  = nn.Linear(self.embed_dim, self.latent_dim, bias=False)
        
        # projections: tokens -> latents
        self.to_q_token  = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.to_k_latent = nn.Linear(self.latent_dim, self.embed_dim, bias=False)
        self.to_v_latent = nn.Linear(self.latent_dim, self.embed_dim, bias=False)
        self.out_token   = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
        # dropout
        p = getattr(config, "attention_probs_dropout_prob", 0.0)
        self.dropout = nn.Dropout(p)

    @classmethod
    def from_existing(cls, orig_attn: nn.Module, num_latents: Optional[int] = None) -> "MultiHeadLatentAttention":
        """
        Create from a pretrained self-attention module, copying its k,v,out weights.
        """
        config = getattr(orig_attn, 'config', orig_attn)
        module = cls(config)
        with torch.no_grad():
            # copy keys, values, output projections
            module.to_k_token.weight.copy_(orig_attn.k_proj.weight)
            module.to_v_token.weight.copy_(orig_attn.v_proj.weight)
            module.out_token.weight.copy_(orig_attn.out_proj.weight)
        return module

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """
        hidden_states: [batch, seq_len, embed_dim]
        attention_mask: optional mask broadcastable to [batch, heads, seq_len, seq_len]

        Returns:
            out: [batch, seq_len, embed_dim]
        """
        x = hidden_states
        batch_size, seq_len, _ = x.size()
        # Step 1: latents cross-attend to tokens
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, latent_dim]
        Q_lat = self.to_q_latent(latents)
        K_tok = self.to_k_token(x)
        V_tok = self.to_v_token(x)

        # reshape to heads
        Q_lat = Q_lat.view(batch_size, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        K_tok = K_tok.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V_tok = V_tok.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q_lat @ K_tok.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        latent_context = attn @ V_tok  # [B, heads, L, head_dim]
        latent_context = latent_context.transpose(1, 2).contiguous().view(batch_size, self.num_latents, self.embed_dim)
        latent_context = self.out_latent(latent_context)

        # Step 2: tokens cross-attend to updated latents
        Q_tok = self.to_q_token(x)
        K_lat = self.to_k_latent(latent_context)
        V_lat = self.to_v_latent(latent_context)

        Q_tok = Q_tok.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_lat = K_lat.view(batch_size, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        V_lat = V_lat.view(batch_size, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)

        scores2 = (Q_tok @ K_lat.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn2 = F.softmax(scores2, dim=-1)
        attn2 = self.dropout(attn2)

        token_context = attn2 @ V_lat  # [B, heads, S, head_dim]
        token_context = token_context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_token(token_context)
        return out

import torch
import math
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, hidden_states, attention_mask=None):
        return self.attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=attention_mask,
            need_weights=False
        )[0]
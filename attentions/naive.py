import torch
import math

class NaiveAttention(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Split into heads
        q = q.view(*q.shape[:2], self.num_heads, self.head_dim)
        k = k.view(*k.shape[:2], self.num_heads, self.head_dim)
        v = v.view(*v.shape[:2], self.num_heads, self.head_dim)

        # Attention scores
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores += attention_mask
            
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        output = output.contiguous().view(*output.shape[:2], -1)
        
        return self.o_proj(output), attn_weights
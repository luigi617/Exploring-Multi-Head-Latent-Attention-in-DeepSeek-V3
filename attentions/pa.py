import torch
import math
class PagedAttention(torch.nn.Module):
    def __init__(self, config, block_size: int = 64):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.block_size = block_size
        
        self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Split into blocks
        bs, seq_len, _ = q.shape
        q = q.view(bs, seq_len // self.block_size, self.block_size, self.num_heads, self.head_dim)
        k = k.view(bs, seq_len // self.block_size, self.block_size, self.num_heads, self.head_dim)
        v = v.view(bs, seq_len // self.block_size, self.block_size, self.num_heads, self.head_dim)
        
        # Process blocks sequentially
        outputs = []
        for i in range(q.size(1)):
            q_block = q[:, i]
            k_block = k[:, :i+1]
            v_block = v[:, :i+1]
            
            scores = torch.einsum("bqhgd,bkhgd->bhqk", q_block, k_block) / math.sqrt(self.head_dim)
            attn_weights = torch.softmax(scores, dim=-1)
            block_output = torch.einsum("bhqk,bkhgd->bqhg", attn_weights, v_block)
            outputs.append(block_output)
            
        output = torch.cat(outputs, dim=1)
        return self.o_proj(output.contiguous().view(bs, seq_len, -1))
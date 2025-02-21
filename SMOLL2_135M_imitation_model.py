import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer  # Import AutoTokenizer


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x):
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return x * emb.unsqueeze(0).unsqueeze(0)  # Apply rotary embeddings


class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads=8, num_kv_heads=2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads  # Number of query heads
        self.num_kv_heads = num_kv_heads  # Number of key-value heads
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, dim)  # Queries for all heads
        self.k_proj = nn.Linear(dim, dim * num_kv_heads // n_heads)  # Reduced KV
        self.v_proj = nn.Linear(dim, dim * num_kv_heads // n_heads)  # Reduced KV
        self.o_proj = nn.Linear(dim, dim)  # Output projection
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Compute queries (for all heads)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Compute keys and values (only for num_kv_heads)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        v = v.transpose(1, 2)
        
        # Expand keys and values to match query heads
        expand_factor = self.n_heads // self.num_kv_heads
        k = k.repeat_interleave(expand_factor, dim=1)  # Expand kv heads to match query heads
        v = v.repeat_interleave(expand_factor, dim=1)
        
        # Apply rotary embeddings
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, v)
        
        x = x.transpose(1, 2).contiguous()  # [batch, seq_len, n_heads, head_dim]
        x = x.view(batch_size, seq_len, -1)  # Reshape to original dimension
        
        return self.o_proj(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, num_kv_heads, hidden_dim):
        super().__init__()
        self.attn = SelfAttention(dim, n_heads, num_kv_heads)
        self.ff = FeedForward(dim, hidden_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class GQA_LLM(nn.Module):
    def __init__(self, vocab_size, dim, n_heads, num_kv_heads, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, num_kv_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.output_layer = nn.Linear(dim, vocab_size)

    def generate_causal_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, x, targets=None):
        mask = self.generate_causal_mask(x.shape[1], x.device)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        logits = self.output_layer(x)  # (B, T, vocab_size)

        if targets is not None:
            # Calculate loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits.view(B, T, C), loss
        
        return logits  # If no targets provided, just return logits


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")

        # Load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        tokens = self.tokenizer.encode(text)  # Use the same tokenizer as in train.py
        self.tokens = torch.tensor(tokens)
        print(f'Loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # State
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # Inputs
        y = (buf[1:]).view(B, T)    # Targets
        # Advance the position in the tensor
        self.current_position += B * T
        # If loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


    

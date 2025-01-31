import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Ensure dim is even
        assert dim % 2 == 0, "Dimension must be divisible by 2"
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        n = x.shape[-1]
        half_n = n // 2  # Only rotate half the dimensions
        
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()[:, None, :]
        sin_emb = emb.sin()[:, None, :]
        
        x1, x2 = x.chunk(2, dim=-1)
        # Ensure dimensions match before operations
        x1 = x1[..., :half_n]
        x2 = x2[..., :half_n]
        cos_emb = cos_emb[..., :half_n]
        sin_emb = sin_emb[..., :half_n]
        
        return torch.cat((x1 * cos_emb - x2 * sin_emb, x1 * sin_emb + x2 * cos_emb), dim=-1)

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.n_heads = 8
        self.head_dim = dim // 8
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply rotary embeddings
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous()  # [batch, seq_len, n_heads, head_dim]
        x = x.view(batch_size, seq_len, -1)
        
        return self.o_proj(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.self_attn = SelfAttention(dim)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = self.ln1(x)
        x = self.self_attn(x) + x
        x = self.ln2(x)
        x = self.feed_forward(x) + x
        return x

class LlamaModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, hidden_dim):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, hidden_dim) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(dim)
        
    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        return self.ln_final(x)

class LlamaForCausalLM(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, hidden_dim):
        super().__init__()
        print(f"Initializing model with vocab_size: {vocab_size}")
        self.model = LlamaModel(vocab_size, dim, num_layers, hidden_dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, input_ids):
        x = self.model(input_ids)
        logits = self.lm_head(x)
        return logits

    def generate(self, input_ids, max_length=5, temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                # Get the last token's output
                logits = self(input_ids[:, -1:])
                # Apply temperature scaling
                logits = logits / temperature
                # Get probabilities
                probs = torch.softmax(logits[:, -1], dim=-1)
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                # Append to input
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, x) -> torch.Tensor:
        B, T, C = x.shape

        Q = self.query(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.1)
        attention = attention.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(attention)
    

class FeedForward(nn.Module):
    def __init__(self, d_model:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class Encoder(nn.Module):
    def __init__(self, d_model:int, num_heads:int) -> None:
        super().__init__()
        self.attn = MultiheadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x

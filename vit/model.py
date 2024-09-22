import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .modules import Encoder


class ViT(nn.Module):
    def __init__(self, 
            n_classes:int,
            patch_size:int, 
            img_size:Tuple[int, int],
            d_model:int = 512,
            num_layers:int = 6,
            num_heads:int = 8,
            ) -> None:
        super().__init__()

        H, W = img_size
        assert (H * W) % (patch_size ** 2) == 0, f"We cannot make equal patches with {patch_size=} and {img_size=}"
        N = H * W // patch_size**2

        self.patch_layer = nn.Unfold(
            kernel_size = (patch_size, patch_size), 
            stride = (patch_size, patch_size)
        )
        self.linear_proj = nn.Linear(3 * patch_size * patch_size, d_model)
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.max_seq_len = N 
        self.pe = nn.Embedding(self.max_seq_len, d_model) # +1 for class token
        self.encoders = nn.Sequential(
            *[Encoder(d_model, num_heads) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_proj(x)
        x = self.patch_layer(x)
        B, T, C = x.shape

        if T > self.max_seq_len:
            _T = torch.arange(self.max_seq_len, device=x.device).view(1, 1, -1).float()
            T = F.interpolate(_T, size=T, mode='nearest')
            T = T.view(-1).long()
            pe = self.pe(T)
        else:
            pe = self.pe(torch.arange(T, device=x.device))

        x += pe
        x = torch.cat([self.class_token, x], dim=1)
        x = self.encoders(x)
        x = self.ln(x)
        x = self.fc(x)
        return x


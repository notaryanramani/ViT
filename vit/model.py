import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from .modules import Encoder
from huggingface_hub import PyTorchModelHubMixin


class ViT(
    nn.Module, 
    PyTorchModelHubMixin,
    ):
    """
    Vision Transformer model for Classification tasks.

    Args:
        n_classes: int
            Number of classes in the classification task.

        patch_size: int
            Size of the patches in the image.

        img_size: Union[int, Tuple[int, int]]
            Size of the input image.

        d_model: int, default=512
            Dimension of the model.

        num_layers: int, default=6
            Number of encoder layers.

        num_heads: int, default=8
            Number of attention heads.
    """
    def __init__(self, 
            n_classes:int,
            patch_size:int, 
            img_size: Union[int, Tuple[int, int]],
            d_model:int = 512,
            num_layers:int = 6,
            num_heads:int = 8,
            ) -> None:
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
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
        self.pe = nn.Embedding(self.max_seq_len, d_model) 
        self.encoders = nn.Sequential(
            *[Encoder(d_model, num_heads) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)
    
    def forward(self, x: torch.Tensor, get_logits:bool=False) -> torch.Tensor:
        x = self.patch_layer(x).transpose(1, 2).contiguous()
        x = self.linear_proj(x)
        B, T, C = x.shape

        if T > self.max_seq_len:
            _T = torch.arange(self.max_seq_len, device=x.device).view(1, 1, -1).float()
            T = F.interpolate(_T, size=T, mode='nearest')
            T = T.view(-1).long()
            pe = self.pe(T)
        else:
            pe = self.pe(torch.arange(T, device=x.device))

        x += pe
        x = torch.cat([self.class_token.repeat(B, 1, 1), x], dim=1)
        x = self.encoders(x)
        x = self.ln(x)
        if get_logits:
            return x
        clf = self.fc(x[:, 0, :])
        return clf
    
    def predict(self, x: torch.Tensor) -> torch.Tensor: 
        probs = self(x, get_logits=False)
        return torch.argmax(probs, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor: 
        probs = F.softmax(self(x, get_logits=True), dim=-1)
        return probs

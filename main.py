import torch
import torch.nn.functional as F
from torch.optim import AdamW
from vit.utils import DataLoader, get_lr
from vit.model import ViT
from dotenv import load_dotenv
import os
load_dotenv()

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
m = ViT(n_classes=10, patch_size=4, img_size=(32, 32))
m.to(device)
lr = 1e-6
min_lr = lr
opt = AdamW(m.parameters(), lr=lr)

steps = 20000
dl = DataLoader()

for step in range(steps):
    lr = get_lr(step, lr, min_lr=min_lr, total_steps=steps)
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    x, y = dl.get_batch()
    x, y = x.to(device), y.to(device)
    logits = m(x)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if not step % 1:
        print(f'Step: {step} | Loss: {loss.item()}')
        m.push_to_hub("ViT-cifar10", token=os.getenv('hf_token'), branch='v1')

m.push_to_hub("ViT-cifar10", token=os.getenv('hf_token'), branch='main')
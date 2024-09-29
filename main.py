import torch
import torch.nn.functional as F
from torch.optim import AdamW
from vit.utils import DataLoader, get_lr, scale
from vit.model import ViT
from dotenv import load_dotenv
import os
from huggingface_hub import create_branch
from tqdm import tqdm
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

steps = 4000
push_to_hub = 1000 # push to hub every 1000 steps
val_step = 100 # validate every 100 steps
dl = DataLoader()

train_losses = []
val_losses = []

num_parameters = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"Number of parameters: {num_parameters}")

previous_val_avg = 0
pb = tqdm(range(steps))
for step in pb:
    lr = get_lr(step, lr, min_lr=min_lr, warmup_steps=800, total_steps=steps)
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    x, y = dl.get_batch(batch_size=64)
    x = scale(x)
    x, y = x.to(device), y.to(device)
    logits = m(x)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    pb.set_postfix(loss=loss.item(), val=previous_val_avg)
    train_losses.append(loss.item())
    if not (step + 1) % push_to_hub:
      m.push_to_hub('ViT-cifar10', token=os.environ['hf_token'], branch='v1')
    if not step % val_step:
      batch_loss = 0 
      m.eval()
      with torch.no_grad():
        for i in range(32):
          x, y = dl.get_test_batch(batch_size=64)
          x = scale(x)
          x, y = x.to(device), y.to(device)
          logits = m(x)
          loss = F.cross_entropy(logits, y)
          batch_loss += loss.item()
      previous_val_avg = batch_loss / 32
      val_losses.append(previous_val_avg)
      m.train()
m.push_to_hub('ViT-cifar10', token=os.environ['hf_token'], branch='main')
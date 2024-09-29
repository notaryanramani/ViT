import numpy as np
import pickle
import torch


def get_lr(step: int, 
           lr: float,
           max_lr:float = 3e-4, 
           min_lr:float = 1e-6, 
           warmup_steps:int = 2000, 
           total_steps:int = 10000) -> float:
    
    if step < warmup_steps:
        lr += (max_lr - min_lr) / warmup_steps
        return lr
    else:
        fact = (max_lr - min_lr) / (total_steps - warmup_steps)
        lr -= fact
        return lr
    
def scale(x: torch.Tensor) -> torch.Tensor:
    return (x / 127.5) - 1 # returns x in range [-1, 1]

class DataLoader:
    def __init__(self):
        def get_batch(filename):
            with open(filename, 'rb') as f:
                _dict = pickle.load(f, encoding='bytes')
            return _dict[b'data'], _dict[b'labels']
        
        data = []
        labels = []
        for i in range(1, 6):
            filename = f'cifar-10-batches-py/data_batch_{i}'
            _data, _labels = get_batch(filename)
            data.append(_data)
            labels.extend(_labels)

        data = np.concatenate(data)
        data = data.reshape(-1, 3, 32, 32)
        self.data = torch.from_numpy(data).float()
        self.labels = torch.tensor(labels)

        test_filename = 'cifar-10-batches-py/test_batch'
        self.test_data, self.test_labels = get_batch(test_filename)
    
    def get_batch(self, batch_size:int = 8, overfit:bool = False):
        if overfit:
            idx = torch.arange(batch_size)
            return self.data[idx], self.labels[idx]
        idx = torch.randint(0, len(self.data), (batch_size,))
        return self.data[idx], self.labels[idx]
    
    def get_test_batch(self, batch_size:int = 8):
        idx = torch.randint(0, len(self.test_data), (batch_size,))
        return self.test_data[idx], self.test_labels[idx]
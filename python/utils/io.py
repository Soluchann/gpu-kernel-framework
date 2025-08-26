import numpy as np
import torch

# python/utils/io.py
import torch
import numpy as np

# utils/io.py
def save_fp16(data, path):
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    data.astype(np.float16).tofile(path)

def save_bf16(data, path):
    if torch.is_tensor(data):
        data = data.view(torch.uint16).cpu().numpy()
    data.tofile(path)

def load_fp16(path, shape):
    return np.fromfile(path, dtype=np.float16).reshape(shape)

def load_bf16(path, shape):
    return np.fromfile(path, dtype=np.uint16).reshape(shape)
    
def numpy_to_torch(arr, device='cuda'):
    return torch.from_numpy(arr).to(device).half()
  
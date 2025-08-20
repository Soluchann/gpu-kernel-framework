import numpy as np
import torch

def load_bin(path, shape, dtype=np.float16):
    return np.fromfile(path, dtype=dtype).reshape(shape)

def save_bin(data, path):
    if torch.is_tensor(data):
        data = data.cpu().numpy() 
    data.astype(np.float16).tofile(path)

def numpy_to_torch(arr, device='cuda'):
    return torch.from_numpy(arr).to(device).half()
  
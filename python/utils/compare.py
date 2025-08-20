# python/utils/compare.py
import torch
import numpy as np

def compare_tensors(a, b, rtol=1e-2, atol=1e-3):
    if torch.is_tensor(a):
        a = a.cpu().numpy()
    if torch.is_tensor(b):
        b = b.cpu().numpy()
    return np.allclose(a, b, rtol=rtol, atol=atol), np.max(np.abs(a - b))
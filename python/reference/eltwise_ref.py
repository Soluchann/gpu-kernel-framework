import torch

def eltwise_add(a, b):
    return a + b

def relu(x):
    return torch.maximum(x, torch.zeros_like(x))

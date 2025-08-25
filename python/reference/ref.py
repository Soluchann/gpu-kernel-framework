# python/reference/ref.py
import torch
import torch.nn.functional as F

def eltwise_add_bf16(a, b):
    return a + b

def matmul_bf16(a, b):
    return torch.matmul(a, b)

def conv2d(input, weight, stride=1, padding=0):
    return F.conv2d(input, weight, stride=stride, padding=padding)

def eltwise_add(a, b):
    return a + b

def matmul_fp16(a, b):
    return torch.matmul(a, b)

def relu(x):
    return torch.maximum(x, torch.zeros_like(x))
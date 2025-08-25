# kernels/matmul_fp16/tests/single_element/test.py
import torch

def generate_case():
    M, N, K = 1, 1, 1
    A = torch.tensor([[3.14]], dtype=torch.half)
    B = torch.tensor([[2.86]], dtype=torch.half)
    Out = torch.matmul(A, B)
    return {
        "name": "single_element",
        "shape": (M, N, K),
        "inputs": {"A": A, "B": B},
        "expected": Out,
        "path": "./"
    }
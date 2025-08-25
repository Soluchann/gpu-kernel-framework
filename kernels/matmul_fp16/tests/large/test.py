# kernels/matmul_fp16/tests/large/test.py
import torch

def generate_case():
    M, N, K = 16, 16, 16
    A = torch.randn((16, 16), dtype=torch.half)
    B = torch.randn((16, 16), dtype=torch.half)
    Out = torch.matmul(A, B)
    return {
        "name": "large",
        "shape": (M, N, K),
        "inputs": {"A": A, "B": B},
        "expected": Out,
        "path": "./"
    }
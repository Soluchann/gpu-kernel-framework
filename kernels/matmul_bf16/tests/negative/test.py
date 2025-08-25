# kernels/matmul/tests/negative/test.py
import torch

def generate_case():
    M, N, K = 2, 2, 3
    A = torch.tensor([[-1.5, 2.0, -3.5],
                      [4.0, -5.5, 6.0]], dtype=torch.bfloat16)
    B = torch.tensor([[1.0, -2.0],
                      [-3.0, 4.0],
                      [5.0, -6.0]], dtype=torch.bfloat16)
    Out = torch.matmul(A, B)
    return {
        "name": "negative",
        "shape": (M, N, K),
        "inputs": {"A": A, "B": B},
        "expected": Out,
        "path": "./"
    }
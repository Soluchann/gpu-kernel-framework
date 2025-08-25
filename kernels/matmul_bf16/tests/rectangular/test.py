# kernels/matmul/tests/rectangular/test.py
import torch

def generate_case():
    M, N, K = 3, 5, 2
    A = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]], dtype=torch.bfloat16)
    B = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=torch.bfloat16)
    Out = torch.matmul(A, B)
    return {
        "name": "rectangular",
        "shape": (M, N, K),
        "inputs": {"A": A, "B": B},
        "expected": Out,
        "path": "./"
    }
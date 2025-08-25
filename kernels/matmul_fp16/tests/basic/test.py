# kernels/matmul_fp16/tests/basic/test.py
import torch

def generate_case():
    M, N, K = 4, 4, 4
    A = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [1.0, 3.0, 5.0, 7.0],
                      [2.0, 4.0, 6.0, 8.0]], dtype=torch.half)
    B = torch.tensor([[1.0, 0.0, 2.0, 0.0],
                      [0.0, 1.0, 0.0, 2.0],
                      [1.0, 0.0, 2.0, 0.0],
                      [0.0, 1.0, 0.0, 2.0]], dtype=torch.half)
    Out = torch.matmul(A, B)
    return {
        "name": "basic",
        "shape": (M, N, K),
        "inputs": {"A": A, "B": B},
        "expected": Out,
        "path": "./"
    }
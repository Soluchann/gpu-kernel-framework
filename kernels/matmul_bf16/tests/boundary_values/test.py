# kernels/matmul/tests/boundary_values/test.py
import torch

def generate_case():
    M, N, K = 2, 2, 2
    # Test with values near bf16 limits
    A = torch.tensor([[3.3895e+38, -3.3895e+38],
                      [1.0, -1.0]], dtype=torch.bfloat16)
    B = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0]], dtype=torch.bfloat16)
    Out = torch.matmul(A, B)
    return {
        "name": "boundary_values",
        "shape": (M, N, K),
        "inputs": {"A": A, "B": B},
        "expected": Out,
        "path": "./"
    }
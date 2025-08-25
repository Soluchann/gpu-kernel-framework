# kernels/eltwise_add_fp16/tests/boundary_values/test.py
import torch

def generate_case():
    shape = (8,)
    # Test with values near fp16 limits
    A = torch.tensor([65500.0, -65500.0, 0.000061035, -0.000061035, 1.0, -1.0, 0.0, 1000.0], dtype=torch.half)
    B = torch.tensor([1.0, -1.0, 0.000061035, -0.000061035, -1.0, 1.0, 0.0, -1000.0], dtype=torch.half)
    Out = A + B
    return {
        "name": "boundary_values",
        "shape": shape,
        "inputs": {"A": A, "B": B},
        "expected": Out,
        "path": "./"
    }
    
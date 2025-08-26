import torch

def generate_case():
    shape = (64, 64)
    # Use bfloat16 instead of half
    A = torch.randn(shape).bfloat16()
    B = torch.randn(shape).bfloat16()
    Out = A + B
    return {
        "name": "test1",
        "shape": shape,
        "inputs": {"A": A, "B": B},
        "expected": Out,
        "path": "./"  # Will be overridden by generator
    }
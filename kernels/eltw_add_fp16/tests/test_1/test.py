# kernels/eltwise_add_fp16/tests/test1/test.py
import torch

def generate_case():
    shape = (64, 64)
    A = torch.randn(shape).half()
    B = torch.randn(shape).half()
    Out = A + B
    return {
        "name": "test1",
        "shape": shape,
        "inputs": {"A": A, "B": B},
        "expected": Out,
        "path": "./"  # Will be overridden by generator
    }
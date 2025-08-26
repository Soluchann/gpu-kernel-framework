# kernels/eltwise_add_fp16/tests/edge_case/test.py
import torch

def generate_case():
    shape = (1, 1)
    A = torch.zeros(1, 1).half()
    B = torch.ones(1, 1).half()
    Out = A + B
    return {
        "name": "edge_case",
        "shape": shape,
        "inputs": {"A": A, "B": B},
        "expected": Out,
        "path": "./"
    }
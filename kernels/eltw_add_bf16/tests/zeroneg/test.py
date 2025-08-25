import torch

def generate_case():
    shape = (256, 128)
    # Test with zeros and negative numbers
    A = torch.zeros(shape, dtype=torch.bfloat16)
    B = torch.full(shape, -2.5, dtype=torch.bfloat16)
    Out = A + B  # Should be all -2.5
    return {
        "name": "zeros_and_negatives",
        "shape": shape,
        "inputs": {"A": A, "B": B},
        "expected": Out,
        "path": "./"
    }
    
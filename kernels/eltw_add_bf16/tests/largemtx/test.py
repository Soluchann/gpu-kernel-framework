import torch

def generate_case():
    shape = (512, 512)  # Larger matrix
    # Use consistent seed for reproducibility
    torch.manual_seed(42)
    A = torch.randn(shape, dtype=torch.bfloat16)
    B = torch.randn(shape, dtype=torch.bfloat16)
    Out = A + B
    return {
        "name": "large_matrix",
        "shape": shape,
        "inputs": {"A": A, "B": B},
        "expected": Out,
        "path": "./"
    }
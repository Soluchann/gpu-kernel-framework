import torch

def generate_case():
    """
    Diagnostic test for A(16, 16) @ B(16, 16) using bfloat16.
    Here, input and output shapes are identical.
    """
    # Use a square matrix where input/output shapes match
    shape = (16, 16)

    A = torch.randn(shape).bfloat16()
    B = torch.randn(shape).bfloat16()
    expected_out = torch.matmul(A, B)

    return {
        "name": "square_matmul_diagnostic",
        "shape": shape,  # Shape is (16, 16)
        "inputs": {"A": A, "B": B},
        "expected": expected_out,
    }
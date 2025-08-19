import torch

def compare_tensors(a, b, rtol=1e-2, atol=1e-3):
    """
    Compare two tensors (on CPU or GPU)
    """
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol)
    else:
        a = torch.tensor(a)
        b = torch.tensor(b)
        return torch.allclose(a, b, rtol=rtol, atol=atol)

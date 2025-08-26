import torch
import torch.nn.functional as F

def generate_case():
    """
    Generates a test case with larger input dimensions for BF16 convolution.
    """
    params = {
        "N": 2, "C": 16, "H": 64, "W": 64,
        "K": 32, "R": 3, "S": 3,
        "stride": 1, "padding": 1
    }
    H_out = (params["H"] - params["R"] + 2 * params["padding"]) // params["stride"] + 1
    W_out = (params["W"] - params["S"] + 2 * params["padding"]) // params["stride"] + 1
    output_shape = (params["N"], params["K"], H_out, W_out)
    X = torch.randn(params["N"], params["C"], params["H"], params["W"]).bfloat16()
    Weights = torch.randn(params["K"], params["C"], params["R"], params["S"]).bfloat16()
    expected_out = F.conv2d(X, Weights, stride=params["stride"], padding=params["padding"])
    return {
        "name": "conv2d_bf16_large",
        "params": params,
        "inputs": {"X": X, "Weights": Weights},
        "shape": output_shape,
        "expected": expected_out,
    }

import torch
import torch.nn.functional as F

def generate_case():
    """
    Generates a test case with more input and output channels for BF16 convolution.
    """
    params = {
        "N": 1, "C": 64, "H": 16, "W": 16,
        "K": 128, "R": 1, "S": 1,
        "stride": 1, "padding": 0
    }
    H_out = (params["H"] - params["R"] + 2 * params["padding"]) // params["stride"] + 1
    W_out = (params["W"] - params["S"] + 2 * params["padding"]) // params["stride"] + 1
    output_shape = (params["N"], params["K"], H_out, W_out)
    X = torch.randn(params["N"], params["C"], params["H"], params["W"]).bfloat16()
    Weights = torch.randn(params["K"], params["C"], params["R"], params["S"]).bfloat16()
    expected_out = F.conv2d(X, Weights, stride=params["stride"], padding=params["padding"])
    return {
        "name": "conv2d_bf16_deeper",
        "params": params,
        "inputs": {"X": X, "Weights": Weights},
        "shape": output_shape,
        "expected": expected_out,
    }

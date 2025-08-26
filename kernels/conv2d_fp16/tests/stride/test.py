import torch
import torch.nn.functional as F

def generate_case():
    """
    Generates a test case with a stride of 2.
    """
    params = {
        "N": 1, "C": 8, "H": 32, "W": 32,
        "K": 16, "R": 3, "S": 3,
        "stride": 2, "padding": 1
    }
    H_out = (params["H"] - params["R"] + 2 * params["padding"]) // params["stride"] + 1
    W_out = (params["W"] - params["S"] + 2 * params["padding"]) // params["stride"] + 1
    output_shape = (params["N"], params["K"], H_out, W_out)
    X = torch.randn(params["N"], params["C"], params["H"], params["W"]).half()
    Weights = torch.randn(params["K"], params["C"], params["R"], params["S"]).half()
    expected_out = F.conv2d(X, Weights, stride=params["stride"], padding=params["padding"])
    return {
        "name": "conv2d_fp16_strided",
        "params": params,
        "inputs": {"X": X, "Weights": Weights},
        "shape": output_shape,
        "expected": expected_out,
    }

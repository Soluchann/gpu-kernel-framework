import torch
import torch.nn.functional as F

def generate_case():
    """
    Generates a test case for a simple FP16 2D convolution.
    """
    # Define convolution parameters
    params = {
        "N": 1, "C": 3, "H": 32, "W": 32,
        "K": 8, "R": 3, "S": 3,
        "stride": 1, "padding": 1
    }

    # Calculate output dimensions
    H_out = (params["H"] - params["R"] + 2 * params["padding"]) // params["stride"] + 1
    W_out = (params["W"] - params["S"] + 2 * params["padding"]) // params["stride"] + 1
    output_shape = (params["N"], params["K"], H_out, W_out)

    # Generate random input and weight tensors
    X = torch.randn(params["N"], params["C"], params["H"], params["W"]).half()
    Weights = torch.randn(params["K"], params["C"], params["R"], params["S"]).half()

    # Calculate the ground truth using PyTorch
    expected_out = F.conv2d(X, Weights, stride=params["stride"], padding=params["padding"])

    return {
        "name": "conv2d_fp16_small",
        "params": params,
        "inputs": {
            "X": X,
            "Weights": Weights
        },
        "shape": output_shape, # Legacy shape for display, main info is in params
        "expected": expected_out,
    }

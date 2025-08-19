import torch
import numpy as np
import kernel_lib  # This will be built by CMake + pybind11
from utils.io import load_bin, numpy_to_torch
from utils.compare import compare_tensors
from reference.eltwise_ref import eltwise_add

def float16_to_uint16(arr):
    """Convert float16 NumPy array to uint16 (bit reinterpretation)"""
    return np.frombuffer(arr.tobytes(), dtype=np.uint16)

def uint16_to_float16(arr, shape):
    """Convert uint16 back to float16"""
    return np.frombuffer(arr.tobytes(), dtype=np.float16).reshape(shape)

def test_eltwise_add():
    shape = (64, 64)

    # Fix path resolution
    import os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    input1_path = os.path.join(PROJECT_ROOT, "kernels", "eltwise", "tests", "input1.bin")
    input2_path = os.path.join(PROJECT_ROOT, "kernels", "eltwise", "tests", "input2.bin")

    # Load inputs
    A_np = load_bin(input1_path, shape)  # float16
    B_np = load_bin(input2_path, shape)  # float16

    # PyTorch reference
    A_pt = numpy_to_torch(A_np)
    B_pt = numpy_to_torch(B_np)
    expected = eltwise_add(A_pt, B_pt).cpu().numpy()  # float16

    # Convert to uint16 for C++ kernel
    def float16_to_uint16(arr):
        return np.frombuffer(arr.tobytes(), dtype=np.uint16)

    def uint16_to_float16(arr, shape):
        return np.frombuffer(arr.tobytes(), dtype=np.float16).reshape(shape)

    A_uint16 = float16_to_uint16(A_np.flatten())
    B_uint16 = float16_to_uint16(B_np.flatten())

    # Run C++ kernel
    actual_uint16 = kernel_lib.eltwise_add_fp16(A_uint16, B_uint16)

    # Convert back
    actual = uint16_to_float16(actual_uint16, shape)

    # Compare
    if compare_tensors(actual, expected, rtol=1e-2, atol=1e-3):
        print("✅ [eltwise_add] Test Passed")
        return True
    else:
        print("❌ [eltwise_add] Test Failed")
        print(f"   Max diff: {np.max(np.abs(actual - expected))}")
        return False

if __name__ == "__main__":
    test_eltwise_add()
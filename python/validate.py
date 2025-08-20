import torch
import numpy as np
import time
from tabulate import tabulate  # pip install tabulate
import kernel_lib  # Your CUDA kernel
from utils.io import load_bin, numpy_to_torch
from utils.compare import compare_tensors
from reference.eltwise_ref import eltwise_add

def float16_to_uint16(arr):
    return np.frombuffer(arr.tobytes(), dtype=np.uint16)

def uint16_to_float16(arr, shape):
    return np.frombuffer(arr.tobytes(), dtype=np.float16).reshape(shape)

# Store test results
results = []

def test_eltwise_add(shape):
    print(f"Running eltwise add test: {shape} FP16")

    import os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    input1_path = os.path.join(PROJECT_ROOT, "kernels", "eltwise", "tests", "input1.bin")
    input2_path = os.path.join(PROJECT_ROOT, "kernels", "eltwise", "tests", "input2.bin")

    A_np = load_bin(input1_path, shape)
    B_np = load_bin(input2_path, shape)

    A_pt = numpy_to_torch(A_np)
    B_pt = numpy_to_torch(B_np)

    # 🔹 PyTorch Reference (GPU)
    torch.cuda.synchronize()
    start_time = time.time()
    expected_pt = eltwise_add(A_pt, B_pt)
    torch.cuda.synchronize()
    torch_time = time.time() - start_time
    expected_np = expected_pt.cpu().numpy()

    # 🔹 CUDA Kernel
    A_uint16 = float16_to_uint16(A_np.flatten())
    B_uint16 = float16_to_uint16(B_np.flatten())

    torch.cuda.synchronize()
    start_time = time.time()
    actual_uint16 = kernel_lib.eltwise_add_fp16(A_uint16, B_uint16)
    torch.cuda.synchronize()
    kernel_time = time.time() - start_time

    actual_np = uint16_to_float16(actual_uint16, shape)

    # 🔍 Compare Results
    rtol = 1e-2
    atol = 1e-3
    passed, max_error = compare_tensors(actual_np, expected_np, rtol=rtol, atol=atol)

    status = "✅ Pass" if passed else "❌ Fail"
    results.append({
        "Test": "eltwise_add",
        "Shape": str(shape),
        "PyTorch Time (ms)": f"{torch_time * 1000:.3f}",
        "CUDA Time (ms)": f"{kernel_time * 1000:.3f}",
        "Max Error": f"{max_error:.6f}",
        "Status": status
    })
    return passed

def test_matmul():
    M, N, K = 64, 64, 64
    print(f"Running matmul test: ({M}x{K}) @ ({K}x{N}) FP16")

    import os
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    A_path = os.path.join(PROJECT_ROOT, "kernels", "matmul", "tests", "A.bin")
    B_path = os.path.join(PROJECT_ROOT, "kernels", "matmul", "tests", "B.bin")
    expected_path = os.path.join(PROJECT_ROOT, "kernels", "matmul", "tests", "expected_output.bin")

    A_np = load_bin(A_path, (M, K))
    B_np = load_bin(B_path, (K, N))
    expected_np = load_bin(expected_path, (M, N))

    A_pt = torch.from_numpy(A_np).to('cuda')
    B_pt = torch.from_numpy(B_np).to('cuda')

    # PyTorch reference
    torch.cuda.synchronize()
    start = time.time()
    computed_pt = torch.matmul(A_pt, B_pt)
    torch.cuda.synchronize()
    torch_time = time.time() - start
    computed_np = computed_pt.cpu().numpy()

    # CUDA kernel
    A_uint16 = float16_to_uint16(A_np.flatten())
    B_uint16 = float16_to_uint16(B_np.flatten())

    torch.cuda.synchronize()
    start = time.time()
    actual_uint16 = kernel_lib.matmul_fp16(A_uint16, B_uint16, M, N, K)
    torch.cuda.synchronize()
    kernel_time = time.time() - start

    actual_np = uint16_to_float16(actual_uint16, (M, N))

    # Compare
    passed = np.allclose(actual_np, expected_np, rtol=1e-2, atol=1e-2)
    max_error = np.max(np.abs(actual_np - expected_np))

    status = "✅ Pass" if passed else "❌ Fail"
    results.append({
        "Test": "matmul",
        "Shape": f"({M}x{K}) @ ({K}x{N})",
        "PyTorch Time (ms)": f"{torch_time * 1000:.3f}",
        "CUDA Time (ms)": f"{kernel_time * 1000:.3f}",
        "Max Error": f"{max_error:.6f}",
        "Status": status
    })
    return passed

if __name__ == "__main__":
    shapes = [(64, 64)]
    for shape in shapes:
        test_eltwise_add(shape)
    test_matmul()

    # Print final table
    print("\n📊 Test Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
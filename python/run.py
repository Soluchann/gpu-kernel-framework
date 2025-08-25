# python/run.py
import os
import sys
import importlib.util
import torch
import time
import numpy as np
from tabulate import tabulate
from utils.io import load_fp16, load_bf16
import argparse
import torch.nn.functional as F

# --- Project Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
KERNELS_DIR = os.path.join(PROJECT_ROOT, "kernels")

# --- Import Reference Implementations ---
from reference import ref

# --- Operator Abstraction ---

class Operator:
    """
    Base class for all operators.
    """
    def get_pytorch_baseline(self, a, b):
        raise NotImplementedError

    def get_kernel_call(self, kernel_lib, a, b, shape, expected_shape, kernel_name):
        raise NotImplementedError

class EltwiseAdd(Operator):
    """
    Element-wise addition operator.
    """
    def get_pytorch_baseline(self, a, b):
        return ref.eltwise_add(a, b)

    def get_kernel_call(self, kernel_lib, a, b, shape, expected_shape, kernel_name):
        if "fp16" in kernel_name:
            return kernel_lib.eltwise_add_fp16_cu(a, b)
        elif "bf16" in kernel_name:
            return kernel_lib.eltwise_add_bf16_cu(a, b)
        else:
            raise ValueError("Unsupported data type for EltwiseAdd")


class MatMul(Operator):
    """
    Matrix multiplication operator.
    """
    def get_pytorch_baseline(self, a, b):
        return ref.matmul_fp16(a, b) if a.dtype == torch.float16 else ref.matmul_bf16(a, b)

    def get_kernel_call(self, kernel_lib, a, b, shape, expected_shape, kernel_name):
        m, n, k = shape[0], expected_shape[1], shape[1]
        if "fp16" in kernel_name:
            return kernel_lib.matmul_fp16_cu(a, b, m, n, k)
        elif "bf16" in kernel_name:
            return kernel_lib.matmul_bf16_cu(a, b, m, n, k)
        else:
            raise ValueError("Unsupported data type for MatMul")

# --- Operator Registry ---

class Conv2D(Operator):
    def get_pytorch_baseline(self, inputs, params):
        return F.conv2d(inputs["X"], inputs["Weights"], stride=params["stride"], padding=params["padding"])

    def get_kernel_call(self, kernel_lib, inputs, params, kernel_name):
        p = params # shortcut
        h_out = (p["H"] - p["R"] + 2 * p["padding"]) // p["stride"] + 1
        w_out = (p["W"] - p["S"] + 2 * p["padding"]) // p["stride"] + 1
        output_shape = (p["N"], p["K"], h_out, w_out)
        out = np.zeros(output_shape, dtype=np.uint16)

        if "fp16" in kernel_name:
            kernel_lib.conv2d_fp16_cu(
                inputs["X"], inputs["Weights"], out,
                p["N"], p["C"], p["H"], p["W"],
                p["K"], p["R"], p["S"],
                p["stride"], p["padding"]
            )
        else:
            raise ValueError("Unsupported data type for Conv2D")
        return out


OPERATOR_REGISTRY = {
    "eltw_add": EltwiseAdd(),
    "matmul": MatMul(),
    "conv2d": Conv2D(),
}

# --- Data Handling ---

def get_data_handler(data_type):
    if data_type == "fp16":
        return {
            "load": load_fp16,
            "torch_dtype": torch.float16,
            "to_torch": lambda x: torch.from_numpy(x),
            "to_kernel": lambda x: np.frombuffer(x.tobytes(), dtype=np.uint16),
            "from_kernel": lambda x, shape: np.frombuffer(x.tobytes(), dtype=np.float16).reshape(shape),
            "atol": 1e-3,
            "rtol": 1e-2,
        }
    elif data_type == "bf16":
        return {
            "load": load_bf16,
            "torch_dtype": torch.bfloat16,
            "to_torch": lambda x: torch.from_numpy(x.astype(np.float32)),
            "to_kernel": lambda x: x.flatten(),
            # --- FIX: Reshape the output from the kernel ---
            "from_kernel": lambda x, shape: x.astype(np.float32).reshape(shape),
            "atol": 1e-2,
            "rtol": 1e-1,
        }
    else:
        raise ValueError(f"Unknown data type: {data_type}")

# --- Test Execution ---

results = []

def discover_and_run_tests(kernel_filter=None, test_filter=None):
    """
    Discover and run test cases.
    """
    global results
    results.clear()

    print(f"Discovering tests (kernel='{kernel_filter}', test='{test_filter}')\n")

    for kernel_name in sorted(os.listdir(KERNELS_DIR)):
        if kernel_filter and kernel_name != kernel_filter:
            continue

        kernel_path = os.path.join(KERNELS_DIR, kernel_name)
        if not os.path.isdir(kernel_path):
            continue

        tests_root = os.path.join(kernel_path, "tests")
        if not os.path.exists(tests_root):
            continue

        for test_name in sorted(os.listdir(tests_root)):
            if test_filter and test_name != test_filter:
                continue

            test_path = os.path.join(tests_root, test_name)
            if not os.path.isdir(test_path):
                continue

            test_script = os.path.join(test_path, "test.py")
            if not os.path.exists(test_script):
                continue

            run_test_case(kernel_name, test_name, test_path)

    return results


def run_test_case(kernel_name, test_name, test_path):
    try:
        # --- Setup ---
        operator_name, data_type = kernel_name.rsplit('_', 1)
        operator = OPERATOR_REGISTRY[operator_name]
        data_handler = get_data_handler(data_type)

        # --- Load Test Case ---
        spec = importlib.util.spec_from_file_location(f"{kernel_name}.{test_name}", os.path.join(test_path, "test.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        case = module.generate_case()
        
        print(f"Running: {kernel_name} / {test_name} [{case['shape']}]")

        # --- THIS IS THE MAIN CHANGE: HYBRID LOGIC FOR DIFFERENT OPERATORS ---
        if operator_name == "conv2d":
            # New, flexible logic for Conv2D
            params = case.get("params", {})
            inputs_np = {name: data_handler["load"](os.path.join(test_path, f"{name}.bin"), tensor.shape)
                         for name, tensor in case["inputs"].items()}
            expected = data_handler["load"](os.path.join(test_path, "expected_output.bin"), case["expected"].shape)
            inputs_pt = {name: data_handler["to_torch"](tensor).to('cuda').to(data_handler["torch_dtype"])
                         for name, tensor in inputs_np.items()}
            inputs_kernel = {name: data_handler["to_kernel"](tensor) for name, tensor in inputs_np.items()}

            # PyTorch Baseline
            torch.cuda.synchronize()
            start = time.time()
            ref_out = operator.get_pytorch_baseline(inputs_pt, params)
            torch.cuda.synchronize()
            torch_time = time.time() - start

            # Kernel Execution
            torch.cuda.synchronize()
            start = time.time()
            actual_kernel_out = operator.get_kernel_call(kernel_lib, inputs_kernel, params, kernel_name)
            torch.cuda.synchronize()
            kernel_time = time.time() - start

        else:
            # Original, simple logic for EltwiseAdd and MatMul
            shape = case["shape"]
            expected_shape = case["expected"].shape
            a = data_handler["load"](os.path.join(test_path, "A.bin"), shape)
            b = data_handler["load"](os.path.join(test_path, "B.bin"), shape)
            expected = data_handler["load"](os.path.join(test_path, "expected_output.bin"), expected_shape)

            # PyTorch Baseline
            a_pt = data_handler["to_torch"](a).to('cuda').to(data_handler["torch_dtype"])
            b_pt = data_handler["to_torch"](b).to('cuda').to(data_handler["torch_dtype"])
            torch.cuda.synchronize()
            start = time.time()
            ref_out = operator.get_pytorch_baseline(a_pt, b_pt)
            torch.cuda.synchronize()
            torch_time = time.time() - start

            # Kernel Execution
            a_kernel = data_handler["to_kernel"](a)
            b_kernel = data_handler["to_kernel"](b)
            torch.cuda.synchronize()
            start = time.time()
            actual_kernel_out = operator.get_kernel_call(kernel_lib, a_kernel, b_kernel, shape, expected_shape, kernel_name)
            torch.cuda.synchronize()
            kernel_time = time.time() - start

        # --- Compare (common logic for all operators) ---
        actual = data_handler["from_kernel"](actual_kernel_out, case["expected"].shape)
        passed = np.allclose(actual, expected, rtol=data_handler["rtol"], atol=data_handler["atol"])
        max_error = np.max(np.abs(actual - expected))

        results.append({
            "Kernel": kernel_name, "Test": test_name, "Shape": str(case['shape']),
            "PyTorch Time (ms)": f"{torch_time * 1000:.3f}",
            "Kernel Time (ms)": f"{kernel_time * 1000:.3f}",
            "Max Error": f"{max_error:.6f}",
            "Status": "✅ Pass" if passed else "❌ Fail"
        })

    except Exception as e:
        results.append({
            "Kernel": kernel_name,
            "Test": test_name,
            "Shape": "N/A",
            "PyTorch Time (ms)": "–",
            "Kernel Time (ms)": "–",
            "Max Error": "–",
            "Status": "❌ Error",
            "Error": str(e)[:100]
        })

# --- Main ---

def print_results(results):
    if not results:
        print("📭 No test cases were run.")
        return

    print("\n Test Results")
    print(tabulate(results, headers="keys", tablefmt="grid"))

    passed = sum(1 for r in results if "✅" in r["Status"])
    failed = len(results) - passed
    print(f"\n{passed} passed |{failed} failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run kernel tests")
    parser.add_argument("--k", "--kernel", dest="kernel_filter", help="Filter by kernel name (e.g., eltw_add_fp16)")
    parser.add_argument("--t", "--test", dest="test_filter", help="Filter by test case name (e.g., test_1)")
    parser.add_argument("--bld", "--build_dir", dest="build_dir", default="../build", help="Path to the build directory containing kernel_lib")

    args = parser.parse_args()

    build_bindings_path = os.path.join(os.path.abspath(args.build_dir), "bindings")
    sys.path.insert(0, build_bindings_path)

    try:
        import kernel_lib
    except ImportError as e:
        print(f"❌ Failed to import kernel_lib from {build_bindings_path}")
        print(e)
        sys.exit(1)

    results = discover_and_run_tests(args.kernel_filter, args.test_filter)
    print_results(results)

    if any("❌" in r["Status"] for r in results):
        sys.exit(1)
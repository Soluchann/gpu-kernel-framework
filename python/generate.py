# python/generate.py
import os
import sys
import importlib.util
import argparse
import torch
from utils.io import save_fp16, save_bf16

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

KERNELS_DIR = os.path.join(PROJECT_ROOT, "kernels")


def discover_test_dirs():
    test_dirs = []
    for kernel_name in sorted(os.listdir(KERNELS_DIR)):
        kernel_path = os.path.join(KERNELS_DIR, kernel_name)
        if not os.path.isdir(kernel_path) or kernel_name.startswith("__"):
            continue

        tests_root = os.path.join(kernel_path, "tests")
        if not os.path.isdir(tests_root):
            continue

        for test_name in sorted(os.listdir(tests_root)):
            test_path = os.path.join(tests_root, test_name)
            if not os.path.isdir(test_path):
                continue

            test_script = os.path.join(test_path, "test.py")
            if os.path.isfile(test_script):
                test_dirs.append((kernel_name, test_name, test_path, test_script))
    return test_dirs


def run_generate_and_save(selected_kernel=None, selected_test=None):
    print("⚙️ Generating test data...\n")
    test_dirs = discover_test_dirs()

    # Filter if user provided args
    if selected_kernel:
        test_dirs = [t for t in test_dirs if t[0] == selected_kernel]
        if selected_test:
            test_dirs = [t for t in test_dirs if t[1] == selected_test]

    if not test_dirs:
        print("❌ No matching test cases found.")
        return

    for kernel, test_name, test_path, script_path in test_dirs:
        print(f"📦 Processing: {kernel} / {test_name}")

        # Import test.py
        spec = importlib.util.spec_from_file_location(f"{kernel}.{test_name}", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "generate_case"):
            print(f"  ❌ {test_name}: No generate_case() function")
            continue

        case = module.generate_case()
        case["kernel"] = kernel
        case["test_name"] = test_name
        case["path"] = test_path  # Override path to local dir

        inputs = case["inputs"]
        expected = case["expected"]

        # Save inputs
        for name, tensor in inputs.items():
            filepath = os.path.join(test_path, f"{name}.bin")
            if "bf16" in kernel.lower():
                data = tensor.view(torch.uint16).cpu().numpy()
                save_bf16(data, filepath)
            else:
                save_fp16(tensor.cpu().numpy(), filepath)

        # Save expected
        expected_path = os.path.join(test_path, "expected_output.bin")
        if "bf16" in kernel.lower():
            data = expected.view(torch.uint16).cpu().numpy()
            save_bf16(data, expected_path)
        else:
            save_fp16(expected.cpu().numpy(), expected_path)

        print(f"    ✅ Generated {len(inputs)} inputs + expected_output.bin")

    print("\n✅ All test data generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test data bins for kernels")
    parser.add_argument("--kernel", type=str, help="Kernel name (e.g., eltw_add_fp16)")
    parser.add_argument("--test", type=str, help="Test case name (e.g., test_1)")
    args = parser.parse_args()

    run_generate_and_save(selected_kernel=args.kernel, selected_test=args.test)

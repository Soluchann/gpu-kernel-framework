# GPU Kernel Framework

A framework for developing and testing custom GPU kernels, with a focus on deep learning operations. This framework provides a streamlined process for building, testing, and benchmarking custom CUDA kernels against PyTorch baselines.

-----

## Features ✨

  * **Custom CUDA Kernels**: Implement high-performance GPU kernels for operations like matrix multiplication (`matmul`), element-wise addition (`eltw_add`), and 2D convolution (`conv2d`).
  * **Multiple Data Types**: Supports both `fp16` (half-precision) and `bf16` (bfloat16) data types.
  * **PyTorch Integration**: Kernels are exposed to Python through `pybind11`, allowing for easy integration with PyTorch.
  * **Automated Testing**: A Python-based testing framework (`run.py`) allows you to automatically discover and run test cases.
  * **Benchmarking**: The test runner provides performance comparisons between your custom kernels and PyTorch's native implementations.
  * **Bazel Build System**: Uses Bazel with Bzlmod for robust dependency management and builds.

-----

## Requirements 📜

  * **NVIDIA GPU**: A CUDA-enabled GPU is required.
  * **CUDA Toolkit**: Version 11.0 or newer.
  * **Python**: **Version 3.11** (Required for current compiled extensions).
  * **Bazel**: Version 6.3 or newer.
  * **PyTorch**: Required for running baseline comparisons and generating test data.
  * **Requirements File**: Python dependencies are listed in `requirements.txt`.

-----

## Building the Project 🛠️

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/Soluchann/gpu-kernel-framework.git  
    cd gpu_kernel_framework
    ```

2.  **(Recommended) Set up a Python virtual environment using Python 3.11**:

    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Build using Bazel**:

    ```bash
    # Build all targets
    bazel build //...
    ```

    This will compile the CUDA kernels and create the Python extension.

-----

## Run Tests

To generate data use `python/generate.py`

*   **Generate test bins**:
*   ```bash
    bazel run //python:generate
    ```
Use simliar args like run.py to generate specific bins

The `python/run.py` script executes tests using the generated data.

*   **Run all tests**:
    ```bash
    bazel run //python:run
    ```
*   **Run tests for a specific kernel**:
    ```bash
    bazel run //python:run -- --k eltw_add_bf16
    ```
*   **Run a specific test case**:
    ```bash
    bazel run //python:run -- --k eltw_add_bf16 --t test_1
    ```
    *(Note: You can also pass paths like `--k kernels/eltw_add_bf16/` or `--t kernels/eltw_add_bf16/tests/test_1`)*

-----

## Adding a New Kernel 🚀

1.  **Create a new kernel directory**:

    ```bash
    mkdir kernels/my_kernel_fp16
    ```

2.  **Implement the CUDA kernel**:
    Create a `kernel.cu` file inside the new directory with your kernel implementation.

3.  **Add test cases**:

      * Create a `tests` subdirectory within your kernel directory.
      * Inside `tests`, create subdirectories for each test case (e.g., `test_basic`, `test_large`).
      * Each test case directory should contain:
          * `A.bin`, `B.bin`: Input data files.
          * `expected_output.bin`: The expected output file.
          * `test.py`: A Python script that defines the test case parameters (e.g., shape, data type).

4.  **Update Bazel**:
    Add your new kernel to `kernels/BUILD.bazel`.

5.  **Update Python Bindings**:
    Expose your new kernel function in `bindings/kernel_bindings.cpp`.

6.  **Add to the Operator Registry**:
    Add a new operator class to `python/run.py` and register it in the `OPERATOR_REGISTRY`.

-----

## Project Structure 📁

```
gpu_kernel_framework/
├── .bazelrc # Bazel configuration
├── .bazelignore # Bazel ignore file
├── MODULE.bazel # Bazel module dependencies (Bzlmod)
├── MODULE.bazel.lock # Lock file for Bazel dependencies
├── BUILD.bazel # Root BUILD file
├── requirements.txt # Python dependencies
├── bindings/ # C++/Python bindings (pybind11)
│ └── BUILD.bazel
├── kernels/ # CUDA kernel implementations
│ ├── BUILD.bazel
│ ├── eltw_add_bf16/
│ ├── eltw_add_fp16/
│ ├── ...
├── python/
│ ├── BUILD.bazel
│ ├── run.py # Main test runner script
│ ├── generate.py # Test data generator script
│ ├── reference/ # PyTorch reference implementations
│ │ └── BUILD.bazel
│ └── utils/ # Utility functions
│ └── BUILD.bazel
└── src/ # C++/CUDA source/header files
├── BUILD.bazel
└── gpu_utils.h # Utility functions and macros (header only)
```

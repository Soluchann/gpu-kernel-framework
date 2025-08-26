# GPU Kernel Framework

A framework for developing and testing custom GPU kernels, with a focus on deep learning operations. This framework provides a streamlined process for building, testing, and benchmarking custom CUDA kernels against PyTorch baselines.

-----

## Features ✨

  * **Custom CUDA Kernels**: Implement high-performance GPU kernels for operations like matrix multiplication (`matmul`), element-wise addition (`eltw_add`), and 2D convolution (`conv2d`).
  * **Multiple Data Types**: Supports both `fp16` (half-precision) and `bf16` (bfloat16) data types.
  * **PyTorch Integration**: Kernels are exposed to Python through `pybind11`, allowing for easy integration with PyTorch.
  * **Automated Testing**: A Python-based testing framework (`run.py`) allows you to automatically discover and run test cases.
  * **Benchmarking**: The test runner provides performance comparisons between your custom kernels and PyTorch's native implementations.
  * **CMake Build System**: A robust and easy-to-use CMake build system that handles CUDA, C++, and Python bindings.

-----

## Requirements 📜

  * **NVIDIA GPU**: A CUDA-enabled GPU is required.
  * **CUDA Toolkit**: Version 11.0 or newer.
  * **C++ Compiler**: A C++17 compatible compiler (e.g., GCC, Clang, MSVC).
  * **Python**: Version 3.12.3
  * **CMake**: Version 3.18 or newer.
  * **PyTorch**: Required for running baseline comparisons.
  * **pybind11**: Fetched automatically by CMake.

-----

## Building the Project 🛠️

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/Soluchann/gpu-kernel-framework.git
    cd gpu_kernel_framework
    ```

2.  **Create a build directory**:

    ```bash
    mkdir build
    cd build
    ```

3.  **Run CMake and build**:

    ```bash
    cmake ..
    make -j
    ```

    This will compile the CUDA kernels and create a Python library in the `build/bindings` directory.

-----

## Running Tests 🧪

The `python/run.py` script is the main entry point for running tests.

### Run all tests

To discover and run all test cases for all kernels:

```bash
python python/run.py
```

### Filter by kernel

To run tests for a specific kernel (e.g., `eltw_add_fp16`):

```bash
python python/run.py --k eltw_add_fp16
```

### Filter by test case

To run a specific test case for a specific kernel:

```bash
python python/run.py --k eltw_add_fp16 --t test_1
```
### Use custom build

To run a using a specific build:

default_use : build

```bash
python python/run.py --k eltw_add_fp16 --t test_1 --bld custom_build
```
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

4.  **Update CMake**:
    Add your new kernel to `kernels/CMakeLists.txt`.

5.  **Update Python Bindings**:
    Expose your new kernel function in `bindings/kernel_bindings.cpp`.

6.  **Add to the Operator Registry**:
    Add a new operator class to `python/run.py` and register it in the `OPERATOR_REGISTRY`.

-----

## Project Structure 📁

```
gpu_kernel_framework/
├── bindings/             # C++/Python bindings (pybind11)
├── build/                # Build output
├── CMakeLists.txt        # Main CMake file
├── kernels/              # CUDA kernel implementations
│   ├── eltw_add_bf16/
│   ├── eltw_add_fp16/
│   ├── ...
├── python/
│   ├── run.py            # Main test runner script
│   ├── reference/        # PyTorch reference implementations
│   └── utils/            # Utility functions
└── src/                  # C++/CUDA source files
    ├── gpu_utils.cu
    └── tensor.h
```

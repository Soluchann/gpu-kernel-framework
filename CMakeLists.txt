cmake_minimum_required(VERSION 3.18)
project(GPUKernelFramework LANGUAGES CXX CUDA)

# Require C++17 and CUDA 11+
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable FP16 and BF16 support
add_compile_definitions(CUDA_HAS_FP16=1 CUDA_HAS_BF16=1)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# -----------------------
# Fetch pybind11 automatically
# -----------------------
include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6   # Stable release
    GIT_SHALLOW    TRUE      # Faster clone
)

# Avoid installing pybind11 globally
set(PYBIND11_INSTALL OFF CACHE BOOL "Disable installation" FORCE)
set(PYBIND11_NOPYTHON OFF CACHE BOOL "Enable Python support" FORCE)

FetchContent_MakeAvailable(pybind11)

# Now you can use 'pybind11' as a target  # Install via pip install pybind11 or use FetchContent

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

# Add subdirectory for bindings
add_subdirectory(bindings)

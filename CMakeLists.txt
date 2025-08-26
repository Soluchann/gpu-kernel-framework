cmake_minimum_required(VERSION 3.18)
project(GPUKernelFramework LANGUAGES CXX CUDA)

# Require C++17 and CUDA 11+
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Explicitly set CUDA standard on target level (safeguard)
set_property(TARGET ${CUDA_LIBRARIES} PROPERTY CUDA_STANDARD 17)

# Enable FP16 and BF16 support
add_compile_definitions(CUDA_HAS_FP16=1 CUDA_HAS_BF16=1)

# Optional: Set CUDA architectures (e.g., for A100, Hopper, etc.)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 CACHE STRING "CUDA architectures")

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

# Now you can use 'pybind11' as a target  
# (Alternatively, install via: pip install pybind11)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

# Add subdirectory for Python bindings
add_subdirectory(kernels) 
add_subdirectory(bindings)

# -----------------------
# Custom target: Generate test data + run validation
# -----------------------
add_custom_target(run_tests
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/generate.py
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/validate.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "🧪 Generating test data and running validation..."
    USES_TERMINAL
)

# -----------------------
# Custom target: Run specific test (e.g., make run_test eltw_add_fp16 test_1)
# -----------------------
add_custom_target(run_test
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/run.py ${KERNEL_NAME} ${TEST_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "🧪 Running specific test case..."
    USES_TERMINAL
)

# Allow passing KERNEL_NAME and TEST_NAME to run_test
set(KERNEL_NAME "" CACHE STRING "Kernel name to test")
set(TEST_NAME "" CACHE STRING "Test case name to run")
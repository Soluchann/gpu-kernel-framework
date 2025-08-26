# /home/adarsh/gpu_kernel_framework/bindings/CMakeLists.txt

# --- Define the Python extension module ---
pybind11_add_module(kernel_lib kernel_bindings.cpp)

# --- Link libraries ---
# Link the kernel_lib module against all the static kernel libraries defined in kernels/CMakeLists.txt

# --- Set output properties ---
set_target_properties(kernel_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PREFIX "" # Optional: removes 'lib' prefix on Linux
)

target_include_directories(kernel_lib PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    # Include other necessary directories if kernel_bindings.cpp needs them
    # ${CMAKE_SOURCE_DIR}/src # If you have common utils there used by bindings
    # ${CMAKE_SOURCE_DIR}/kernels # If you have common kernel headers there
)

target_link_libraries(kernel_lib PRIVATE
    ${ALL_KERNEL_LIBS}
    ${CUDA_LIBRARIES}
)

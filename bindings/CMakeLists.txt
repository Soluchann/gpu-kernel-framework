# bindings/CMakeLists.txt

add_library(eltwise_kernel STATIC
    ../kernels/eltwise/eltwise.cu
)

set_target_properties(eltwise_kernel PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(eltwise_kernel PRIVATE
    ../kernels/eltwise
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}  
)

pybind11_add_module(kernel_lib kernel_bindings.cpp)

# 👇 Ensure .so is placed in build/, not bindings/
set_target_properties(kernel_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PREFIX ""
)

target_include_directories(kernel_lib PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ../kernels/eltwise
)

target_link_libraries(kernel_lib PRIVATE eltwise_kernel ${CUDA_LIBRARIES})
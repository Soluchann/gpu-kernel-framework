// kernels/matmul/matmul.cu.h
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launch_matmul_fp16(
    const void* h_A,
    const void* h_B,
    void* h_C,
    int M, int N, int K
);
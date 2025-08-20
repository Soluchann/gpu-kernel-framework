#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void eltwise_add_kernel_half2(const half2* A, const half2* B, half2* Out, int N_half2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_half2) {
        Out[idx] = __hadd2(A[idx], B[idx]);
    }
}
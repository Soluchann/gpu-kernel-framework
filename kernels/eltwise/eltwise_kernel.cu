#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void eltwise_add_kernel(const __half* A, const __half* B, __half* Out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Out[idx] = __hadd(A[idx], B[idx]);
    }
}

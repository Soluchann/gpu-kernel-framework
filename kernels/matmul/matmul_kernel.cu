// kernels/matmul/matmul_kernel.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void matmul_fp16_kernel(
    const __half* A,
    const __half* B,
    __half* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum);
    }
}
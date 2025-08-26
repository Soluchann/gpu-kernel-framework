// kernels/matmul/matmul_bf16.cu
#include "kernels.cu.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "gpu_utils.cu"

__global__ void matmul_bf16_kernel(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
        }
        C[row * N + col] = __float2bfloat16(sum);
    }
}

void launch_matmul_bf16(
    const uint16_t* h_A,
    const uint16_t* h_B,
    uint16_t* h_C,
    int M, int N, int K
) {
    const __nv_bfloat16* A = reinterpret_cast<const __nv_bfloat16*>(h_A);
    const __nv_bfloat16* B = reinterpret_cast<const __nv_bfloat16*>(h_B);
    __nv_bfloat16* C = reinterpret_cast<__nv_bfloat16*>(h_C);

    size_t bytes_A = M * K * sizeof(__nv_bfloat16);
    size_t bytes_B = K * N * sizeof(__nv_bfloat16);
    size_t bytes_C = M * N * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
     
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_bf16_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
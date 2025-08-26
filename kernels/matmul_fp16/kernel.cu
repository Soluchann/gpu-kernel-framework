#include "kernels.cu.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include "gpu_utils.cu"

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

void launch_matmul_fp16(
    const void* h_A,
    const void* h_B,
    void* h_C,
    int M, int N, int K
) {
    const __half* A = static_cast<const __half*>(h_A);
    const __half* B = static_cast<const __half*>(h_B);
    __half* C = static_cast<__half*>(h_C);

    size_t bytes_A = M * K * sizeof(__half);
    size_t bytes_B = K * N * sizeof(__half);
    size_t bytes_C = M * N * sizeof(__half);

    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_fp16_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
#include "matmul.cu.h"
#include "matmul_kernel.cu"
#include <iostream>

inline void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
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
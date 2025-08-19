#include "eltwise.cu.h"
#include "eltwise_kernel.cu"
#include <iostream>

void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void launch_eltwise_add_fp16(const __half* h_A, const __half* h_B, __half* h_Out, int N) {
    __half *d_A, *d_B, *d_Out;
    size_t bytes = N * sizeof(__half);

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_Out, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    eltwise_add_kernel<<<grid_size, block_size>>>(d_A, d_B, d_Out, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_Out, d_Out, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_Out);
}

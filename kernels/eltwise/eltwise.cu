#include "eltwise.cu.h"
#include "eltwise_kernel.cu"
#include <iostream>

inline void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void launch_eltwise_add_fp16(const __half* h_A, const __half* h_B, __half* h_Out, int N) {
    if (N % 2 != 0) {
        std::cerr << "Error: N must be even for half2 kernel" << std::endl;
        exit(1);
    }

    int N_half2 = N / 2;
    size_t bytes = N * sizeof(__half);

    __half *d_A = nullptr, *d_B = nullptr, *d_Out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_Out, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    int block_size = 512;  // higher occupancy than 256
    int grid_size = (N_half2 + block_size - 1) / block_size;

    // Setup CUDA events for timing kernel only
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaEventRecord(start);
    eltwise_add_kernel_half2<<<grid_size, block_size>>>(
        reinterpret_cast<const half2*>(d_A),
        reinterpret_cast<const half2*>(d_B),
        reinterpret_cast<half2*>(d_Out),
        N_half2);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);

    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_Out, d_Out, bytes, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_Out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
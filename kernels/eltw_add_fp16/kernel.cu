#include "kernels.cu.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include "gpu_utils.cu"

__global__ void eltwise_add_kernel_half2(const half2* A, const half2* B, half2* Out, int N_half2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_half2) {
        Out[idx] = __hadd2(A[idx], B[idx]);
    }
}

void launch_eltwise_add_fp16(const __half* h_A, const __half* h_B, __half* h_Out, int N) {
    // Pad to even size
    int N_padded = (N % 2 == 0) ? N : N + 1;
    size_t bytes_original = N * sizeof(__half);
    size_t bytes_padded = N_padded * sizeof(__half);
    
    // Allocate padded arrays on host
    __half* h_A_padded = new __half[N_padded];
    __half* h_B_padded = new __half[N_padded];
    __half* h_Out_padded = new __half[N_padded];
    
    // Copy original data
    memcpy(h_A_padded, h_A, bytes_original);
    memcpy(h_B_padded, h_B, bytes_original);
    
    // Pad with zeros if needed
    if (N_padded > N) {
        h_A_padded[N] = __float2half(0.0f);
        h_B_padded[N] = __float2half(0.0f);
    }
    
    // Allocate device memory
    __half *d_A = nullptr, *d_B = nullptr, *d_Out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_padded));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_padded));
    CUDA_CHECK(cudaMalloc(&d_Out, bytes_padded));

    CUDA_CHECK(cudaMemcpy(d_A, h_A_padded, bytes_padded, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_padded, bytes_padded, cudaMemcpyHostToDevice));

    int block_size = 512;
    int N_half2 = N_padded / 2;
    int grid_size = (N_half2 + block_size - 1) / block_size;
    
    // Launch kernel
    eltwise_add_kernel_half2<<<grid_size, block_size>>>(
        reinterpret_cast<const half2*>(d_A),
        reinterpret_cast<const half2*>(d_B),
        reinterpret_cast<half2*>(d_Out),
        N_half2);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back (only original size)
    CUDA_CHECK(cudaMemcpy(h_Out_padded, d_Out, bytes_padded, cudaMemcpyDeviceToHost));
    
    // Copy only the original size back to output
    memcpy(h_Out, h_Out_padded, bytes_original);

    // Cleanup
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_Out);
    delete[] h_A_padded;
    delete[] h_B_padded;
    delete[] h_Out_padded;
}
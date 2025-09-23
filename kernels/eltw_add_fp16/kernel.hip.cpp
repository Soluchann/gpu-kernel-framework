#include "gpu_runtime.h"
#include "../kernels.h"

__global__ void eltwise_add_kernel_half2(const half2* A, const half2* B, half2* Out, int N_half2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N_half2) {
        Out[idx] = __hadd2(A[idx], B[idx]);
    }
}

void launch_eltwise_add_fp16(const gpuHalf* h_A, const gpuHalf* h_B, gpuHalf* h_Out, int N) {
    int N_padded = (N % 2 == 0) ? N : N + 1;
    size_t bytes_original = N * sizeof(gpuHalf);
    size_t bytes_padded = N_padded * sizeof(gpuHalf);
    
    gpuHalf* h_A_padded = new gpuHalf[N_padded];
    gpuHalf* h_B_padded = new gpuHalf[N_padded];
    gpuHalf* h_Out_padded = new gpuHalf[N_padded];
    
    memcpy(h_A_padded, h_A, bytes_original);
    memcpy(h_B_padded, h_B, bytes_original);
    
    if (N_padded > N) {
        h_A_padded[N] = __float2half(0.0f);
        h_B_padded[N] = __float2half(0.0f);
    }
    
    gpuHalf *d_A = nullptr, *d_B = nullptr, *d_Out = nullptr;
    GPU_CHECK(gpuMalloc(&d_A, bytes_padded));
    GPU_CHECK(gpuMalloc(&d_B, bytes_padded));
    GPU_CHECK(gpuMalloc(&d_Out, bytes_padded));

    GPU_CHECK(gpuMemcpy(d_A, h_A_padded, bytes_padded, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_B, h_B_padded, bytes_padded, gpuMemcpyHostToDevice));

    int block_size = 512;
    int N_half2 = N_padded / 2;
    int grid_size = (N_half2 + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(eltwise_add_kernel_half2, dim3(grid_size), dim3(block_size), 0, 0,
        reinterpret_cast<const half2*>(d_A),
        reinterpret_cast<const half2*>(d_B),
        reinterpret_cast<half2*>(d_Out),
        N_half2);
    
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    GPU_CHECK(gpuMemcpy(h_Out_padded, d_Out, bytes_padded, gpuMemcpyDeviceToHost));
    
    memcpy(h_Out, h_Out_padded, bytes_original);

    gpuFree(d_A); 
    gpuFree(d_B); 
    gpuFree(d_Out);
    delete[] h_A_padded;
    delete[] h_B_padded;
    delete[] h_Out_padded;
} 
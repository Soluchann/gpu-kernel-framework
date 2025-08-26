// kernels/eltwise/eltwise_bf16.cu
#include "kernels.cu.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "gpu_utils.cu"

__global__ void eltwise_add_bf16_kernel(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* Out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Out[idx] = __hadd(A[idx], B[idx]);
    }
}

void launch_eltwise_add_bf16(
    const uint16_t* h_A,
    const uint16_t* h_B,
    uint16_t* h_Out,
    int N
) {
    const __nv_bfloat16* A = reinterpret_cast<const __nv_bfloat16*>(h_A);
    const __nv_bfloat16* B = reinterpret_cast<const __nv_bfloat16*>(h_B);
    __nv_bfloat16* Out = reinterpret_cast<__nv_bfloat16*>(h_Out);

    size_t bytes = N * sizeof(__nv_bfloat16);
    __nv_bfloat16 *d_A, *d_B, *d_Out;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_Out, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    eltwise_add_bf16_kernel<<<grid_size, block_size>>>(d_A, d_B, d_Out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Out, d_Out, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_Out);
}
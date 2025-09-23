#include "gpu_runtime.h"
#include "../kernels.h"

__global__ void eltwise_add_bf16_kernel(
    const gpuBfloat16* A,
    const gpuBfloat16* B,
    gpuBfloat16* Out,
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
    const gpuBfloat16* A = reinterpret_cast<const gpuBfloat16*>(h_A);
    const gpuBfloat16* B = reinterpret_cast<const gpuBfloat16*>(h_B);
    gpuBfloat16* Out = reinterpret_cast<gpuBfloat16*>(h_Out);

    size_t bytes = N * sizeof(gpuBfloat16);
    gpuBfloat16 *d_A, *d_B, *d_Out;

    GPU_CHECK(gpuMalloc(&d_A, bytes));
    GPU_CHECK(gpuMalloc(&d_B, bytes));
    GPU_CHECK(gpuMalloc(&d_Out, bytes));

    GPU_CHECK(gpuMemcpy(d_A, A, bytes, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_B, B, bytes, gpuMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    hipLaunchKernelGGL(eltwise_add_bf16_kernel, dim3(grid_size), dim3(block_size), 0, 0, d_A, d_B, d_Out, N);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    GPU_CHECK(gpuMemcpy(Out, d_Out, bytes, gpuMemcpyDeviceToHost));

    gpuFree(d_A);
    gpuFree(d_B);
    gpuFree(d_Out);
}
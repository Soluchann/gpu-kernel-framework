#include "gpu_runtime.h"
#include "../kernels.h"

__global__ void matmul_fp16_kernel(
    const gpuHalf* A,
    const gpuHalf* B,
    gpuHalf* C,
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
    const gpuHalf* A = static_cast<const gpuHalf*>(h_A);
    const gpuHalf* B = static_cast<const gpuHalf*>(h_B);
    gpuHalf* C = static_cast<gpuHalf*>(h_C);

    size_t bytes_A = M * K * sizeof(gpuHalf);
    size_t bytes_B = K * N * sizeof(gpuHalf);
    size_t bytes_C = M * N * sizeof(gpuHalf);

    gpuHalf *d_A, *d_B, *d_C;
    GPU_CHECK(gpuMalloc(&d_A, bytes_A));
    GPU_CHECK(gpuMalloc(&d_B, bytes_B));
    GPU_CHECK(gpuMalloc(&d_C, bytes_C));

    GPU_CHECK(gpuMemcpy(d_A, A, bytes_A, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_B, B, bytes_B, gpuMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    hipLaunchKernelGGL(matmul_fp16_kernel, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    GPU_CHECK(gpuMemcpy(C, d_C, bytes_C, gpuMemcpyDeviceToHost));

    gpuFree(d_A);
    gpuFree(d_B);
    gpuFree(d_C);
}
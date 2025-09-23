#include "gpu_runtime.h"
#include "../kernels.h"

__global__ void matmul_bf16_kernel(
    const gpuBfloat16* A,
    const gpuBfloat16* B,
    gpuBfloat16* C,
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
    const gpuBfloat16* A = reinterpret_cast<const gpuBfloat16*>(h_A);
    const gpuBfloat16* B = reinterpret_cast<const gpuBfloat16*>(h_B);
    gpuBfloat16* C = reinterpret_cast<gpuBfloat16*>(h_C);

    size_t bytes_A = M * K * sizeof(gpuBfloat16);
    size_t bytes_B = K * N * sizeof(gpuBfloat16);
    size_t bytes_C = M * N * sizeof(gpuBfloat16);

    gpuBfloat16 *d_A, *d_B, *d_C;

    GPU_CHECK(gpuMalloc(&d_A, bytes_A));
    GPU_CHECK(gpuMalloc(&d_B, bytes_B));
    GPU_CHECK(gpuMalloc(&d_C, bytes_C));

    GPU_CHECK(gpuMemcpy(d_A, A, bytes_A, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_B, B, bytes_B, gpuMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    hipLaunchKernelGGL(matmul_bf16_kernel, grid, block, 0, 0, d_A, d_B, d_C, M, N, K);
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    GPU_CHECK(gpuMemcpy(C, d_C, bytes_C, gpuMemcpyDeviceToHost));

    gpuFree(d_A);
    gpuFree(d_B);
    gpuFree(d_C);
}
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include "kernels.cu.h"
#include "gpu_utils.cu"

__global__ void conv2d_bf16_kernel(
    const __nv_bfloat16* X, const __nv_bfloat16* W, __nv_bfloat16* Y,
    int N, int C, int H, int W_in,
    int K, int R, int S,
    int stride, int padding,
    int H_out, int W_out) {

    int y_w = blockIdx.x * blockDim.x + threadIdx.x;
    int y_h = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z % K;
    int n = blockIdx.z / K;

    if (y_h >= H_out || y_w >= W_out || n >= N) return;

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int x_h = y_h * stride - padding + r;
                int x_w = y_w * stride - padding + s;
                if (x_h >= 0 && x_h < H && x_w >= 0 && x_w < W_in) {
                    int x_idx = n*(C*H*W_in) + c*(H*W_in) + x_h*W_in + x_w;
                    int w_idx = k*(C*R*S) + c*(R*S) + r*S + s;
                    sum += __bfloat162float(X[x_idx]) * __bfloat162float(W[w_idx]);
                }
            }
        }
    }
    int y_idx = n*(K*H_out*W_out) + k*(H_out*W_out) + y_h*W_out + y_w;
    Y[y_idx] = __float2bfloat16(sum);
}

void launch_conv2d_bf16(
    const uint16_t* h_X, const uint16_t* h_W, uint16_t* h_Y,
    int N, int C, int H, int W_in,
    int K, int R, int S,
    int stride, int padding) {

    const __nv_bfloat16* X = reinterpret_cast<const __nv_bfloat16*>(h_X);
    const __nv_bfloat16* W = reinterpret_cast<const __nv_bfloat16*>(h_W);
    __nv_bfloat16* Y = reinterpret_cast<__nv_bfloat16*>(h_Y);

    const int H_out = (H - R + 2 * padding) / stride + 1;
    const int W_out = (W_in - S + 2 * padding) / stride + 1;

    const size_t input_bytes = N * C * H * W_in * sizeof(__nv_bfloat16);
    const size_t weight_bytes = K * C * R * S * sizeof(__nv_bfloat16);
    const size_t output_bytes = N * K * H_out * W_out * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_X, *d_W, *d_Y;
    CUDA_CHECK(cudaMalloc(&d_X, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_W, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_Y, output_bytes));

    CUDA_CHECK(cudaMemcpy(d_X, X, input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W, weight_bytes, cudaMemcpyHostToDevice));

    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_in_grid(
        (W_out + threads_per_block.x - 1) / threads_per_block.x,
        (H_out + threads_per_block.y - 1) / threads_per_block.y,
        N * K
    );

    conv2d_bf16_kernel<<<blocks_in_grid, threads_per_block>>>(
        d_X, d_W, d_Y, N, C, H, W_in, K, R, S, stride, padding, H_out, W_out
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(Y, d_Y, output_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_Y));
}

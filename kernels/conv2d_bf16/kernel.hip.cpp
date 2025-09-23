#include "gpu_runtime.h"
#include "../kernels.h"

__global__ void conv2d_bf16_kernel(
    const gpuBfloat16* X, const gpuBfloat16* W, gpuBfloat16* Y,
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

    const gpuBfloat16* X = reinterpret_cast<const gpuBfloat16*>(h_X);
    const gpuBfloat16* W = reinterpret_cast<const gpuBfloat16*>(h_W);
    gpuBfloat16* Y = reinterpret_cast<gpuBfloat16*>(h_Y);

    const int H_out = (H - R + 2 * padding) / stride + 1;
    const int W_out = (W_in - S + 2 * padding) / stride + 1;

    const size_t input_bytes = N * C * H * W_in * sizeof(gpuBfloat16);
    const size_t weight_bytes = K * C * R * S * sizeof(gpuBfloat16);
    const size_t output_bytes = N * K * H_out * W_out * sizeof(gpuBfloat16);

    gpuBfloat16 *d_X, *d_W, *d_Y;
    GPU_CHECK(gpuMalloc(&d_X, input_bytes));
    GPU_CHECK(gpuMalloc(&d_W, weight_bytes));
    GPU_CHECK(gpuMalloc(&d_Y, output_bytes));

    GPU_CHECK(gpuMemcpy(d_X, X, input_bytes, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_W, W, weight_bytes, gpuMemcpyHostToDevice));

    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_in_grid(
        (W_out + threads_per_block.x - 1) / threads_per_block.x,
        (H_out + threads_per_block.y - 1) / threads_per_block.y,
        N * K
    );

    hipLaunchKernelGGL(conv2d_bf16_kernel, blocks_in_grid, threads_per_block, 0, 0,
        d_X, d_W, d_Y, N, C, H, W_in, K, R, S, stride, padding, H_out, W_out
    );
    
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());

    GPU_CHECK(gpuMemcpy(Y, d_Y, output_bytes, gpuMemcpyDeviceToHost));

    GPU_CHECK(gpuFree(d_X));
    GPU_CHECK(gpuFree(d_W));
    GPU_CHECK(gpuFree(d_Y));
} 
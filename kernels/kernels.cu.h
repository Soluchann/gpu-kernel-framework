#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

void launch_eltwise_add_fp16(const __half* h_A, const __half* h_B, __half* h_Out, int N);
void launch_eltwise_add_bf16(
    const uint16_t* h_A,
    const uint16_t* h_B,
    uint16_t* h_Out,
    int N
);
void launch_matmul_fp16(
    const void* h_A,
    const void* h_B,
    void* h_C,
    int M, int N, int K
);
void launch_matmul_bf16(
    const uint16_t* h_A,
    const uint16_t* h_B,
    uint16_t* h_C,
    int M, int N, int K
);
void launch_conv2d_fp16(
    const uint16_t* h_X, const uint16_t* h_W, uint16_t* h_Y,
    int N, int C, int H, int W_in,
    int K, int R, int S,
    int stride, int padding);
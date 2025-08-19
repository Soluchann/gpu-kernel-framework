#include <cuda_fp16.h>
#include <cuda_runtime.h>

void launch_eltwise_add_fp16(const __half* h_A, const __half* h_B, __half* h_Out, int N);

#pragma once

#if defined(__HIP_PLATFORM_NVIDIA__)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define gpuSuccess cudaSuccess
#define gpuError_t cudaError_t
#define gpuMalloc cudaMalloc
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuFree cudaFree
#define gpuGetLastError cudaGetLastError
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetErrorString cudaGetErrorString

#define gpuHalf __half
#define gpuBfloat16 __nv_bfloat16

#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#define gpuSuccess hipSuccess
#define gpuError_t hipError_t
#define gpuMalloc hipMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuFree hipFree
#define gpuGetLastError hipGetLastError
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetErrorString hipGetErrorString

#define gpuHalf __half
#define gpuBfloat16 __hip_bfloat16

#endif

#define GPU_CHECK(call)                                    \
    do {                                                   \
        gpuError_t error = call;                           \
        if (error != gpuSuccess) {                         \
            fprintf(stderr, "GPU error at %s:%d - %s\n",   \
                    __FILE__, __LINE__,                    \
                    gpuGetErrorString(error));             \
            exit(1);                                       \
        }                                                  \
    } while (0)
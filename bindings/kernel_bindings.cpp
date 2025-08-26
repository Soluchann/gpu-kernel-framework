#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_fp16.h>
#include <vector>
#include "kernels.cu.h"  

namespace py = pybind11;

py::array_t<uint16_t> eltwise_add_fp16_wrapper(py::array_t<uint16_t> A_arr, py::array_t<uint16_t> B_arr) {
    auto bufA = A_arr.request(), bufB = B_arr.request();

    if (bufA.ndim != 1 || bufB.ndim != 1 || bufA.size != bufB.size) {
        throw std::runtime_error("Inputs must be 1D and same size");
    }

    int N = bufA.size;
    uint16_t *A = static_cast<uint16_t*>(bufA.ptr);
    uint16_t *B = static_cast<uint16_t*>(bufB.ptr);

    // Use vector for safety
    std::vector<__half> h_A(N), h_B(N), h_Out(N);

    for (int i = 0; i < N; i++) {
        h_A[i] = *reinterpret_cast<__half*>(&A[i]);
        h_B[i] = *reinterpret_cast<__half*>(&B[i]);
    }

    launch_eltwise_add_fp16(h_A.data(), h_B.data(), h_Out.data(), N);

    // Create result array
    py::array_t<uint16_t> result = py::array_t<uint16_t>({N});
    uint16_t *C = static_cast<uint16_t*>(result.mutable_data());
    for (int i = 0; i < N; i++) {
        C[i] = *reinterpret_cast<uint16_t*>(&h_Out[i]);
    }

    return result;
}

py::array_t<uint16_t> matmul_fp16_wrapper(
    py::array_t<uint16_t> A_arr,
    py::array_t<uint16_t> B_arr,
    int M, int N, int K
) {
    auto bufA = A_arr.request(), bufB = B_arr.request();

    if (bufA.ndim != 1 || bufB.ndim != 1)
        throw std::runtime_error("Inputs must be 1D (flattened)");

    if (bufA.size != M * K || bufB.size != K * N)
        throw std::runtime_error("Input sizes do not match M, N, K");

    uint16_t *A = static_cast<uint16_t*>(bufA.ptr);
    uint16_t *B = static_cast<uint16_t*>(bufB.ptr);

    std::vector<__half> h_A(M * K), h_B(K * N), h_C(M * N);

    for (int i = 0; i < M * K; i++)
        h_A[i] = *reinterpret_cast<__half*>(&A[i]);
    for (int i = 0; i < K * N; i++)
        h_B[i] = *reinterpret_cast<__half*>(&B[i]);

    launch_matmul_fp16(h_A.data(), h_B.data(), h_C.data(), M, N, K);

    py::array_t<uint16_t> result = py::array_t<uint16_t>({M * N});
    uint16_t *C = static_cast<uint16_t*>(result.mutable_data());
    for (int i = 0; i < M * N; i++) {
        C[i] = *reinterpret_cast<uint16_t*>(&h_C[i]);
    }

    return result;
}

py::array_t<uint16_t> eltwise_add_bf16_wrapper(py::array_t<uint16_t> A_arr, py::array_t<uint16_t> B_arr) {
    auto bufA = A_arr.request(), bufB = B_arr.request();
    if (bufA.ndim != 1 || bufB.ndim != 1 || bufA.size != bufB.size)
        throw std::runtime_error("Inputs must be 1D and same size");

    int N = bufA.size;
    uint16_t *A = static_cast<uint16_t*>(bufA.ptr);
    uint16_t *B = static_cast<uint16_t*>(bufB.ptr);

    // ✅ Pass raw bit patterns to CUDA kernel
    py::array_t<uint16_t> result = py::array_t<uint16_t>({N});
    uint16_t *C = static_cast<uint16_t*>(result.mutable_data());

    // ✅ Call kernel with uint16* (bit reinterpretation happens in CUDA code)
    launch_eltwise_add_bf16(A, B, C, N);

    return result;
}

// BF16 Matmul
py::array_t<uint16_t> matmul_bf16_wrapper(
    py::array_t<uint16_t> A_arr,
    py::array_t<uint16_t> B_arr,
    int M, int N, int K
) {
    auto bufA = A_arr.request(), bufB = B_arr.request();
    if (bufA.size != M * K || bufB.size != K * N)
        throw std::runtime_error("Size mismatch");

    uint16_t *A = static_cast<uint16_t*>(bufA.ptr);
    uint16_t *B = static_cast<uint16_t*>(bufB.ptr);

    py::array_t<uint16_t> result = py::array_t<uint16_t>({M * N});
    uint16_t *C = static_cast<uint16_t*>(result.mutable_data());

    launch_matmul_bf16(A, B, C, M, N, K);

    return result;
}

py::array_t<uint16_t> conv2d_fp16_wrapper(
    py::array_t<uint16_t> X_arr, py::array_t<uint16_t> W_arr, py::array_t<uint16_t> Y_arr,
    int N, int C, int H, int W_in,
    int K, int R, int S,
    int stride, int padding
) {
    auto bufX = X_arr.request(), bufW = W_arr.request(), bufY = Y_arr.request();

    // Basic size validation
    if (bufX.size != N * C * H * W_in) throw std::runtime_error("Input tensor size mismatch");
    if (bufW.size != K * C * R * S) throw std::runtime_error("Weight tensor size mismatch");

    launch_conv2d_fp16(
        static_cast<uint16_t*>(bufX.ptr),
        static_cast<uint16_t*>(bufW.ptr),
        static_cast<uint16_t*>(bufY.ptr),
        N, C, H, W_in, K, R, S, stride, padding
    );
    return Y_arr;
}

py::array_t<uint16_t> conv2d_bf16_wrapper(
    py::array_t<uint16_t> X_arr, py::array_t<uint16_t> W_arr, py::array_t<uint16_t> Y_arr,
    int N, int C, int H, int W_in,
    int K, int R, int S,
    int stride, int padding
) {
    auto bufX = X_arr.request(), bufW = W_arr.request(), bufY = Y_arr.request();

    if (bufX.size != N * C * H * W_in) throw std::runtime_error("Input tensor size mismatch");
    if (bufW.size != K * C * R * S) throw std::runtime_error("Weight tensor size mismatch");

    launch_conv2d_bf16(
        static_cast<uint16_t*>(bufX.ptr),
        static_cast<uint16_t*>(bufW.ptr),
        static_cast<uint16_t*>(bufY.ptr),
        N, C, H, W_in, K, R, S, stride, padding
    );
    return Y_arr;
}


PYBIND11_MODULE(kernel_lib, m) {
    m.doc() = "CUDA kernel bindings";
    m.def("eltwise_add_fp16_cu", &eltwise_add_fp16_wrapper, "Add two tensors in FP16");
    m.def("matmul_fp16_cu", &matmul_fp16_wrapper, "FP16 matmul: (MxK) @ (KxN)");
    m.def("eltwise_add_bf16_cu", &eltwise_add_bf16_wrapper, "BF16 element-wise add");
    m.def("matmul_bf16_cu", &matmul_bf16_wrapper, "BF16 matmul: (MxK) @ (KxN)");
    m.def("conv2d_fp16_cu", &conv2d_fp16_wrapper, "FP16 Conv2D (NCHW)");
    m.def("conv2d_bf16_cu", &conv2d_bf16_wrapper, "BF16 Conv2D (NCHW)");
}
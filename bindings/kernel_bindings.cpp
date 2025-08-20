#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_fp16.h>
#include <vector>
#include "../kernels/eltwise/eltwise.cu.h"
#include "../kernels/matmul/matmul.cu.h"    

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

// 🔥 Required: Module definition
PYBIND11_MODULE(kernel_lib, m) {
    m.doc() = "CUDA FP16 element-wise add";
    m.def("eltwise_add_fp16", &eltwise_add_fp16_wrapper, "Add two tensors in FP16");
    m.def("matmul_fp16", &matmul_fp16_wrapper, "FP16 matmul: (MxK) @ (KxN)");
}
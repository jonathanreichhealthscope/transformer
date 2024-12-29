#pragma once

#ifdef USE_CUDA
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(error) + " at " + __FILE__ + \
                               ":" + std::to_string(__LINE__));                \
    }                                                                          \
  } while (0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(std::string("cuBLAS error: ") +                 \
                               std::to_string(static_cast<int>(status)) +      \
                               " at " + __FILE__ + ":" +                       \
                               std::to_string(__LINE__));                      \
    }                                                                          \
  } while (0)

#ifdef __CUDACC__
// Kernel launch macro - only available in CUDA source files
#define CUDA_LAUNCH(kernel, gridSize, blockSize, sharedMem, stream, ...)       \
  do {                                                                         \
    kernel<<<gridSize, blockSize, sharedMem, stream>>>(__VA_ARGS__);           \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)
#else
// Stub for non-CUDA source files
#define CUDA_LAUNCH(kernel, gridSize, blockSize, sharedMem, stream, ...)       \
  do {                                                                         \
    throw std::runtime_error("CUDA_LAUNCH called from non-CUDA code");         \
  } while (0)
#endif

// Add advanced memory management
#define CUDA_MALLOC_ASYNC(ptr, size)                                           \
  do {                                                                         \
    cudaError_t error = cudaMallocAsync(&(ptr), (size), cudaStreamPerThread);  \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error("CUDA async malloc failed");                    \
    }                                                                          \
  } while (0)

// Add tensor core operations for matrix multiply
template <typename T>
void tensor_core_gemm(const T *A, const T *B, T *C, int m, int n, int k,
                      cudaStream_t stream = nullptr) {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  if (stream)
    CUBLAS_CHECK(cublasSetStream(handle, stream));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                            A, CUDA_R_32F, m, B, CUDA_R_32F, k, &beta, C,
                            CUDA_R_32F, m, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  CUBLAS_CHECK(cublasDestroy(handle));
}

#endif
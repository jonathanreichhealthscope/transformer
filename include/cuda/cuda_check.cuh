#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Error checking macro
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t error = call;                                                                  \
        if (error != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                       \
                    cudaGetErrorString(error));                                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t status = call;                                                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                     \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,                     \
                    static_cast<int>(status));                                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                         \
    } while (0)
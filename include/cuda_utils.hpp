#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + \
                               std::string(cudaGetErrorString(err)) + \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuBLAS error code: ") + \
                               std::to_string(status) + \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

// Global cuBLAS handle
extern cublasHandle_t cublas_handle; 
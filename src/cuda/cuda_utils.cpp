#include "../../include/cuda_utils.hpp"

// Global cuBLAS handle
cublasHandle_t cublas_handle;

// Initialize cuBLAS handle at program start
__attribute__((constructor))
static void init_cublas() {
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
}

// Destroy cuBLAS handle at program end
__attribute__((destructor))
static void destroy_cublas() {
    if (cublas_handle != nullptr) {
        cublasDestroy(cublas_handle);
    }
} 
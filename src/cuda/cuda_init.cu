#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cublas_check.cuh"

// Global cuBLAS handle
cublasHandle_t cublas_handle;

void initialize_cuda() {
    // Select first available GPU
    CUDA_CHECK(cudaSetDevice(0));
    
    // Create cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // Set cuBLAS to use tensor cores if available
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
    
    // Enable asynchronous execution
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleAuto));
}

void cleanup_cuda() {
    // Destroy cuBLAS handle
    if (cublas_handle != nullptr) {
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
    }
    
    // Reset device
    CUDA_CHECK(cudaDeviceReset());
} 
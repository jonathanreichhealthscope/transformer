#include "../../include/cuda/cuda_check.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

cublasHandle_t cublas_handle;

void initialize_cuda() {
    // Get number of devices
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device count: " +
                                 std::string(cudaGetErrorString(error)));
    }

    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA-capable devices found");
    }

    // Get device properties
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0)); // Use first device

    // Print device info
    printf("Using CUDA Device %d: %s\n", 0, deviceProp.name);

    // Set device
    CUDA_CHECK(cudaSetDevice(0));

    // Initialize cuBLAS
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to initialize cuBLAS");
    }

    // Ensure device is ready
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cleanup_cuda() {
    if (cublas_handle != nullptr) {
        cublasDestroy(cublas_handle);
    }
    cudaDeviceReset();
}
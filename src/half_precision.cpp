#include "../include/half_precision.hpp"
#ifdef USE_CUDA
#include "../include/cuda/cuda_check.cuh"
#include "../include/cuda/cuda_utils.cuh"
#include "../include/cuda/half_precision_kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

std::vector<half_type> HalfPrecisionTraining::half_data;

void HalfPrecisionTraining::convert_to_fp16(Matrix& matrix) {
    const size_t size = matrix.rows() * matrix.cols();
    if (size == 0) {
        return;
    }
    half_data.resize(size);

#ifdef USE_CUDA
    // Check if device supports FP16
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 6) {
        throw std::runtime_error("GPU does not support FP16 operations");
    }

    half_type* d_half = nullptr;
    float* d_float = nullptr;
    
    try {
        // Print debug info
        std::cout << "Converting matrix of size " << size << " to FP16" << std::endl;
        
        // Allocate memory
        CUDA_CHECK(cudaMalloc(&d_float, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_half, size * sizeof(half_type)));

        // Copy input to GPU
        CUDA_CHECK(cudaMemcpy(d_float, matrix.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Synchronize before kernel launch
        CUDA_CHECK(cudaDeviceSynchronize());

        // Launch conversion
        launch_fp32_to_fp16(d_float, reinterpret_cast<__half*>(d_half), size);
        
        // Synchronize after kernel launch
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check for kernel execution errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("Kernel execution failed: ") + cudaGetErrorString(error));
        }

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(half_data.data(), d_half, size * sizeof(half_type), cudaMemcpyDeviceToHost));

        // Free device memory
        if (d_float) CUDA_CHECK(cudaFree(d_float));
        if (d_half) CUDA_CHECK(cudaFree(d_half));

        std::cout << "Successfully converted to FP16" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CUDA error in convert_to_fp16: " << e.what() << std::endl;
        
        // Clean up
        if (d_float) cudaFree(d_float);
        if (d_half) cudaFree(d_half);
        
        std::cerr << "Falling back to CPU implementation" << std::endl;
        
        // CPU fallback
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            half_data[i] = static_cast<half_type>(matrix.data()[i]);
        }
    }
#else
    // CPU version
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        half_data[i] = static_cast<half_type>(matrix.data()[i]);
    }
#endif
}

void HalfPrecisionTraining::convert_to_fp32(Matrix& matrix) {
    const size_t size = matrix.rows() * matrix.cols();
    if (size == 0) {
        return;
    }

#ifdef USE_CUDA
    // Check if device supports FP16
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 6) {
        throw std::runtime_error("GPU does not support FP16 operations");
    }

    float* d_float = nullptr;
    half_type* d_half = nullptr;
    
    try {
        // Print debug info
        std::cout << "Converting matrix of size " << size << " from FP16 to FP32" << std::endl;
        
        // Allocate memory
        CUDA_CHECK(cudaMalloc(&d_float, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_half, size * sizeof(half_type)));

        // Copy input to GPU
        CUDA_CHECK(cudaMemcpy(d_half, half_data.data(), size * sizeof(half_type), cudaMemcpyHostToDevice));
        
        // Synchronize before kernel launch
        CUDA_CHECK(cudaDeviceSynchronize());

        // Launch conversion
        launch_fp16_to_fp32(reinterpret_cast<__half*>(d_half), d_float, size);
        
        // Synchronize after kernel launch
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check for kernel execution errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("Kernel execution failed: ") + cudaGetErrorString(error));
        }

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(matrix.data(), d_float, size * sizeof(float), cudaMemcpyDeviceToHost));

        // Free device memory
        if (d_float) CUDA_CHECK(cudaFree(d_float));
        if (d_half) CUDA_CHECK(cudaFree(d_half));

        std::cout << "Successfully converted to FP32" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CUDA error in convert_to_fp32: " << e.what() << std::endl;
        
        // Clean up
        if (d_float) cudaFree(d_float);
        if (d_half) cudaFree(d_half);
        
        std::cerr << "Falling back to CPU implementation" << std::endl;
        
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            matrix.data()[i] = static_cast<float>(half_data[i]);
        }
    }
#else
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        matrix.data()[i] = static_cast<float>(half_data[i]);
    }
#endif
}
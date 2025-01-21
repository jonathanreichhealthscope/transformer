#include "../include/half_precision.hpp"
#ifdef USE_CUDA
#include "../include/cuda/cuda_check.cuh"
#include "../include/cuda/cuda_utils.cuh"
#include "../include/cuda/half_precision_kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif
#include "../include/cuda_manager.hpp"
#include "../include/memory_pool.hpp"
#include "../include/config.hpp"

std::vector<half_type> HalfPrecisionTraining::half_data;

// Static instance with default pool size (512MB)
static std::unique_ptr<CudaManager> cuda_manager = std::make_unique<CudaManager>(
    0,  // device id
    512 * 1024 * 1024  // default 512MB
);

// Initialize function to be called after config is loaded
void HalfPrecisionTraining::initialize(const TransformerConfig& config) {
    cuda_manager = std::make_unique<CudaManager>(
        0,  // device id
        config.memory_pool_size * 1024 * 1024  // convert MB to bytes
    );
}

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

    try {
        // Memory now comes from pool
        float* d_float = static_cast<float*>(cuda_manager->allocate(size * sizeof(float)));
        half_type* d_half = static_cast<half_type*>(cuda_manager->allocate(size * sizeof(half_type)));

        // Print debug info
        std::cout << "Converting matrix of size " << size << " to FP16" << std::endl;
        
        // Copy input to GPU
        CUDA_CHECK(cudaMemcpy(d_float, matrix.data(), size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Ensure synchronization
        cuda_manager->synchronize();

        // Launch conversion
        launch_fp32_to_fp16(d_float, reinterpret_cast<__half*>(d_half), size);
        
        cuda_manager->synchronize();

        // Check for kernel execution errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("Kernel execution failed: ") + cudaGetErrorString(error));
        }

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(half_data.data(), d_half, size * sizeof(half_type), cudaMemcpyDeviceToHost));

        // Memory returns to pool instead of being freed
        cuda_manager->deallocate(d_float);
        cuda_manager->deallocate(d_half);

        std::cout << "Successfully converted to FP16" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CUDA error: " << e.what() << std::endl;
        
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

    try {
        // Use CUDA manager for memory allocation
        float* d_float = static_cast<float*>(cuda_manager->allocate(size * sizeof(float)));
        half_type* d_half = static_cast<half_type*>(cuda_manager->allocate(size * sizeof(half_type)));

        // Print debug info
        std::cout << "Converting matrix of size " << size << " from FP16 to FP32" << std::endl;

        // Copy input to GPU
        CUDA_CHECK(cudaMemcpy(d_half, half_data.data(), size * sizeof(half_type), cudaMemcpyHostToDevice));
        
        // Ensure synchronization
        cuda_manager->synchronize();

        // Launch conversion
        launch_fp16_to_fp32(reinterpret_cast<__half*>(d_half), d_float, size);
        
        cuda_manager->synchronize();

        // Check for kernel execution errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("Kernel execution failed: ") + cudaGetErrorString(error));
        }

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(matrix.data(), d_float, size * sizeof(float), cudaMemcpyDeviceToHost));

        // Safe cleanup
        cuda_manager->deallocate(d_float);
        cuda_manager->deallocate(d_half);

        std::cout << "Successfully converted to FP32" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CUDA error: " << e.what() << std::endl;
        
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
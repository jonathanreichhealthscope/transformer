#include "../include/half_precision.hpp"
#ifdef USE_CUDA
#include "../include/cuda/half_precision_kernels.cuh"
#include "../include/cuda/cuda_utils.cuh"
#endif

std::vector<half_type> HalfPrecisionTraining::half_data;

void HalfPrecisionTraining::convert_to_fp16(Matrix& matrix) {
    const size_t size = matrix.rows() * matrix.cols();
    half_data.resize(size);
    
    #ifdef USE_CUDA
        // GPU version
        half_type* d_half;
        float* d_float;
        CUDA_CHECK(cudaMalloc(&d_half, size * sizeof(half_type)));
        CUDA_CHECK(cudaMalloc(&d_float, size * sizeof(float)));
        
        // Copy input to GPU
        CUDA_CHECK(cudaMemcpy(d_float, matrix.data(), 
                             size * sizeof(float), 
                             cudaMemcpyHostToDevice));
        
        // Launch conversion
        launch_fp32_to_fp16(d_float, d_half, size);
        
        // Copy result back
        CUDA_CHECK(cudaMemcpy(half_data.data(), d_half, 
                             size * sizeof(half_type), 
                             cudaMemcpyDeviceToHost));
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_half));
        CUDA_CHECK(cudaFree(d_float));
    #else
        // CPU version - just copy since we're using floats
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            half_data[i] = matrix.data()[i];
        }
    #endif
}

void HalfPrecisionTraining::convert_to_fp32(Matrix& matrix) {
    const size_t size = matrix.rows() * matrix.cols();
    
    #ifdef USE_CUDA
        // GPU version
        float* d_float;
        half_type* d_half;
        CUDA_CHECK(cudaMalloc(&d_float, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_half, size * sizeof(half_type)));
        CUDA_CHECK(cudaMemcpy(d_half, half_data.data(), 
                             size * sizeof(half_type), 
                             cudaMemcpyHostToDevice));
        
        // Launch conversion
        launch_fp16_to_fp32(d_half, d_float, size);
            
        CUDA_CHECK(cudaMemcpy(matrix.data(), d_float, 
                             size * sizeof(float), 
                             cudaMemcpyDeviceToHost));
                             
        CUDA_CHECK(cudaFree(d_float));
        CUDA_CHECK(cudaFree(d_half));
    #else
        // CPU version - just copy since we're using floats
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            matrix.data()[i] = half_data[i];
        }
    #endif
} 
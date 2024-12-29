#include "layernorm.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "cuda/layernorm_kernels.cuh"
#include "cuda/cuda_utils.cuh"
#endif

// Only keep the CPU implementation and load/save methods here 

Matrix LayerNorm::backward(const Matrix& grad, const Matrix& input) const {
    const size_t batch_size = input.rows();
    const size_t hidden_size = input.cols();
    Matrix dx(batch_size, hidden_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        // Compute mean and variance
        float mean = 0.0f;
        float var = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            mean += input(i, j);
        }
        mean /= hidden_size;
        
        for (size_t j = 0; j < hidden_size; ++j) {
            float diff = input(i, j) - mean;
            var += diff * diff;
        }
        var /= hidden_size;
        float std = std::sqrt(var + eps);
        
        // Compute gradients
        float sum_grad = 0.0f;
        float sum_grad_diff = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            float diff = input(i, j) - mean;
            sum_grad += grad(i, j) * gamma[j];
            sum_grad_diff += grad(i, j) * gamma[j] * diff;
        }
        
        for (size_t j = 0; j < hidden_size; ++j) {
            float diff = input(i, j) - mean;
            dx(i, j) = gamma[j] * (grad(i, j) - (sum_grad + diff * sum_grad_diff / var) / hidden_size) / std;
        }
    }
    
    return dx;
} 

Matrix LayerNorm::backward_cuda(const Matrix& grad, const Matrix& input) const {
#ifdef USE_CUDA
    const size_t batch_size = input.rows();
    const size_t hidden_size = input.cols();
    Matrix dx(batch_size, hidden_size);
    
    // Allocate device memory
    float *d_grad, *d_input, *d_gamma, *d_dx;
    CUDA_CHECK(cudaMalloc(&d_grad, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, batch_size * hidden_size * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_grad, grad.data(), batch_size * hidden_size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), batch_size * hidden_size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), hidden_size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Launch kernel
    const int block_size = 256;
    const int grid_size = (batch_size + block_size - 1) / block_size;
    const size_t shared_mem_size = 4 * block_size * sizeof(float);
    
#ifdef __CUDACC__
    layer_norm_backward_kernel<<<grid_size, block_size, shared_mem_size, 0>>>(
        d_grad, d_input, d_gamma, d_dx,
        batch_size, hidden_size, eps
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#else
    throw std::runtime_error("CUDA kernel launch from non-CUDA code");
#endif
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(dx.data(), d_dx, batch_size * hidden_size * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_dx));
    
    return dx;
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
} 
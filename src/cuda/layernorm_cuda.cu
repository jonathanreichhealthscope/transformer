#include "cuda/layernorm_kernels.cuh"
#include "layernorm.hpp"

Matrix LayerNorm::forward_cuda(const Matrix& x) {
    const size_t batch_size = x.rows();
    const size_t hidden_size = x.cols();
    Matrix output(batch_size, hidden_size);
    
    // Allocate device memory
    float *d_input, *d_output, *d_mean, *d_variance, *d_gamma, *d_beta;
    cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_output, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_mean, batch_size * sizeof(float));
    cudaMalloc(&d_variance, batch_size * sizeof(float));
    cudaMalloc(&d_gamma, hidden_size * sizeof(float));
    cudaMalloc(&d_beta, hidden_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, x.data(), batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernels
    const int block_size = 256;
    const int grid_size = (batch_size + block_size - 1) / block_size;
    
    layer_norm_stats_kernel<<<grid_size, block_size, 2 * block_size * sizeof(float)>>>(
        d_input, d_mean, d_variance, hidden_size, batch_size
    );
    
    const int norm_grid_size = (batch_size * hidden_size + block_size - 1) / block_size;
    layer_norm_kernel<<<norm_grid_size, block_size>>>(
        d_input, d_mean, d_variance, d_gamma, d_beta,
        d_output, hidden_size, batch_size, eps
    );
    
    // Copy result back to host
    cudaMemcpy(output.data(), d_output, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_variance);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    
    return output;
} 
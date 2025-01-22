#define USE_CUDA
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/layer_norm.hpp"
#include <cuda_runtime.h>

#ifdef USE_CUDA

namespace cuda {

__global__ void layernorm_backward_kernel(const float* grad_output, const float* input,
                                          const float* gamma, float* grad_gamma,
                                          float* grad_beta, int batch_size, int hidden_size,
                                          float eps) {
    extern __shared__ float shared_mem[];
    float* mean = shared_mem;
    float* variance = &shared_mem[batch_size];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Compute mean and variance for each sequence position
    if (tid < hidden_size) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float val = input[i * hidden_size + tid];
            sum += val;
            sq_sum += val * val;
        }
        mean[tid] = sum / batch_size;
        variance[tid] = sq_sum / batch_size - mean[tid] * mean[tid] + eps;
    }
    __syncthreads();

    // Compute gradients
    for (int i = tid; i < batch_size * hidden_size; i += blockDim.x) {
        int seq_pos = i / hidden_size;
        int hidden_pos = i % hidden_size;

        float x = input[i];
        float dy = grad_output[i];
        float mu = mean[hidden_pos];
        float var = variance[hidden_pos];
        float std = sqrt(var);
        float gamma_val = gamma[hidden_pos];

        // Gradient with respect to gamma and beta
        atomicAdd(&grad_gamma[hidden_pos], dy * (x - mu) / std);
        atomicAdd(&grad_beta[hidden_pos], dy);
    }
}

void layer_norm_backward(const Matrix& grad_output, const Matrix& input,
                              const Matrix& gamma, Matrix& grad_gamma,
                              Matrix& grad_beta, float eps) {
    const int hidden_size = input.cols();
    const int batch_size = input.rows();
    
    // Allocate device memory
    float *d_input = nullptr;
    float *d_grad_output = nullptr;
    float *d_gamma = nullptr;
    float *d_grad_gamma = nullptr;
    float *d_grad_beta = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_gamma, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_beta, hidden_size * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), batch_size * hidden_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_output, grad_output.data(),
                         batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), hidden_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((batch_size * hidden_size + block.x - 1) / block.x);
    size_t shared_mem_size = 2 * hidden_size * sizeof(float);
    
    layernorm_backward_kernel<<<grid, block, shared_mem_size>>>(
        d_grad_output, d_input, d_gamma, d_grad_gamma, d_grad_beta,
        batch_size, hidden_size, eps);
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(grad_gamma.data(), d_grad_gamma, hidden_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grad_beta.data(), d_grad_beta, hidden_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_grad_gamma));
    CUDA_CHECK(cudaFree(d_grad_beta));
}

void layer_norm_forward(const Matrix& input, const Matrix& gamma, const Matrix& beta,
                          Matrix& output, float eps) {
    const int batch_size = input.rows();
    const int hidden_size = input.cols();
    
    float* d_input, *d_gamma, *d_beta, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, gamma.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, beta.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output.size() * sizeof(float)));
    
    // ... implementation details ...
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_output));
}

} // namespace cuda

#endif // USE_CUDA
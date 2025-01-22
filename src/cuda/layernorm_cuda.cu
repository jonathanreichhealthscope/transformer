#define USE_CUDA
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/layer_norm.hpp"
#include <cuda_runtime.h>
#include <iostream>

#ifdef USE_CUDA

namespace cuda {

__global__ void layernorm_backward_kernel(const float* grad_output, const float* input,
                                          const float* gamma, float* grad_gamma,
                                          float* grad_beta, int batch_size, int hidden_size,
                                          float eps) {

    extern __shared__ float shared_mem[];
    // Each block handles a subset of features
    float* mean = shared_mem;
    float* variance = &shared_mem[blockDim.x];  // Use blockDim.x instead of hidden_size
    
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;  // Local index within the block

    // Compute mean and variance for each sequence position
    if (feature_idx < hidden_size) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float val = input[i * hidden_size + feature_idx];
            sum += val;
            sq_sum += val * val;
        }
        mean[local_idx] = sum / batch_size;
        variance[local_idx] = sq_sum / batch_size - mean[local_idx] * mean[local_idx] + eps;
    }
    __syncthreads();

    // Update gradient computation to use local indices
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        if (feature_idx < hidden_size) {
            int idx = batch_idx * hidden_size + feature_idx;
            float x = input[idx];
            float dy = grad_output[idx];
            float mu = mean[local_idx];
            float var = variance[local_idx];
            float std = sqrt(var);
            float gamma_val = gamma[feature_idx];

            atomicAdd(&grad_gamma[feature_idx], dy * (x - mu) / std);
            atomicAdd(&grad_beta[feature_idx], dy);
        }
    }
}

void layer_norm_backward(const Matrix& grad_output, const Matrix& input,
                         const Matrix& gamma, Matrix& grad_gamma,
                         Matrix& grad_beta, float eps) {
    std::cout << "\n=== LayerNorm Backward Debug ===" << std::endl << std::flush;
    std::cout << "grad_output dims: " << grad_output.rows() << "x" << grad_output.cols() << std::endl << std::flush;
    std::cout << "input dims: " << input.rows() << "x" << input.cols() << std::endl << std::flush;
    std::cout << "gamma size: " << gamma.size() << std::endl;
    std::cout << "grad_gamma size: " << grad_gamma.size() << std::endl;
    std::cout << "grad_beta size: " << grad_beta.size() << std::endl;

    int batch_size = input.rows();
    int hidden_size = input.cols();

    std::cout << "Allocating device memory..." << std::endl;
    // Allocate device memory
    float *d_grad_output, *d_input, *d_gamma;
    float *d_grad_gamma, *d_grad_beta;
    std::cout << "Allocating device memory..." << std::endl;
    std::cout << "batch_size: " << batch_size << ", hidden_size: " << hidden_size << std::endl;
    std::cout << "grad_output size: " << grad_output.size() << std::endl;
    std::cout << "input size: " << input.size() << std::endl;
    std::cout << "gamma size: " << gamma.size() << std::endl;
    std::cout << "grad_gamma size: " << grad_gamma.size() << std::endl;
    std::cout << "grad_beta size: " << grad_beta.size() << std::endl;
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_gamma, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_beta, hidden_size * sizeof(float)));

    std::cout << "Copying data to device..." << std::endl;
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_grad_output, grad_output.data(), 
                        batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), 
                        batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), 
                        hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Launching kernel..." << std::endl;
    // Launch kernel
    // Use smaller block size to ensure enough blocks for all features
    dim3 block(32, 1);  // 32 threads per block is more reasonable
    // Calculate grid size to cover all features
    int num_blocks = (hidden_size + block.x - 1) / block.x;
    dim3 grid(num_blocks, 1);

    // Calculate shared memory size needed for mean and variance
    // Each block needs space for its own mean and variance arrays
    size_t shared_mem_size = 2 * block.x * sizeof(float);  // 2 arrays of 32 floats each
    
    // Verify shared memory size is sufficient
    size_t max_shared_mem = 48 * 1024;  // 48KB typical limit
    if (shared_mem_size > max_shared_mem) {
        printf("Error: Required shared memory (%zu bytes) exceeds maximum (%zu bytes)\n", 
               shared_mem_size, max_shared_mem);
        return;
    }
    
    // Zero out grad arrays before kernel launch
    CUDA_CHECK(cudaMemset(d_grad_gamma, 0, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_beta, 0, hidden_size * sizeof(float)));


    layernorm_backward_kernel<<<grid, block, shared_mem_size>>>(
        d_grad_output, d_input, d_gamma, d_grad_gamma, d_grad_beta,
        batch_size, hidden_size, eps);
    
    // Ensure kernel completion before proceeding
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    std::cout << "Copying results back to host..." << std::endl;
    std::cout << "Copying grad_gamma..." << std::endl;
    CUDA_CHECK(cudaMemcpy(grad_gamma.data(), d_grad_gamma, hidden_size * sizeof(float),
                        cudaMemcpyDeviceToHost));
    std::cout << "Copying grad_beta..." << std::endl;
    CUDA_CHECK(cudaMemcpy(grad_beta.data(), d_grad_beta, hidden_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

    std::cout << "Freeing device memory..." << std::endl;
    // Free device memory
    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_grad_gamma));
    CUDA_CHECK(cudaFree(d_grad_beta));
    std::cout << "=== LayerNorm Backward Complete ===" << std::endl;
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
#include "../../include/feed_forward.hpp"
#include "../include/cuda/cuda_check.cuh"
#include "../include/cuda/cuda_launch.cuh"
#include "../include/cuda/feed_forward_kernels.cuh"

Matrix FeedForward::backward_cuda(const Matrix& grad, const Matrix& input) const {
    const size_t batch_size = grad.rows();
    const size_t hidden_size = grad.cols();
    const size_t intermediate_size = w1.cols();

    // Allocate device memory
    float *d_grad, *d_w2, *d_intermediate, *d_input, *d_w1, *d_dx;
    CUDA_CHECK(cudaMalloc(&d_grad, batch_size * hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, hidden_size * intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_intermediate, batch_size * intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w1, hidden_size * intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, batch_size * hidden_size * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_grad, grad.data(), batch_size * hidden_size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, w2.data(), hidden_size * intermediate_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch kernels
    const int block_size = 256;
    const int grid_size = (batch_size * hidden_size + block_size - 1) / block_size;
    dim3 grid(grid_size);
    dim3 block(block_size);

    // First backward pass through second linear layer
    CUDA_LAUNCH(feed_forward_backward_kernel_1, grid, block, 0, nullptr, d_grad, d_w2,
                d_intermediate, batch_size, hidden_size, intermediate_size);

    // Backward through GELU
    CUDA_LAUNCH(gelu_backward_kernel, grid, block, 0, nullptr, d_intermediate, d_input,
                batch_size * intermediate_size);

    // Second backward pass
    CUDA_LAUNCH(feed_forward_backward_kernel_2, grid, block, 0, nullptr, d_intermediate, d_w1, d_dx,
                batch_size, hidden_size, intermediate_size);

    // Copy result back to host
    Matrix dx(batch_size, hidden_size);
    CUDA_CHECK(cudaMemcpy(dx.data(), d_dx, batch_size * hidden_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_w2));
    CUDA_CHECK(cudaFree(d_intermediate));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_dx));

    return dx;
}
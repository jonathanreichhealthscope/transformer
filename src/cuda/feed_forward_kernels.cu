#include "../../include/cuda/feed_forward_kernels.cuh"

__global__ void feed_forward_backward_kernel_1(const float* grad, const float* w2,
                                             float* d_intermediate,
                                             const int batch_size,
                                             const int hidden_size,
                                             const int intermediate_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * intermediate_size) {
        int batch_idx = idx / intermediate_size;
        int inter_idx = idx % intermediate_size;
        
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; ++i) {
            sum += grad[batch_idx * hidden_size + i] * w2[inter_idx * hidden_size + i];
        }
        d_intermediate[idx] = sum;
    }
}

__global__ void gelu_backward_kernel(const float* d_intermediate, float* d_input,
                                   const int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float x = d_input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.797884f * (x + 0.044715f * x * x * x)));
        float pdf = 0.797884f * (1.0f - tanhf(0.797884f * x) * tanhf(0.797884f * x));
        d_input[idx] = d_intermediate[idx] * (cdf + x * pdf);
    }
}

__global__ void feed_forward_backward_kernel_2(const float* d_intermediate,
                                             const float* w1, float* d_dx,
                                             const int batch_size,
                                             const int hidden_size,
                                             const int intermediate_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * hidden_size) {
        int batch_idx = idx / hidden_size;
        int hidden_idx = idx % hidden_size;
        
        float sum = 0.0f;
        for (int i = 0; i < intermediate_size; ++i) {
            sum += d_intermediate[batch_idx * intermediate_size + i] * w1[hidden_idx * intermediate_size + i];
        }
        d_dx[idx] = sum;
    }
} 
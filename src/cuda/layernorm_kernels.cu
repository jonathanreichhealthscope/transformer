#include "layernorm_kernels.cuh"

namespace cuda {

__global__ void layer_norm_stats_kernel(const float* input, float* mean, float* variance,
                                        const int hidden_size, const int batch_size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const float MIN_VAR = 1e-6f;  // Minimum variance threshold

    extern __shared__ float shared_mem[];
    float* shared_sum = shared_mem;
    float* shared_sq_sum = shared_mem + blockDim.x;

    for (int batch_idx = tid; batch_idx < batch_size; batch_idx += stride) {
        float sum = 0.0f;
        float sq_sum = 0.0f;

        // Compute sum and squared sum for this sequence
        for (int j = 0; j < hidden_size; ++j) {
            float val = input[batch_idx * hidden_size + j];
            sum += val;
            sq_sum += val * val;
        }

        mean[batch_idx] = sum / hidden_size;
        float var = (sq_sum / hidden_size) - (mean[batch_idx] * mean[batch_idx]);
        variance[batch_idx] = max(var, MIN_VAR);  // Apply minimum variance threshold
    }
}

__global__ void layer_norm_kernel(const float* input, const float* mean, const float* variance,
                                  const float* gamma, const float* beta, float* output,
                                  const int hidden_size, const int batch_size, const float eps) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total_elements = batch_size * hidden_size;
    const float MIN_VAR = 1e-6f;  // Minimum variance threshold

    for (int idx = tid; idx < total_elements; idx += stride) {
        const int batch_idx = idx / hidden_size;
        const int hidden_idx = idx % hidden_size;

        float val = input[idx];
        float mean_val = mean[batch_idx];
        float var_val = max(variance[batch_idx], MIN_VAR);  // Apply minimum variance threshold
        float std_dev = sqrt(var_val + eps);

        output[idx] = gamma[hidden_idx] * ((val - mean_val) / std_dev) + beta[hidden_idx];
    }
}

} // namespace cuda
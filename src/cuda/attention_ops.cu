#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../include/cuda/attention_ops.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/kernel_declarations.cuh"

namespace cuda {
    // Forward declare kernels
    CUDA_KERNEL void attention_scores_kernel(const float* Q, const float* K, float* scores,
                                                      float scale, int seq_len, int head_dim);
    CUDA_KERNEL void softmax_kernel(float* matrix, int rows, int cols);
    CUDA_KERNEL void attention_kernel(const float* Q, const float* K, const float* V,
                                   float* output, int batch_size, int seq_len, int head_dim);

    void compute_attention_scores(const Matrix& Q, const Matrix& K, Matrix& scores, float scale, int num_heads) {
        // Q and K are [batch_size x hidden_dim]
        int batch_size = Q.rows();
        int hidden_dim = Q.cols();
        int head_dim = hidden_dim / num_heads;
        int seq_len = batch_size;  // In this case, seq_len is same as batch_size
        
        // Verify scores dimensions
        if (scores.rows() != batch_size || scores.cols() != seq_len) {
            throw std::runtime_error("Scores matrix has wrong dimensions: expected " +
                std::to_string(batch_size) + "x" + std::to_string(seq_len) + " got " +
                std::to_string(scores.rows()) + "x" + std::to_string(scores.cols()));
        }

        // Launch kernel with correct dimensions
        dim3 block(32, 32);
        dim3 grid((seq_len + 31) / 32, (seq_len + 31) / 32);
        
        float* d_Q, *d_K, *d_scores;
        size_t Q_size = Q.size() * sizeof(float);
        size_t K_size = K.size() * sizeof(float);
        size_t scores_size = scores.size() * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_Q, Q_size));
        CUDA_CHECK(cudaMalloc(&d_K, K_size));
        CUDA_CHECK(cudaMalloc(&d_scores, scores_size));

        CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), Q_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, K.data(), K_size, cudaMemcpyHostToDevice));
        
        attention_scores_kernel<<<grid, block>>>(d_Q, d_K, d_scores,
            scale, seq_len, head_dim);

        CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, scores_size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_scores));
    }

    void apply_softmax(Matrix& matrix) {
        float* d_matrix;
        size_t size = matrix.size() * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_matrix, size));
        CUDA_CHECK(cudaMemcpy(d_matrix, matrix.data(), size, cudaMemcpyHostToDevice));

        softmax_kernel<<<matrix.rows(), 1>>>(d_matrix, matrix.rows(), matrix.cols());

        CUDA_CHECK(cudaMemcpy(matrix.data(), d_matrix, size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_matrix));
    }

    void attention_forward(const Matrix& Q, const Matrix& K, const Matrix& V, 
                         Matrix& output, int batch_size, int num_heads, int seq_len) {
        dim3 grid(batch_size, num_heads);
        
        float *d_Q, *d_K, *d_V, *d_output;
        size_t QKV_size = Q.size() * sizeof(float);
        size_t output_size = output.size() * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_Q, QKV_size));
        CUDA_CHECK(cudaMalloc(&d_K, QKV_size));
        CUDA_CHECK(cudaMalloc(&d_V, QKV_size));
        CUDA_CHECK(cudaMalloc(&d_output, output_size));

        CUDA_CHECK(cudaMemcpy(d_Q, Q.data(), QKV_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, K.data(), QKV_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, V.data(), QKV_size, cudaMemcpyHostToDevice));

        attention_kernel<<<grid, seq_len>>>(d_Q, d_K, d_V, d_output, 
                                          batch_size, seq_len, Q.cols() / num_heads);

        CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_Q));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_output));
    }

    // CUDA kernel launcher
    void launch_attention_scores_kernel(const float* Q, const float* K, float* scores, float scale,
                                      int seq_len, int head_dim, cudaStream_t stream) {
        dim3 block_dim(16, 16);
        dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x,
                      (seq_len + block_dim.y - 1) / block_dim.y);

        attention_scores_kernel<<<grid_dim, block_dim, 0, stream>>>(Q, K, scores, scale, seq_len,
                                                                 head_dim);
    }
}

// Kernel implementations
CUDA_KERNEL void attention_scores_kernel(const float* Q, const float* K, float* scores,
                                                    float scale, int seq_len, int head_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            sum += Q[row * head_dim + i] * K[col * head_dim + i];
        }
        scores[row * seq_len + col] = sum * scale;
    }
}

CUDA_KERNEL void softmax_kernel(float* matrix, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        // Find max for numerical stability
        float max_val = matrix[row * cols];
        for (int i = 1; i < cols; i++) {
            max_val = max(max_val, matrix[row * cols + i]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            matrix[row * cols + i] = expf(matrix[row * cols + i] - max_val);
            sum += matrix[row * cols + i];
        }

        // Normalize
        for (int i = 0; i < cols; i++) {
            matrix[row * cols + i] /= sum;
        }
    }
}

CUDA_KERNEL void attention_kernel(const float* Q, const float* K, const float* V,
                               float* output, int batch_size, int seq_len, int head_dim) {
    int b = blockIdx.x;  // batch index
    int h = blockIdx.y;  // head index
    int i = threadIdx.x; // sequence position

    if (i < seq_len) {
        // Compute attention scores
        float scores[1024];  // Assuming max sequence length
        for (int j = 0; j < seq_len; j++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += Q[b * seq_len * head_dim + i * head_dim + d] *
                        K[b * seq_len * head_dim + j * head_dim + d];
            }
            scores[j] = score / sqrtf(float(head_dim));
        }

        // Apply softmax
        float max_score = scores[0];
        for (int j = 1; j < seq_len; j++) {
            max_score = max(max_score, scores[j]);
        }

        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[j] = expf(scores[j] - max_score);
            sum += scores[j];
        }

        for (int j = 0; j < seq_len; j++) {
            scores[j] /= sum;
        }

        // Compute weighted sum
        for (int d = 0; d < head_dim; d++) {
            float weighted_sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                weighted_sum += scores[j] * V[b * seq_len * head_dim + j * head_dim + d];
            }
            output[b * seq_len * head_dim + i * head_dim + d] = weighted_sum;
        }
    }
} 
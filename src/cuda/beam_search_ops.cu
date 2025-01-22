#include "../../include/cuda/beam_search_ops.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/kernel_declarations.cuh"
#include <cuda_runtime.h>

namespace cuda {
    // CUDA kernels
    CUDA_KERNEL void topk_kernel(const float* scores, float* output_scores, 
                               int* output_indices, int n, int k) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n) return;

        // Each thread maintains its own top-k values and indices
        float local_topk[32];  // Assuming k <= 32 for simplicity
        int local_indices[32];

        // Initialize with smallest possible value
        for (int i = 0; i < k; i++) {
            local_topk[i] = -INFINITY;
            local_indices[i] = -1;
        }

        // Process scores in chunks
        for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
            float val = scores[i];
            
            // Insert into local top-k
            for (int j = 0; j < k; j++) {
                if (val > local_topk[j]) {
                    // Shift down existing values
                    for (int m = k - 1; m > j; m--) {
                        local_topk[m] = local_topk[m-1];
                        local_indices[m] = local_indices[m-1];
                    }
                    local_topk[j] = val;
                    local_indices[j] = i;
                    break;
                }
            }
        }

        // Write results to global memory
        if (tid < k) {
            output_scores[tid] = local_topk[tid];
            output_indices[tid] = local_indices[tid];
        }
    }

    CUDA_KERNEL void beam_search_step_kernel(const float* current_scores, const float* next_scores,
                                          float* output_scores, int* output_indices,
                                          int batch_size, int vocab_size, int beam_width) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= batch_size * beam_width) return;

        int batch_idx = tid / beam_width;
        int beam_idx = tid % beam_width;

        // Compute combined scores for each vocabulary item
        float current_score = current_scores[tid];
        float max_score = -INFINITY;
        int max_idx = -1;

        for (int v = 0; v < vocab_size; v++) {
            float score = current_score + next_scores[batch_idx * vocab_size + v];
            if (score > max_score) {
                max_score = score;
                max_idx = v;
            }
        }

        // Store results
        output_scores[tid] = max_score;
        output_indices[tid] = max_idx;
    }

    void topk(const std::vector<float>& scores, Matrix& output_scores, 
              std::vector<int>& output_indices, int k) {
        float* d_scores;
        float* d_output_scores;
        int* d_output_indices;
        
        size_t scores_size = scores.size() * sizeof(float);
        size_t output_size = k * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_scores, scores_size));
        CUDA_CHECK(cudaMalloc(&d_output_scores, output_size));
        CUDA_CHECK(cudaMalloc(&d_output_indices, k * sizeof(int)));
        
        CUDA_CHECK(cudaMemcpy(d_scores, scores.data(), scores_size, cudaMemcpyHostToDevice));
        
        // Launch top-k kernel
        dim3 block(256);
        dim3 grid((scores.size() + block.x - 1) / block.x);
        topk_kernel<<<grid, block>>>(d_scores, d_output_scores, d_output_indices, 
                                   scores.size(), k);
        
        CUDA_CHECK(cudaMemcpy(output_scores.data(), d_output_scores, output_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(output_indices.data(), d_output_indices, k * sizeof(int), cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaFree(d_scores));
        CUDA_CHECK(cudaFree(d_output_scores));
        CUDA_CHECK(cudaFree(d_output_indices));
    }

    void beam_search_step(const Matrix& current_scores, const Matrix& next_scores,
                         Matrix& output_scores, std::vector<int>& output_indices, int beam_width) {
        float* d_current;
        float* d_next;
        float* d_output;
        int* d_indices;
        
        size_t current_size = current_scores.size() * sizeof(float);
        size_t next_size = next_scores.size() * sizeof(float);
        size_t output_size = output_scores.size() * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_current, current_size));
        CUDA_CHECK(cudaMalloc(&d_next, next_size));
        CUDA_CHECK(cudaMalloc(&d_output, output_size));
        CUDA_CHECK(cudaMalloc(&d_indices, beam_width * sizeof(int)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_current, current_scores.data(), current_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_next, next_scores.data(), next_size, cudaMemcpyHostToDevice));
        
        // Launch beam search kernel
        int batch_size = current_scores.rows();
        int vocab_size = next_scores.cols();
        
        dim3 block(256);
        dim3 grid((batch_size * beam_width + block.x - 1) / block.x);
        beam_search_step_kernel<<<grid, block>>>(d_current, d_next, d_output, d_indices,
                                               batch_size, vocab_size, beam_width);
        
        // Copy results back
        CUDA_CHECK(cudaMemcpy(output_scores.data(), d_output, output_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(output_indices.data(), d_indices, beam_width * sizeof(int), cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_next));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_indices));
    }
} 
#pragma once
#include <cuda_runtime.h>

// Ensure proper linkage for CUDA kernels
#ifdef __CUDACC__
#define CUDA_KERNEL extern "C" __global__
#else
#define CUDA_KERNEL
#endif

namespace cuda {
    // Attention kernels
    CUDA_KERNEL void attention_scores_kernel(const float* queries, const float* keys,
                                           float* scores, float scale,
                                           int seq_len, int head_dim);
    
    CUDA_KERNEL void softmax_kernel(float* matrix, int rows, int cols);
    
    CUDA_KERNEL void attention_kernel(const float* Q, const float* K, const float* V,
                                    float* output, int batch_size, int seq_len, 
                                    int head_dim, int hidden_dim);

    // Beam search kernels
    CUDA_KERNEL void topk_kernel(const float* scores, float* output_scores, 
                                int* output_indices, int n, int k);
    
    CUDA_KERNEL void beam_search_step_kernel(const float* current_scores, 
                                           const float* next_scores,
                                           float* output_scores, int* output_indices,
                                           int batch_size, int vocab_size, int beam_width);

    // Tokenizer kernels
    CUDA_KERNEL void parallel_tokenize_kernel(const char* text, size_t text_len,
                                            const char* vocab_data, const int* vocab_lengths,
                                            size_t vocab_size, int* output_tokens, size_t* positions);
} 
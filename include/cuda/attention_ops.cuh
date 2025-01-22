#pragma once
#include "../matrix.hpp"
#include <cuda_runtime.h>
#include "kernel_declarations.cuh"

namespace cuda {
    // Attention operation wrappers
    void compute_attention_scores(const Matrix& Q, const Matrix& K, Matrix& scores, float scale, int num_heads = 1);
    void apply_softmax(Matrix& matrix);
    void attention_forward(const Matrix& Q, const Matrix& K, const Matrix& V, 
                         Matrix& output, int batch_size, int num_heads, int seq_len);

    // CUDA kernel launcher
    void launch_attention_scores_kernel(const float* Q, const float* K, float* scores, float scale,
                                      int seq_len, int head_dim, cudaStream_t stream);
} 
#pragma once
#include "../matrix.hpp"
#include <cuda_runtime.h>

namespace cuda {
    void initialize_cuda();
    void cleanup_cuda();
    void launch_softmax_kernel(float* scores, int seq_len, cudaStream_t stream);
}

#ifdef CUDA_AVAILABLE
Matrix cuda_matmul(const Matrix& A, const Matrix& B);
void launch_attention_scores(const float* Q, const float* K, float* scores, float scale,
                             int seq_len, int head_dim, cudaStream_t stream);
void launch_softmax(float* scores, int seq_len, cudaStream_t stream);
#endif
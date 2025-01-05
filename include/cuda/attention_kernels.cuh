#pragma once
#include <cuda_runtime.h>

void launch_attention_scores(const float* Q, const float* K,
                           float* scores, float scale,
                           int seq_len, int head_dim,
                           cudaStream_t stream);

void launch_softmax(float* scores, int seq_len,
                   cudaStream_t stream); 
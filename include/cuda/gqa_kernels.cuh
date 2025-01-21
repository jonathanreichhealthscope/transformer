#pragma once
#include <cuda_runtime.h>

void launch_gqa_kernel(const float* query, const float* key, const float* value, float* output,
                       const int batch_size, const int num_heads, const int num_kv_heads,
                       const int seq_len, const int head_dim, cudaStream_t stream);
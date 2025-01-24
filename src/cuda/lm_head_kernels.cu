#include "../../include/lm_head.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {
// CUDA kernels
__global__ void convert_to_fp16_kernel(half* output, const float* input, size_t size) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        LanguageModelHead::convert_to_fp16_kernel(output, input, idx);
    }
}

__global__ void convert_and_expand_vocab_kernel(
    float* output, const half* input, const unsigned char* active_tokens,
    size_t batch_size, size_t vocab_size, size_t active_vocab_size) {
    
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < batch_size && col < vocab_size) {
        LanguageModelHead::convert_and_expand_vocab_kernel(
            output, input, active_tokens, row, col, batch_size, vocab_size, active_vocab_size);
    }
}
} // anonymous namespace

// Device function implementations
__device__ void LanguageModelHead::convert_to_fp16_kernel(
    half* output, const float* input, size_t idx) {
    output[idx] = __float2half(input[idx]);
}

__device__ void LanguageModelHead::convert_and_expand_vocab_kernel(
    float* output, const half* input, const unsigned char* active_tokens,
    size_t row, size_t col, size_t batch_size, size_t vocab_size, size_t active_vocab_size) {
    
    // Find the position in the compressed active vocabulary
    size_t active_pos = 0;
    bool is_active = false;
    
    for (size_t i = 0; i < col; i++) {
        if (active_tokens[i]) {
            active_pos++;
        }
    }
    
    is_active = active_tokens[col];
    
    if (is_active) {
        output[row * vocab_size + col] = __half2float(input[row * active_vocab_size + active_pos]);
    } else {
        output[row * vocab_size + col] = -INFINITY;
    }
}

// Kernel launcher implementations
void LanguageModelHead::launch_convert_to_fp16(half* output, const float* input, size_t size) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    convert_to_fp16_kernel<<<num_blocks, block_size>>>(output, input, size);
    CUDA_CHECK(cudaGetLastError());
}

void LanguageModelHead::launch_convert_and_expand_vocab(
    float* output, const half* input, size_t batch_size, size_t vocab_size, size_t active_vocab_size) {
    
    const dim3 block_size(16, 16);
    const dim3 num_blocks(
        (batch_size + block_size.x - 1) / block_size.x,
        (vocab_size + block_size.y - 1) / block_size.y
    );
    
    convert_and_expand_vocab_kernel<<<num_blocks, block_size>>>(
        output, input, d_active_tokens, batch_size, vocab_size, active_vocab_size);
    CUDA_CHECK(cudaGetLastError());
} 
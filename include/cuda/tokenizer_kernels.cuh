#pragma once
#include <cuda_runtime.h>
#include "../vocabulary.hpp"
#include <vector>
#include <string>
#include "kernel_declarations.cuh"

// Maximum length of a token in characters
constexpr size_t MAX_TOKEN_LENGTH = 64;

namespace cuda {
    // Forward declaration of CUDA kernel
    __global__ void parallel_tokenize_kernel(const char* text, size_t text_len,
                                           const char* vocab_data, const int* vocab_lengths,
                                           size_t vocab_size, int* output_tokens, size_t* positions);

    void parallel_tokenize(const std::string& text, const Vocabulary& vocab, std::vector<int>& tokens);
} 
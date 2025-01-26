#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "../../third_party/tiktoken/tiktoken/tiktoken.hpp"

namespace cuda {

// Device-side hash table for vocabulary and merges
struct VocabEntry {
    char token[32];  // Fixed size for simplicity
    int id;
};

// Global memory pointer (allocated in the implementation)
extern __device__ VocabEntry* g_vocab;
extern __device__ int g_vocab_size;

// Device functions
__device__ bool find_longest_token(const char* text, int start_pos, int text_length, int* token_id, int* token_length);

// Global kernel
__global__ void bpe_tokenize_kernel(const char* text,
                                  int text_length,
                                  int* output_ids,
                                  int* output_length);

// Host function
void parallel_tokenize(const std::string& text,
                      const tiktoken::Encoding& tokenizer,
                      std::vector<int>& output);

} // namespace cuda 
#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <unordered_map>
#include "../../third_party/tiktoken/tiktoken/tiktoken.hpp"

namespace cuda {

// Device-side structures for vocabulary and merges
struct VocabEntry {
    unsigned char token[32];  // Raw bytes for the token
    int id;
    int length;  // Actual length of the token in bytes
};

struct BPEMerge {
    unsigned char first[32];   // First token in merge rule
    unsigned char second[32];  // Second token in merge rule
    unsigned char result[32];  // Result of merging
    int first_len;            // Length of first token
    int second_len;           // Length of second token
    int result_len;           // Length of result token
};

// Global memory pointers (allocated in the implementation)
extern __device__ VocabEntry* g_vocab;
extern __device__ int g_vocab_size;
extern __device__ BPEMerge* g_merges;
extern __device__ int g_num_merges;

// Device functions
__device__ bool find_token_match(const unsigned char* text, int start_pos, int text_length,
                               const unsigned char* token, int token_length);

__device__ bool apply_bpe_merge(unsigned char* tokens, int* token_lengths, int* num_tokens,
                               const BPEMerge* merge);

__device__ int find_longest_token(const unsigned char* text, int start_pos, int text_length,
                                 int* token_length);

// Global kernel
__global__ void bpe_tokenize_kernel(const unsigned char* text,
                                   int text_length,
                                   int* output_ids,
                                   int* output_length,
                                   int max_output_length);

// Host function
void parallel_tokenize(const std::string& text,
                      const tiktoken::Encoding& tokenizer,
                      std::vector<int>& output);

} // namespace cuda 
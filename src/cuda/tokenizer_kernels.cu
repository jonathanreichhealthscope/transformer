#include "../../include/cuda/tokenizer_kernels.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace cuda {

// Define global memory pointers
__device__ VocabEntry* g_vocab = nullptr;
__device__ int g_vocab_size = 0;
__device__ BPEMerge* g_merges = nullptr;
__device__ int g_num_merges = 0;

__device__ bool find_token_match(const unsigned char* text, int start_pos, int text_length,
                               const unsigned char* token, int token_length) {
    if (start_pos + token_length > text_length) return false;
    
    for (int i = 0; i < token_length; i++) {
        if (text[start_pos + i] != token[i]) return false;
    }
    return true;
}

__device__ bool apply_bpe_merge(unsigned char* tokens, int* token_lengths, int* num_tokens,
                               const BPEMerge* merge) {
    for (int i = 0; i < *num_tokens - 1; i++) {
        if (token_lengths[i] == merge->first_len &&
            token_lengths[i + 1] == merge->second_len &&
            find_token_match(tokens + i * 32, 0, token_lengths[i], merge->first, merge->first_len) &&
            find_token_match(tokens + (i + 1) * 32, 0, token_lengths[i + 1], merge->second, merge->second_len)) {
            
            // Apply merge
            for (int j = 0; j < merge->result_len; j++) {
                tokens[i * 32 + j] = merge->result[j];
            }
            token_lengths[i] = merge->result_len;
            
            // Remove second token
            for (int j = i + 1; j < *num_tokens - 1; j++) {
                for (int k = 0; k < token_lengths[j + 1]; k++) {
                    tokens[j * 32 + k] = tokens[(j + 1) * 32 + k];
                }
                token_lengths[j] = token_lengths[j + 1];
            }
            (*num_tokens)--;
            return true;
        }
    }
    return false;
}

__device__ int find_longest_token(const unsigned char* text, int start_pos, int text_length,
                                 int* token_length) {
    int best_id = -1;
    *token_length = 0;
    
    for (int i = 0; i < g_vocab_size; i++) {
        if (find_token_match(text, start_pos, text_length, g_vocab[i].token, g_vocab[i].length) &&
            g_vocab[i].length > *token_length) {
            *token_length = g_vocab[i].length;
            best_id = g_vocab[i].id;
        }
    }
    
    return best_id;
}

__global__ void bpe_tokenize_kernel(const unsigned char* text,
                                   int text_length,
                                   int* output_ids,
                                   int* output_length,
                                   int max_output_length) {
    __shared__ unsigned char shared_tokens[32 * 256];  // Space for 256 tokens
    __shared__ int shared_token_lengths[256];
    __shared__ int shared_num_tokens;
    
    if (threadIdx.x == 0) {
        shared_num_tokens = 0;
        *output_length = 0;
    }
    __syncthreads();
    
    // First break input into initial tokens
    for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < text_length; pos += blockDim.x * gridDim.x) {
        int token_length;
        int token_id = find_longest_token(text, pos, text_length, &token_length);
        
        if (token_id != -1) {
            int token_idx = atomicAdd(&shared_num_tokens, 1);
            if (token_idx < 256) {
                // Copy token bytes to shared memory
                for (int i = 0; i < token_length; i++) {
                    shared_tokens[token_idx * 32 + i] = text[pos + i];
                }
                shared_token_lengths[token_idx] = token_length;
            }
            pos += token_length - 1;
        }
    }
    __syncthreads();
    
    // Apply BPE merges
    if (threadIdx.x == 0) {
        bool changed;
        do {
            changed = false;
            for (int i = 0; i < g_num_merges && shared_num_tokens > 1; i++) {
                if (apply_bpe_merge(shared_tokens, shared_token_lengths, &shared_num_tokens, &g_merges[i])) {
                    changed = true;
                    break;
                }
            }
        } while (changed && shared_num_tokens > 1);
        
        // Convert final tokens to IDs
        int out_pos = 0;
        for (int i = 0; i < shared_num_tokens && out_pos < max_output_length; i++) {
            int token_length;
            int token_id = find_longest_token(shared_tokens + i * 32, 0, shared_token_lengths[i], &token_length);
            if (token_id != -1) {
                output_ids[out_pos++] = token_id;
            }
        }
        *output_length = out_pos;
    }
}

void parallel_tokenize(const std::string& text,
                      const tiktoken::Encoding& tokenizer,
                      std::vector<int>& output) {
    // Log input text tokenization
    std::cout << "\n=== Tokenization Details ===" << std::endl;
    std::cout << "Input text: '" << text << "'" << std::endl;
    
    // Get CPU tokenization for comparison
    auto cpu_tokens = tokenizer.encode(text);
    std::cout << "CPU Tokenization:" << std::endl;
    for (size_t i = 0; i < cpu_tokens.size(); ++i) {
        std::string token_text = tokenizer.decode({cpu_tokens[i]});
        std::cout << std::setw(5) << cpu_tokens[i] << "(" 
                  << (token_text.empty() ? "<empty>" : token_text) << ")";
        if (i < cpu_tokens.size() - 1) std::cout << " + ";
    }
    std::cout << std::endl;

    // Allocate device memory
    unsigned char* d_text;
    int* d_output;
    int* d_output_length;
    const int max_output_length = text.length() * 2;  // Conservative estimate
    
    CUDA_CHECK(cudaMalloc(&d_text, text.length()));
    CUDA_CHECK(cudaMalloc(&d_output, max_output_length * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_length, sizeof(int)));
    
    // Copy input text to device
    CUDA_CHECK(cudaMemcpy(d_text, text.c_str(), text.length(), cudaMemcpyHostToDevice));
    
    // Initialize vocabulary and merges in global memory (if not already done)
    static bool initialized = false;
    if (!initialized) {
        // Build vocabulary by testing all possible token IDs
        std::vector<VocabEntry> h_vocab;
        h_vocab.reserve(50000);  // Reserve space for a reasonable number of tokens
        
        // First, add single-byte tokens (we know these exist in GPT-2)
        for (int i = 0; i < 256; i++) {
            std::vector<int> token_ids = {i};
            std::string token_bytes = tokenizer.decode(token_ids);
            if (!token_bytes.empty()) {
                VocabEntry entry;
                memcpy(entry.token, token_bytes.data(), token_bytes.length());
                entry.length = token_bytes.length();
                entry.id = i;
                h_vocab.push_back(entry);
            }
        }
        
        // Then try to find multi-byte tokens
        for (int i = 256; i < tokenizer.get_vocab_size(); i++) {
            std::vector<int> token_ids = {i};
            std::string token_bytes = tokenizer.decode(token_ids);
            
            // Verify this is a valid token by encoding it back
            if (!token_bytes.empty() && token_bytes.length() < 32) {
                std::vector<int> encoded = tokenizer.encode(token_bytes);
                if (encoded.size() == 1 && encoded[0] == i) {
                    VocabEntry entry;
                    memcpy(entry.token, token_bytes.data(), token_bytes.length());
                    entry.length = token_bytes.length();
                    entry.id = i;
                    h_vocab.push_back(entry);
                }
            }
        }
        
        // Create merge rules from multi-byte tokens
        std::vector<BPEMerge> h_merges;
        h_merges.reserve(h_vocab.size());
        
        for (const auto& entry : h_vocab) {
            if (entry.length > 1) {
                // Create merge rule for this multi-byte token
                BPEMerge merge;
                // First part is the first byte
                merge.first[0] = entry.token[0];
                merge.first_len = 1;
                // Second part is the rest
                memcpy(merge.second, entry.token + 1, entry.length - 1);
                merge.second_len = entry.length - 1;
                // Result is the full token
                memcpy(merge.result, entry.token, entry.length);
                merge.result_len = entry.length;
                h_merges.push_back(merge);
            }
        }
        
        // Sort merges by token length (longer tokens should be merged first)
        std::sort(h_merges.begin(), h_merges.end(),
                 [](const BPEMerge& a, const BPEMerge& b) {
                     return a.result_len > b.result_len;
                 });
        
        // Allocate and copy to device
        VocabEntry* d_vocab;
        BPEMerge* d_merges;
        CUDA_CHECK(cudaMalloc(&d_vocab, h_vocab.size() * sizeof(VocabEntry)));
        CUDA_CHECK(cudaMalloc(&d_merges, h_merges.size() * sizeof(BPEMerge)));
        
        CUDA_CHECK(cudaMemcpy(d_vocab, h_vocab.data(), h_vocab.size() * sizeof(VocabEntry), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_merges, h_merges.data(), h_merges.size() * sizeof(BPEMerge), cudaMemcpyHostToDevice));
        
        // Set global pointers
        int vocab_size = h_vocab.size();
        int num_merges = h_merges.size();
        CUDA_CHECK(cudaMemcpyToSymbol(g_vocab, &d_vocab, sizeof(VocabEntry*)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_vocab_size, &vocab_size, sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_merges, &d_merges, sizeof(BPEMerge*)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_num_merges, &num_merges, sizeof(int)));
        
        initialized = true;
        
        std::cout << "Initialized CUDA tokenizer with " << vocab_size << " vocabulary entries and "
                  << num_merges << " merge rules" << std::endl;
    }
    
    // Launch kernel
    int block_size = 256;
    int num_blocks = (text.length() + block_size - 1) / block_size;
    bpe_tokenize_kernel<<<num_blocks, block_size>>>(
        d_text,
        text.length(),
        d_output,
        d_output_length,
        max_output_length
    );
    
    // Get output length
    int output_length;
    CUDA_CHECK(cudaMemcpy(&output_length, d_output_length, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Copy results back
    output.resize(output_length);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_length * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Cleanup temporary allocations
    CUDA_CHECK(cudaFree(d_text));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_output_length));

    // Log GPU tokenization results
    std::cout << "GPU Tokenization:" << std::endl;
    for (size_t i = 0; i < output.size(); ++i) {
        std::string token_text = tokenizer.decode({output[i]});
        std::cout << std::setw(5) << output[i] << "(" 
                  << (token_text.empty() ? "<empty>" : token_text) << ")";
        if (i < output.size() - 1) std::cout << " + ";
    }
    std::cout << "\n=== End Tokenization Details ===\n" << std::endl;

    // Compare CPU and GPU results
    if (cpu_tokens != output) {
        std::cout << "WARNING: CPU and GPU tokenization differ!" << std::endl;
        std::cout << "CPU tokens: ";
        for (auto t : cpu_tokens) std::cout << t << " ";
        std::cout << "\nGPU tokens: ";
        for (auto t : output) std::cout << t << " ";
        std::cout << std::endl;
    }
}

} // namespace cuda 
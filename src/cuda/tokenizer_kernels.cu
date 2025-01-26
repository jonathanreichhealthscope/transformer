#include "../../include/cuda/tokenizer_kernels.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda {

// Define global memory pointers
__device__ VocabEntry* g_vocab = nullptr;
__device__ int g_vocab_size = 0;

__device__ bool is_byte_boundary(const char* text, int pos) {
    return (text[pos] & 0xC0) != 0x80;  // Check if it's not a UTF-8 continuation byte
}

__device__ bool find_longest_token(const char* text, int start_pos, int text_length, int* token_id, int* token_length) {
    *token_length = 0;
    *token_id = -1;
    
    // First try multi-byte tokens
    for (int i = 0; i < g_vocab_size; i++) {
        const unsigned char* vocab_token = reinterpret_cast<const unsigned char*>(g_vocab[i].token);
        int j = 0;
        bool match = true;
        
        while (start_pos + j < text_length && vocab_token[j] != 0) {
            if (static_cast<unsigned char>(text[start_pos + j]) != vocab_token[j]) {
                match = false;
                break;
            }
            j++;
        }
        
        if (match && vocab_token[j] == 0 && j > *token_length) {
            *token_length = j;
            *token_id = g_vocab[i].id;
        }
    }
    
    // If no multi-byte token found, fall back to single byte
    if (*token_id == -1 && is_byte_boundary(text, start_pos)) {
        unsigned char byte = static_cast<unsigned char>(text[start_pos]);
        *token_length = 1;
        *token_id = byte;  // Use byte value as token ID for single bytes
    }
    
    return *token_id != -1;
}

__global__ void bpe_tokenize_kernel(const char* text,
                                  int text_length,
                                  int* output_ids,
                                  int* output_length) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    __shared__ int shared_output_pos;
    if (threadIdx.x == 0) {
        shared_output_pos = 0;
    }
    __syncthreads();
    
    for (int pos = start_idx; pos < text_length; pos += stride) {
        if (!is_byte_boundary(text, pos)) continue;  // Skip UTF-8 continuation bytes
        
        int token_id, token_length;
        if (find_longest_token(text, pos, text_length, &token_id, &token_length)) {
            int output_pos = atomicAdd(&shared_output_pos, 1);
            if (output_pos < text_length) {
                output_ids[output_pos] = token_id;
                pos += token_length - 1;  // Skip processed bytes
            }
        }
    }
    
    if (threadIdx.x == 0) {
        *output_length = shared_output_pos;
    }
}

void parallel_tokenize(const std::string& text,
                      const tiktoken::Encoding& tokenizer,
                      std::vector<int>& output) {
    // Allocate device memory
    char* d_text;
    int* d_output;
    int* d_output_length;
    
    CUDA_CHECK(cudaMalloc(&d_text, text.length()));
    CUDA_CHECK(cudaMalloc(&d_output, text.length() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_length, sizeof(int)));
    
    // Copy input text to device
    CUDA_CHECK(cudaMemcpy(d_text, text.c_str(), text.length(), cudaMemcpyHostToDevice));
    
    // Initialize vocabulary in global memory (if not already done)
    static bool vocab_initialized = false;
    if (!vocab_initialized) {
        std::vector<VocabEntry> h_vocab;
        h_vocab.reserve(50000);
        
        // First, add all single bytes as tokens
        for (int i = 0; i < 256; i++) {
            VocabEntry entry;
            entry.token[0] = static_cast<char>(i);
            entry.token[1] = '\0';
            entry.id = i;
            h_vocab.push_back(entry);
        }
        
        // Then add multi-byte tokens by encoding and decoding test strings
        std::vector<int> encoded;
        std::string decoded;
        for (int i = 0; i < 50000; i++) {
            // Try to decode the token ID to get its bytes
            std::vector<int> token_ids = {i};
            decoded = tokenizer.decode(token_ids);
            
            if (!decoded.empty() && decoded.length() < 32) {
                // Verify this is a valid token by encoding it back
                encoded = tokenizer.encode(decoded);
                if (encoded.size() == 1 && encoded[0] == i) {
                    VocabEntry entry;
                    memcpy(entry.token, decoded.data(), decoded.length());
                    entry.token[decoded.length()] = '\0';
                    entry.id = i;
                    h_vocab.push_back(entry);
                }
            }
        }
        
        // Allocate and copy vocabulary to device global memory
        size_t vocab_size = h_vocab.size();
        VocabEntry* d_vocab;
        CUDA_CHECK(cudaMalloc(&d_vocab, vocab_size * sizeof(VocabEntry)));
        CUDA_CHECK(cudaMemcpy(d_vocab, h_vocab.data(), vocab_size * sizeof(VocabEntry), cudaMemcpyHostToDevice));
        
        // Copy pointers to device globals
        int size = static_cast<int>(vocab_size);
        CUDA_CHECK(cudaMemcpyToSymbol(g_vocab, &d_vocab, sizeof(VocabEntry*)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_vocab_size, &size, sizeof(int)));
        
        vocab_initialized = true;
    }
    
    // Launch kernel
    int block_size = 256;
    int num_blocks = (text.length() + block_size - 1) / block_size;
    bpe_tokenize_kernel<<<num_blocks, block_size>>>(
        d_text,
        text.length(),
        d_output,
        d_output_length
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
}

} // namespace cuda 
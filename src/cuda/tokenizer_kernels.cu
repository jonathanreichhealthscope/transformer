#include "../../include/cuda/tokenizer_kernels.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda {

// Define global memory pointers
__device__ VocabEntry* g_vocab = nullptr;
__device__ int g_vocab_size = 0;

__device__ bool find_longest_token(const char* text, int start_pos, int text_length, int* token_id, int* token_length) {
    *token_length = 0;
    *token_id = -1;
    
    for (int i = 0; i < g_vocab_size; i++) {
        const char* vocab_token = g_vocab[i].token;
        int j = 0;
        while (start_pos + j < text_length && 
               vocab_token[j] != '\0' && 
               text[start_pos + j] == vocab_token[j]) {
            j++;
        }
        
        if (vocab_token[j] == '\0' && j > *token_length) {
            *token_length = j;
            *token_id = g_vocab[i].id;
        }
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
        int token_id, token_length;
        if (find_longest_token(text, pos, text_length, &token_id, &token_length)) {
            int output_pos = atomicAdd(&shared_output_pos, 1);
            if (output_pos < text_length) {
                output_ids[output_pos] = token_id;
                pos += token_length - 1;  // Skip processed characters
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
    static VocabEntry* d_vocab = nullptr;
    static int d_vocab_size = 0;
    
    CUDA_CHECK(cudaMalloc(&d_text, text.length()));
    CUDA_CHECK(cudaMalloc(&d_output, text.length() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_length, sizeof(int)));
    
    // Copy input text to device
    CUDA_CHECK(cudaMemcpy(d_text, text.c_str(), text.length(), cudaMemcpyHostToDevice));
    
    // Initialize vocabulary in global memory (if not already done)
    static bool vocab_initialized = false;
    if (!vocab_initialized) {
        // Create temporary vocabulary entries
        std::vector<VocabEntry> h_vocab;
        h_vocab.reserve(50000);
        
        // Process each token in the vocabulary
        std::string token;
        std::vector<int> single_token(1);
        
        // Iterate through possible token IDs
        for (size_t i = 0; i < 50000; i++) {
            single_token[0] = static_cast<int>(i);
            token = tokenizer.decode(single_token);
            
            if (!token.empty() && token.length() < 32) {
                VocabEntry entry;
                strncpy(entry.token, token.c_str(), 31);
                entry.token[31] = '\0';  // Ensure null termination
                entry.id = i;
                h_vocab.push_back(entry);
            }
        }
        
        // Allocate and copy vocabulary to device global memory
        CUDA_CHECK(cudaMalloc(&d_vocab, h_vocab.size() * sizeof(VocabEntry)));
        CUDA_CHECK(cudaMemcpy(d_vocab, h_vocab.data(), h_vocab.size() * sizeof(VocabEntry), cudaMemcpyHostToDevice));
        d_vocab_size = static_cast<int>(h_vocab.size());
        
        // Copy pointers to device globals
        void* d_vocab_ptr = static_cast<void*>(d_vocab);
        CUDA_CHECK(cudaMemcpyToSymbol(g_vocab, &d_vocab_ptr, sizeof(void*)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_vocab_size, &d_vocab_size, sizeof(int)));
        
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
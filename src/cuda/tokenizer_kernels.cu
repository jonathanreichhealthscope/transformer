#include "../../include/cuda/tokenizer_kernels.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include <cuda_runtime.h>

namespace cuda {

__global__ void find_tokens_kernel(const char* text, size_t text_len, 
                                 const char* vocab_data, const int* vocab_lengths,
                                 size_t vocab_size, int* output_tokens, size_t* positions) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= text_len) return;

    // Each thread looks for the longest token starting at its position
    size_t pos = tid;
    int longest_token = -1;  // UNK token by default
    size_t longest_len = 0;

    // Try to match each vocabulary token
    for (size_t i = 0; i < vocab_size; i++) {
        size_t token_len = vocab_lengths[i];
        if (pos + token_len > text_len) continue;

        bool match = true;
        for (size_t j = 0; j < token_len && match; j++) {
            if (text[pos + j] != vocab_data[i * MAX_TOKEN_LENGTH + j]) {
                match = false;
            }
        }

        if (match && token_len > longest_len) {
            longest_token = i;
            longest_len = token_len;
        }
    }

    output_tokens[tid] = longest_token;
    positions[tid] = longest_len;
}

void parallel_tokenize(const std::string& text, const Vocabulary& vocab, std::vector<int>& tokens) {
    const size_t text_len = text.length();
    const size_t vocab_size = vocab.size();

    // Allocate device memory
    char* d_text;
    char* d_vocab_data;
    int* d_vocab_lengths;
    int* d_output_tokens;
    size_t* d_positions;

    CUDA_CHECK(cudaMalloc(&d_text, text_len * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_vocab_data, vocab_size * MAX_TOKEN_LENGTH * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_vocab_lengths, vocab_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_tokens, text_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_positions, text_len * sizeof(size_t)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_text, text.c_str(), text_len * sizeof(char), cudaMemcpyHostToDevice));
    // TODO: Copy vocabulary data and lengths

    // Launch kernel
    dim3 block(256);
    dim3 grid((text_len + block.x - 1) / block.x);
    find_tokens_kernel<<<grid, block>>>(d_text, text_len, d_vocab_data, d_vocab_lengths,
                                      vocab_size, d_output_tokens, d_positions);

    // Copy results back
    std::vector<int> raw_tokens(text_len);
    std::vector<size_t> positions(text_len);
    CUDA_CHECK(cudaMemcpy(raw_tokens.data(), d_output_tokens, text_len * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(positions.data(), d_positions, text_len * sizeof(size_t), cudaMemcpyDeviceToHost));

    // Process results to get final tokens
    tokens.clear();
    for (size_t i = 0; i < text_len;) {
        if (raw_tokens[i] != -1) {
            tokens.push_back(raw_tokens[i]);
            i += positions[i];
        } else {
            tokens.push_back(vocab.get_unk_token_id());
            i++;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_text));
    CUDA_CHECK(cudaFree(d_vocab_data));
    CUDA_CHECK(cudaFree(d_vocab_lengths));
    CUDA_CHECK(cudaFree(d_output_tokens));
    CUDA_CHECK(cudaFree(d_positions));
}

} // namespace cuda 
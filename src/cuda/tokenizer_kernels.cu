#include "../../include/cuda/tokenizer_kernels.cuh"
#include "../../include/cuda/cuda_check.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda {

__global__ void tokenize_kernel(const char* text, 
                               int text_length,
                               int* output,
                               int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < text_length) {
        // Simple character-based tokenization for now
        output[idx] = static_cast<unsigned char>(text[idx]) % vocab_size;
    }
}

void parallel_tokenize(const std::string& text,
                      const Tokenizer& tokenizer,
                      std::vector<int>& output) {
    int text_length = text.length();
    output.resize(text_length);

    // Allocate device memory
    char* d_text;
    int* d_output;
    cudaMalloc(&d_text, text_length);
    cudaMalloc(&d_output, text_length * sizeof(int));

    // Copy input to device
    cudaMemcpy(d_text, text.c_str(), text_length, cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 256;
    int num_blocks = (text_length + block_size - 1) / block_size;
    tokenize_kernel<<<num_blocks, block_size>>>(d_text, 
                                              text_length,
                                              d_output,
                                              tokenizer.vocab_size());

    // Copy result back to host
    cudaMemcpy(output.data(), d_output, text_length * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_text);
    cudaFree(d_output);
}

} // namespace cuda 
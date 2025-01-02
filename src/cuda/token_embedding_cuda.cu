#include "../../include/cuda/cuda_check.cuh"
#include "../../include/cuda/cuda_launch.cuh"
#include "../../include/embeddings.hpp"

#ifdef __CUDACC__
// CUDA kernel for forward embedding lookup
__global__ void embedding_forward_kernel(const int *tokens,
                                         const float *embedding_table,
                                         float *output, int seq_length,
                                         int hidden_size, int vocab_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < seq_length * hidden_size; i += stride) {
    int seq_pos = i / hidden_size;
    int hidden_pos = i % hidden_size;
    int token_id = tokens[seq_pos];
    output[i] = embedding_table[token_id * hidden_size + hidden_pos];
  }
}

// CUDA kernel for projection back to vocabulary space
__global__ void embedding_project_kernel(const float *input,
                                         const float *embedding_table,
                                         float *output, int seq_length,
                                         int hidden_size, int vocab_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < seq_length * vocab_size; i += stride) {
    int seq_pos = i / vocab_size;
    int vocab_pos = i % vocab_size;
    float sum = 0.0f;
    for (int j = 0; j < hidden_size; j++) {
      sum += input[seq_pos * hidden_size + j] *
             embedding_table[vocab_pos * hidden_size + j];
    }
    sum = max(min(sum, 100.0f), -100.0f);
    output[i] = sum;
  }
}
#endif

void TokenEmbedding::forward_cuda(const std::vector<int> &tokens,
                                  Matrix &output) {
#ifdef __CUDACC__
  int seq_length = tokens.size();
  int hidden_size = get_embedding_dim();
  const Matrix &embedding_table = get_embedding_table();

  // Allocate device memory
  int *d_tokens;
  float *d_embedding_table, *d_output;

  CUDA_CHECK(cudaMalloc(&d_tokens, seq_length * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc(&d_embedding_table, embedding_table.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, seq_length * hidden_size * sizeof(float)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_tokens, tokens.data(), seq_length * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_embedding_table, embedding_table.data(),
                        embedding_table.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Launch kernel
  int block_size = 256;
  int grid_size = (seq_length * hidden_size + block_size - 1) / block_size;
  dim3 grid(grid_size);
  dim3 block(block_size);

  CUDA_LAUNCH(embedding_forward_kernel, grid, block, 0, nullptr, d_tokens,
              d_embedding_table, d_output, seq_length, hidden_size,
              get_vocab_size());

  // Copy result back to host
  output.resize(seq_length, hidden_size);
  CUDA_CHECK(cudaMemcpy(output.data(), d_output,
                        seq_length * hidden_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_tokens));
  CUDA_CHECK(cudaFree(d_embedding_table));
  CUDA_CHECK(cudaFree(d_output));
#else
  throw std::runtime_error("CUDA support not enabled");
#endif
}

Matrix TokenEmbedding::project_to_vocab_cuda(const Matrix &input) {
#ifdef __CUDACC__
  int seq_length = input.rows();
  int hidden_size = input.cols();
  int vocab_size = get_vocab_size();
  const Matrix &embedding_table = get_embedding_table();

  // Allocate device memory
  float *d_input, *d_embedding_table, *d_output;

  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc(&d_embedding_table, embedding_table.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, seq_length * vocab_size * sizeof(float)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_embedding_table, embedding_table.data(),
                        embedding_table.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Launch kernel
  int block_size = 256;
  int grid_size = (seq_length * vocab_size + block_size - 1) / block_size;
  dim3 grid(grid_size);
  dim3 block(block_size);

  CUDA_LAUNCH(embedding_project_kernel, grid, block, 0, nullptr, d_input,
              d_embedding_table, d_output, seq_length, hidden_size, vocab_size);

  // Create result matrix and copy data back
  Matrix result(seq_length, vocab_size);
  CUDA_CHECK(cudaMemcpy(result.data(), d_output, result.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_embedding_table));
  CUDA_CHECK(cudaFree(d_output));

  return result;
#else
  throw std::runtime_error("CUDA support not enabled");
#endif
}
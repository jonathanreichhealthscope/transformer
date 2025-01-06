#include "../../include/cuda/cuda_utils.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Declare the CUDA kernels
__global__ void attention_scores_kernel(const float *Q, const float *K,
                                        float *scores, const float scale,
                                        int seq_len, int head_dim) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < seq_len && col < seq_len) {
    float sum = 0.0f;
    for (int i = 0; i < head_dim; i++) {
      sum += Q[row * head_dim + i] * K[col * head_dim + i];
    }
    scores[row * seq_len + col] = sum * scale;
  }
}

__global__ void softmax_kernel(float *scores, int seq_len) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < seq_len) {
    float max_val = scores[row * seq_len];
    for (int i = 1; i < seq_len; i++) {
      max_val = max(max_val, scores[row * seq_len + i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
      scores[row * seq_len + i] = expf(scores[row * seq_len + i] - max_val);
      sum += scores[row * seq_len + i];
    }

    for (int i = 0; i < seq_len; i++) {
      scores[row * seq_len + i] /= sum;
    }
  }
}

// CUDA kernel launcher without template
void launch_attention_scores_kernel(const float *Q, const float *K,
                                    float *scores, float scale, int seq_len,
                                    int head_dim, cudaStream_t stream) {
  dim3 block_dim(16, 16);
  dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x,
                (seq_len + block_dim.y - 1) / block_dim.y);

  attention_scores_kernel<<<grid_dim, block_dim, 0, stream>>>(
      Q, K, scores, scale, seq_len, head_dim);
}

void launch_softmax_kernel(float *scores, int seq_len, cudaStream_t stream) {
  dim3 block_dim(256);
  dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x);

  softmax_kernel<<<grid_dim, block_dim, 0, stream>>>(scores, seq_len);
}

Matrix cuda_matmul(const Matrix &A, const Matrix &B) {
  std::cout << "Starting CUDA matrix multiplication..." << std::endl;
  std::cout << "Matrix A: " << A.rows() << "x" << A.cols() << std::endl;
  std::cout << "Matrix B: " << B.rows() << "x" << B.cols() << std::endl;

  cublasHandle_t handle;
  cublasStatus_t status;
  cudaError_t err;

  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create cuBLAS handle");
  }

  float alpha = 1.0f;
  float beta = 0.0f;

  Matrix C(A.rows(), B.cols(), 0.0f);
  Matrix C_gpu = C.to_gpu();

  try {
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B.cols(), A.rows(),
                         A.cols(), &alpha, B.get_data(), B.cols(), A.get_data(),
                         A.cols(), &beta, C_gpu.get_data(), C_gpu.cols());

    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLAS SGEMM failed with status: " +
                               std::to_string(status));
    }

    // Synchronize to catch any asynchronous errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      throw std::runtime_error("CUDA sync failed: " +
                               std::string(cudaGetErrorString(err)));
    }

    std::cout << "CUDA matrix multiplication completed successfully"
              << std::endl;
    C = C_gpu.to_cpu();
  } catch (const std::exception &e) {
    cublasDestroy(handle);
    throw;
  }

  cublasDestroy(handle);
  return C;
}
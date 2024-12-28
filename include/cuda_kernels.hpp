#pragma once
#include "components.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

class CudaMatrix {
private:
    float* device_ptr;
    size_t rows_;
    size_t cols_;
    cublasHandle_t handle;

public:
    CudaMatrix(const Matrix& host_matrix);
    ~CudaMatrix();
    
    // CUDA operations
    static CudaMatrix matmul(const CudaMatrix& a, const CudaMatrix& b);
    void apply_softmax();
    void apply_relu();
    void scale(float factor);
    
    // Data transfer
    Matrix to_host() const;
    void to_device(const Matrix& host_matrix);
    
    // Getters
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
};

// CUDA kernel declarations
__global__ void softmax_kernel(float* matrix, int rows, int cols);
__global__ void relu_kernel(float* matrix, int size);
__global__ void attention_kernel(float* Q, float* K, float* V, float* output,
                               int batch_size, int seq_len, int head_dim); 
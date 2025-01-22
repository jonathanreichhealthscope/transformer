#pragma once
#include "../matrix.hpp"

namespace cuda {
    // Initialize and cleanup
    void initialize_cuda();
    void cleanup_cuda();

    // Matrix operations
    void matmul(const Matrix& A, const Matrix& B, Matrix& C);
    void add(const Matrix& A, const Matrix& B, Matrix& C);
    
    // GELU operations
    void gelu_forward(Matrix& x);
    void gelu_backward(Matrix& grad_output, const Matrix& input);

    // Beam search operations
    void topk(const std::vector<float>& input, Matrix& top_k_values, 
             std::vector<int>& top_k_indices, int k);
    void beam_search_step(const Matrix& model_output, const Matrix& beam_scores,
                         Matrix& next_scores, std::vector<int>& next_tokens,
                         int beam_width);

    // Layer normalization operations
    void layer_norm_forward(const Matrix& input, const Matrix& gamma, const Matrix& beta,
                          Matrix& output, float eps);
    void layer_norm_backward(const Matrix& grad_output, const Matrix& input,
                           const Matrix& gamma, Matrix& grad_gamma,
                           Matrix& grad_beta, float eps);
} 
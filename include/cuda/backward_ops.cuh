#pragma once
#include "../matrix.hpp"

namespace cuda {
    // Backward operation wrappers
    void layer_norm_backward(const Matrix& grad, const Matrix& input, 
                           const Matrix& gamma, Matrix& dx, float eps);
    void feed_forward_backward(const Matrix& grad, const Matrix& weights, 
                             Matrix& dx, bool is_first_layer);
    void gelu_backward(Matrix& grad_output, const Matrix& input);
    __global__ void gelu_backward_kernel(float* grad_output, const float* input, int size);
} 
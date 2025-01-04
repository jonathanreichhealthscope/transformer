#include "../include/feed_forward.hpp"
#ifdef USE_CUDA
#include "../include/cuda/cuda_check.cuh"
#include "../include/cuda/cuda_launch.cuh"
#include "../include/cuda/feed_forward_kernels.cuh"
#endif
#include <cmath>
#include <iostream>
#include <random>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FeedForward::FeedForward(size_t hidden_size, size_t intermediate_size, float dropout)
    : w1(hidden_size, intermediate_size),
      w2(intermediate_size, hidden_size),
      b1(intermediate_size),
      b2(hidden_size),
      dropout_prob(dropout),
      intermediate_cache(1, intermediate_size),
      // Initialize gradients with same dimensions as their parameters
      w1_grad(hidden_size, intermediate_size),
      w2_grad(intermediate_size, hidden_size),
      b1_grad(intermediate_size),
      b2_grad(hidden_size) {

  std::cout << "FeedForward dimensions:" << std::endl;
  std::cout << "w1: " << w1.rows() << "x" << w1.cols() << std::endl;
  std::cout << "w2: " << w2.rows() << "x" << w2.cols() << std::endl;

  // Initialize weights with Xavier/Glorot initialization
  std::random_device rd;
  std::mt19937 gen(rd());

  float w1_limit = std::sqrt(6.0f / (hidden_size + intermediate_size));
  float w2_limit = std::sqrt(6.0f / (intermediate_size + hidden_size));

  std::uniform_real_distribution<float> w1_dis(-w1_limit, w1_limit);
  std::uniform_real_distribution<float> w2_dis(-w2_limit, w2_limit);

  // Initialize weights
  for (size_t i = 0; i < w1.rows(); ++i) {
    for (size_t j = 0; j < w1.cols(); ++j) {
      w1(i, j) = w1_dis(gen);
    }
  }

  for (size_t i = 0; i < w2.rows(); ++i) {
    for (size_t j = 0; j < w2.cols(); ++j) {
      w2(i, j) = w2_dis(gen);
    }
  }

  // Initialize biases to zero
  for (size_t i = 0; i < b1.size(); ++i)
    b1[i] = 0.0f;
  for (size_t i = 0; i < b2.size(); ++i)
    b2[i] = 0.0f;

  // Initialize gradients to zero
  for (size_t i = 0; i < w1_grad.rows(); ++i) {
    for (size_t j = 0; j < w1_grad.cols(); ++j) {
      w1_grad(i, j) = 0.0f;
    }
  }
  
  for (size_t i = 0; i < w2_grad.rows(); ++i) {
    for (size_t j = 0; j < w2_grad.cols(); ++j) {
      w2_grad(i, j) = 0.0f;
    }
  }

  for (size_t i = 0; i < b1_grad.size(); ++i)
    b1_grad[i] = 0.0f;
  for (size_t i = 0; i < b2_grad.size(); ++i)
    b2_grad[i] = 0.0f;
}

Matrix FeedForward::forward(const Matrix &x) {
    std::cout << "FeedForward::forward dimensions:" << std::endl;
    std::cout << "x: " << x.rows() << "x" << x.cols() << std::endl;
    std::cout << "w1: " << w1.rows() << "x" << w1.cols() << std::endl;
    std::cout << "b1: " << b1.size() << std::endl;
    
    Matrix intermediate = matmul(x, w1);
    std::cout << "intermediate shape: " << intermediate.shape() << std::endl;
    intermediate.add_bias(b1);
    intermediate.apply_gelu();
    std::cout << "intermediate after gelu: " << intermediate.shape() << std::endl;
    
    // Store intermediate values for backward pass
    intermediate_cache = intermediate;  // Direct assignment instead of deep copy
    std::cout << "intermediate_cache shape: " << intermediate_cache.shape() << std::endl;
    
    Matrix output = matmul(intermediate, w2);
    std::cout << "output shape: " << output.shape() << std::endl;
    output.add_bias(b2);
    std::cout << "output after adding bias: " << output.shape() << std::endl;
    return output;
}

void FeedForward::save(std::ostream &os) const {
  size_t hidden_size = w2.cols();
  size_t intermediate_size = w1.cols();

  os.write(reinterpret_cast<const char *>(&hidden_size), sizeof(hidden_size));
  os.write(reinterpret_cast<const char *>(&intermediate_size),
           sizeof(intermediate_size));
  os.write(reinterpret_cast<const char *>(&dropout_prob), sizeof(dropout_prob));

  os.write(reinterpret_cast<const char *>(w1.data()),
           w1.rows() * w1.cols() * sizeof(float));
  os.write(reinterpret_cast<const char *>(w2.data()),
           w2.rows() * w2.cols() * sizeof(float));
  os.write(reinterpret_cast<const char *>(b1.data()),
           b1.size() * sizeof(float));
  os.write(reinterpret_cast<const char *>(b2.data()),
           b2.size() * sizeof(float));
}

std::unique_ptr<FeedForward> FeedForward::load(std::istream &is) {
  size_t hidden_size, intermediate_size;
  float dropout_prob;

  is.read(reinterpret_cast<char *>(&hidden_size), sizeof(hidden_size));
  is.read(reinterpret_cast<char *>(&intermediate_size),
          sizeof(intermediate_size));
  is.read(reinterpret_cast<char *>(&dropout_prob), sizeof(dropout_prob));

  auto ffn = std::make_unique<FeedForward>(hidden_size, intermediate_size,
                                           dropout_prob);

  is.read(reinterpret_cast<char *>(ffn->w1.data()),
          ffn->w1.rows() * ffn->w1.cols() * sizeof(float));
  is.read(reinterpret_cast<char *>(ffn->w2.data()),
          ffn->w2.rows() * ffn->w2.cols() * sizeof(float));
  is.read(reinterpret_cast<char *>(ffn->b1.data()),
          ffn->b1.size() * sizeof(float));
  is.read(reinterpret_cast<char *>(ffn->b2.data()),
          ffn->b2.size() * sizeof(float));

  return ffn;
}

Matrix FeedForward::backward(const Matrix &grad_output, const Matrix &input) {
    if (intermediate_cache.empty()) {
        throw std::runtime_error("No cached intermediate values found for backward pass");
    }
    
    std::cout << "FeedForward::backward dimensions:" << std::endl;
    std::cout << "grad_output: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
    std::cout << "input: " << input.rows() << "x" << input.cols() << std::endl;
    std::cout << "w2: " << w2.rows() << "x" << w2.cols() << std::endl;
    std::cout << "w1: " << w1.rows() << "x" << w1.cols() << std::endl;
    std::cout << "intermediate_cache: " << intermediate_cache.rows() << "x" << intermediate_cache.cols() << std::endl;
    
    // Validate dimensions
    if (grad_output.cols() != w2.cols()) {
        throw std::runtime_error("Dimension mismatch: grad_output.cols (" + 
                                std::to_string(grad_output.cols()) + 
                                ") != w2.cols (" + std::to_string(w2.cols()) + ")");
    }
    if (input.cols() != w1.rows()) {
        throw std::runtime_error("Dimension mismatch: input.cols (" + 
                                std::to_string(input.cols()) + 
                                ") != w1.rows (" + std::to_string(w1.rows()) + ")");
    }
    
    // Create local copy of cache to prevent it being moved/destroyed
    Matrix cache_copy = intermediate_cache;
    
    std::cout << "Computing d_intermediate..." << std::endl;
    Matrix d_intermediate = matmul(grad_output, w2.transpose());
    std::cout << "d_intermediate dims: " << d_intermediate.rows() << "x" << d_intermediate.cols() << std::endl;
    
    // Ensure d_intermediate matches cache dimensions before GELU derivative
    if (d_intermediate.rows() != cache_copy.rows() || d_intermediate.cols() != cache_copy.cols()) {
        std::cout << "Reshaping d_intermediate to match cache dimensions..." << std::endl;
        Matrix reshaped_d_intermediate(cache_copy.rows(), cache_copy.cols());
        for (size_t i = 0; i < cache_copy.rows(); ++i) {
            for (size_t j = 0; j < cache_copy.cols(); ++j) {
                reshaped_d_intermediate(i, j) = d_intermediate(i % d_intermediate.rows(), 
                                                             j % d_intermediate.cols());
            }
        }
        d_intermediate = std::move(reshaped_d_intermediate);
    }
    
    std::cout << "Applying GELU derivative..." << std::endl;
    d_intermediate.apply_gelu_derivative(cache_copy);
    
    // Compute gradients for w2 and b2
    w2_grad = matmul(intermediate_cache.transpose(), grad_output);
    b2_grad = grad_output.row_sum();
    
    // Compute gradients for w1 and b1
    w1_grad = matmul(input.transpose(), d_intermediate);
    b1_grad = d_intermediate.row_sum();
    
    std::cout << "Computing grad_input..." << std::endl;
    Matrix grad_input = matmul(d_intermediate, w1.transpose());
    std::cout << "grad_input dims: " << grad_input.rows() << "x" << grad_input.cols() << std::endl;
    
    return grad_input;
}

Matrix FeedForward::backward_cuda(const Matrix &grad,
                                  const Matrix &input) const {
#ifdef USE_CUDA
  return backward_cuda(grad, input);
#else
  throw std::runtime_error("CUDA support not enabled");
#endif
}
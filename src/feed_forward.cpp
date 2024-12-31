#include "../include/feed_forward.hpp"
#ifdef USE_CUDA
#include "../include/cuda/feed_forward_kernels.cuh"
#include "../include/cuda/cuda_check.cuh"
#include "../include/cuda/cuda_launch.cuh"
#endif
#include <cmath>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

FeedForward::FeedForward(size_t hidden_size, size_t intermediate_size,
                         float dropout)
    : w1(hidden_size, intermediate_size), w2(intermediate_size, hidden_size),
      b1(intermediate_size), b2(hidden_size), dropout_prob(dropout) {

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
}

Matrix FeedForward::forward(const Matrix &x) {
  // First linear layer + GELU activation
  Matrix intermediate(x.rows(), w1.cols());
  for (size_t i = 0; i < x.rows(); ++i) {
    for (size_t j = 0; j < w1.cols(); ++j) {
      float sum = b1[j];
      for (size_t k = 0; k < x.cols(); ++k) {
        sum += x(i, k) * w1(k, j);
      }
      // GELU activation
      intermediate(i, j) =
          0.5f * sum *
          (1.0f + std::tanh(std::sqrt(2.0f / M_PI) *
                            (sum + 0.044715f * std::pow(sum, 3))));
    }
  }

  // Second linear layer
  Matrix output(x.rows(), w2.cols());
  for (size_t i = 0; i < x.rows(); ++i) {
    for (size_t j = 0; j < w2.cols(); ++j) {
      float sum = b2[j];
      for (size_t k = 0; k < intermediate.cols(); ++k) {
        sum += intermediate(i, k) * w2(k, j);
      }
      output(i, j) = sum;
    }
  }

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

Matrix FeedForward::backward(const Matrix &grad, const Matrix &input) const {
  const size_t batch_size = input.rows();
  Matrix dx(batch_size, w1.rows());

  // Backward through second linear layer
  Matrix d_intermediate(batch_size, w1.cols());
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < w1.cols(); ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < w2.cols(); ++k) {
        sum += grad(i, k) * w2(j, k);
      }
      d_intermediate(i, j) = sum;
    }
  }

  // Backward through GELU
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < w1.cols(); ++j) {
      float x = input(i, j);
      float cdf = 0.5f * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) *
                                           (x + 0.044715f * std::pow(x, 3))));
      float pdf = std::exp(-0.5f * x * x) / std::sqrt(2.0f * M_PI);
      d_intermediate(i, j) *= cdf + x * pdf;
    }
  }

  // Backward through first linear layer
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < w1.rows(); ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < w1.cols(); ++k) {
        sum += d_intermediate(i, k) * w1(j, k);
      }
      dx(i, j) = sum;
    }
  }

  return dx;
}

Matrix FeedForward::backward_cuda(const Matrix& grad, const Matrix& input) const {
#ifdef USE_CUDA
    return backward_cuda(grad, input);
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
}
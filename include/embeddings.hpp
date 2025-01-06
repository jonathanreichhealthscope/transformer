#pragma once
#include "matrix.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

// Forward declarations
class TokenEmbedding;
class PositionalEncoding;

class TokenEmbedding {
public:
  TokenEmbedding(size_t vocab_size, size_t embedding_dim);

  // Core functionality
  Matrix forward(const std::vector<std::vector<int>>& batch_tokens);
  Matrix project_to_vocab(const Matrix &hidden_states);
  void backward(const Matrix &grad_output,
                const std::vector<int> &input_tokens);
  virtual void forward_cuda(const std::vector<int> &tokens, Matrix &output);
  virtual Matrix project_to_vocab_cuda(const Matrix &input);

  // Accessors
  const Matrix &get_embedding_table() const { return weights_; }
  Matrix &get_embedding_table() { return weights_; }
  const Matrix &get_gradient_table() const { return weights_grad_; }
  Matrix &get_gradient_table() { return weights_grad_; }
  size_t get_vocab_size() const { return vocab_size_; }
  size_t get_embedding_dim() const { return embedding_dim_; }

  // Parameter and gradient access
  struct Parameters {
    std::vector<std::reference_wrapper<Matrix>> matrices;

    // Add iterator support
    auto begin() { return matrices.begin(); }
    auto end() { return matrices.end(); }
    auto begin() const { return matrices.begin(); }
    auto end() const { return matrices.end(); }
  };

  Parameters &parameters() {
    params_.matrices.clear();
    params_.matrices.emplace_back(weights_);
    return params_;
  }

  const Parameters &parameter_gradients() const {
    param_gradients_.matrices.clear();
    param_gradients_.matrices.emplace_back(
        std::ref(const_cast<Matrix &>(weights_grad_)));
    return param_gradients_;
  }

  // Serialization
  void save(std::ostream &os) const;
  static std::unique_ptr<TokenEmbedding> load(std::istream &is);

private:
  Parameters params_;
  mutable Parameters param_gradients_;
  Matrix weights_;
  Matrix weights_grad_;
  size_t vocab_size_;
  size_t embedding_dim_;
};

class PositionalEncoding {
private:
  Matrix encoding_matrix_;
  size_t max_seq_length_;
  size_t hidden_size_;

public:
  PositionalEncoding() = default;
  PositionalEncoding(size_t max_seq_length, size_t hidden_size);
  virtual ~PositionalEncoding() = default;

  // Core functionality
  Matrix forward(const Matrix &position_ids);

  // Serialization
  void save(std::ostream &os) const;
  static std::unique_ptr<PositionalEncoding> load(std::istream &is);

  // Accessors
  const Matrix &get_encoding_matrix() const { return encoding_matrix_; }
  Matrix &get_encoding_matrix() { return encoding_matrix_; }
  size_t get_max_seq_length() const { return max_seq_length_; }
  size_t get_hidden_size() const { return hidden_size_; }

  std::vector<std::reference_wrapper<Matrix>> &parameters() {
    static std::vector<std::reference_wrapper<Matrix>> params;
    params.clear();
    params.emplace_back(encoding_matrix_);
    return params;
  }
};
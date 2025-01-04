#pragma once
#include "matrix.hpp"
#include <iostream>
#include <memory>
#include <vector>
#include <functional>

// Forward declarations
class TokenEmbedding;
class PositionalEncoding;

class TokenEmbedding {
private:
  Matrix weights_;
  size_t vocab_size_;
  size_t embedding_dim_;

public:
  TokenEmbedding(size_t vocab_size, size_t embedding_dim);

  // Core functionality
  Matrix forward(const std::vector<int> &tokens);
  Matrix project_to_vocab(const Matrix &hidden_states);
  void backward(const Matrix &grad_output,
                const std::vector<int> &input_tokens);
  virtual void forward_cuda(const std::vector<int> &tokens, Matrix &output);
  virtual Matrix project_to_vocab_cuda(const Matrix &input);

  // Accessors
  const Matrix &get_embedding_table() const { return weights_; }
  Matrix &get_embedding_table() { return weights_; }
  size_t get_vocab_size() const { return vocab_size_; }
  size_t get_embedding_dim() const { return embedding_dim_; }

  // Serialization
  void save(std::ostream &os) const;
  static std::unique_ptr<TokenEmbedding> load(std::istream &is);

  std::vector<std::reference_wrapper<Matrix>>& parameters() {
    static std::vector<std::reference_wrapper<Matrix>> params;
    params.clear();
    params.emplace_back(weights_);
    return params;
  }
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

  std::vector<std::reference_wrapper<Matrix>>& parameters() {
    static std::vector<std::reference_wrapper<Matrix>> params;
    params.clear();
    params.emplace_back(encoding_matrix_);
    return params;
  }
};
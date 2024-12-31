#pragma once
#include "components.hpp"

class TokenEmbedding {
private:
  Matrix weights;
  size_t vocab_size_;
  size_t hidden_size_;

public:
  virtual ~TokenEmbedding() = default;
  TokenEmbedding() = default;
  TokenEmbedding(size_t vocab_size, size_t hidden_size);
  Matrix forward(const std::vector<int> &tokens);
  Matrix project_to_vocab(const Matrix &hidden_states);
  void save(std::ostream &os) const;
  static std::unique_ptr<TokenEmbedding> load(std::istream &is);
  friend class Transformer;
  void forward_cuda(const std::vector<int>& tokens, Matrix& output);
  Matrix project_to_vocab_cuda(const Matrix& hidden_states);

  TokenEmbedding(const TokenEmbedding& other)
      : weights(other.weights),
        vocab_size_(other.vocab_size_),
        hidden_size_(other.hidden_size_) {}
  
  TokenEmbedding& operator=(const TokenEmbedding& other) {
      if (this != &other) {
          weights = other.weights;
          vocab_size_ = other.vocab_size_;
          hidden_size_ = other.hidden_size_;
      }
      return *this;
  }
};

class PositionalEncoding {
private:
  Matrix encoding_matrix;

public:
  virtual ~PositionalEncoding() = default;
  PositionalEncoding() = default;
  PositionalEncoding(size_t max_seq_length, size_t hidden_size);
  Matrix forward(const Matrix &position_ids);
  void save(std::ostream &os) const;
  static std::unique_ptr<PositionalEncoding> load(std::istream &is);

  PositionalEncoding(const PositionalEncoding& other)
      : encoding_matrix(other.encoding_matrix) {}
  
  PositionalEncoding& operator=(const PositionalEncoding& other) {
      if (this != &other) {
          encoding_matrix = other.encoding_matrix;
      }
      return *this;
  }
};
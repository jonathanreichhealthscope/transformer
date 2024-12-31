#pragma once
#include "matrix.hpp"
#include <vector>
#include <memory>
#include <iostream>

// Forward declarations
class TokenEmbedding;
class PositionalEncoding;

class TokenEmbedding {
private:
    Matrix embedding_table_;
    size_t vocab_size_;
    size_t embedding_dim_;

public:
    TokenEmbedding(size_t vocab_size, size_t embedding_dim)
        : embedding_table_(vocab_size, embedding_dim),
          vocab_size_(vocab_size),
          embedding_dim_(embedding_dim) {}

    // Core functionality
    void forward_cuda(const std::vector<int>& tokens, Matrix& output);
    Matrix project_to_vocab_cuda(const Matrix& input);

    // Accessors
    const Matrix& get_embedding_table() const { return embedding_table_; }
    Matrix& get_embedding_table() { return embedding_table_; }
    size_t get_vocab_size() const { return vocab_size_; }
    size_t get_embedding_dim() const { return embedding_dim_; }

    // Serialization
    void save(std::ostream& os) const;
    static std::unique_ptr<TokenEmbedding> load(std::istream& is);
};

class PositionalEncoding {
private:
    Matrix encoding_matrix_;
    size_t max_seq_length_;
    size_t hidden_size_;

public:
    PositionalEncoding() = default;
    
    PositionalEncoding(size_t max_seq_length, size_t hidden_size)
        : encoding_matrix_(max_seq_length, hidden_size),
          max_seq_length_(max_seq_length),
          hidden_size_(hidden_size) {}

    virtual ~PositionalEncoding() = default;

    // Core functionality
    Matrix forward(const Matrix& position_ids);

    // Serialization
    void save(std::ostream& os) const;
    static std::unique_ptr<PositionalEncoding> load(std::istream& is);

    // Copy operations
    PositionalEncoding(const PositionalEncoding& other)
        : encoding_matrix_(other.encoding_matrix_),
          max_seq_length_(other.max_seq_length_),
          hidden_size_(other.hidden_size_) {}
    
    PositionalEncoding& operator=(const PositionalEncoding& other) {
        if (this != &other) {
            encoding_matrix_ = other.encoding_matrix_;
            max_seq_length_ = other.max_seq_length_;
            hidden_size_ = other.hidden_size_;
        }
        return *this;
    }

    // Accessors
    const Matrix& get_encoding_matrix() const { return encoding_matrix_; }
    Matrix& get_encoding_matrix() { return encoding_matrix_; }
    size_t get_max_seq_length() const { return max_seq_length_; }
    size_t get_hidden_size() const { return hidden_size_; }
};
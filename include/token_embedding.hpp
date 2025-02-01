#pragma once
#include "matrix.hpp"

class TokenEmbedding {
public:
    // Parameter structure to hold embedding weights
    struct Parameters {
        Matrix weights;  // Embedding weights
    };

    // Gradient structure to hold gradients
    struct Gradients {
        Matrix weights_grad;  // Gradient for weights
    };

private:
    Parameters params_;
    Gradients grads_;
    size_t vocab_size_;
    size_t embedding_dim_;

public:
    TokenEmbedding(size_t vocab_size, size_t embedding_dim);
    virtual ~TokenEmbedding() = default;

    // Forward and backward methods
    Matrix forward(const std::vector<int>& tokens);
    Matrix backward(const Matrix& grad_output, const std::vector<int>& tokens);

    // Update accessor methods to use Parameters structure
    Matrix& get_weights() { return params_.weights; }
    const Matrix& get_weights() const { return params_.weights; }

    // Add parameter accessors
    Parameters& parameters() { return params_; }
    Gradients& param_gradients() { return grads_; }
    const Parameters& parameters() const { return params_; }
    const Gradients& param_gradients() const { return grads_; }

    // Get list of weights for parameter updates
    std::vector<std::reference_wrapper<Matrix>> get_weight_list() {
        return {std::ref(params_.weights)};
    }

    // Update copy constructor to use Parameters/Gradients
    TokenEmbedding(const TokenEmbedding& other)
        : params_(other.params_), grads_(other.grads_),
          vocab_size_(other.vocab_size_), embedding_dim_(other.embedding_dim_) {}

    // Update assignment operator to use Parameters/Gradients
    TokenEmbedding& operator=(const TokenEmbedding& other) {
        if (this != &other) {
            params_ = other.params_;
            grads_ = other.grads_;
            vocab_size_ = other.vocab_size_;
            embedding_dim_ = other.embedding_dim_;
        }
        return *this;
    }

    size_t get_vocab_size() const { return vocab_size_; }
    size_t get_embedding_dim() const { return embedding_dim_; }

    void save(std::ostream& os) const;
    static std::unique_ptr<TokenEmbedding> load(std::istream& is);
}; 
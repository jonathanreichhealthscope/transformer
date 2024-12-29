#pragma once
#include "components.hpp"

class FeedForward {
private:
    Matrix w1, w2;
    Vector b1, b2;
    float dropout_prob;

public:
    FeedForward(size_t hidden_size, size_t intermediate_size, float dropout_prob);
    Matrix forward(const Matrix& x);
    void save(std::ostream& os) const;
    static std::unique_ptr<FeedForward> load(std::istream& is);
    Matrix backward(const Matrix& grad, const Matrix& input) const;
    Matrix backward_cuda(const Matrix& grad, const Matrix& input) const;
    friend class Transformer;
}; 
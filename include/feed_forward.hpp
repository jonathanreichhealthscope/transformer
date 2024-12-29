#pragma once
#include "components.hpp"
#include <cereal/access.hpp>
#ifdef USE_CUDA
#include "cuda/cuda_utils.cuh"
#endif

class FeedForward {
private:
    Matrix w1, w2;
    Vector b1, b2;
    float dropout_prob;

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar) {
        ar(w1, w2, b1, b2, dropout_prob);
    }

public:
    virtual ~FeedForward() = default;
    FeedForward() = default;
    FeedForward(size_t hidden_size, size_t intermediate_size, float dropout_prob);
    Matrix forward(const Matrix& x);
    Matrix backward(const Matrix& grad, const Matrix& input) const;
    Matrix backward_cuda(const Matrix& grad, const Matrix& input) const;
    void save(std::ostream& os) const;
    static std::unique_ptr<FeedForward> load(std::istream& is);
    friend class Transformer;
}; 
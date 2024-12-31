#pragma once
#include "matrix.hpp"
#include <vector>
#include <memory>
#include <iostream>

// Forward declaration
class LayerNorm;

class LayerNorm {
private:
    Vector gamma_;
    Vector beta_;
    float eps_;

public:
    LayerNorm(size_t hidden_size, float eps = 1e-5)
        : gamma_(hidden_size, 1.0f), beta_(hidden_size, 0.0f), eps_(eps) {}

    // CUDA operations
    Matrix forward_cuda(const Matrix& input) const;
    Matrix backward_cuda(const Matrix& grad_output, const Matrix& input) const;

    // Serialization
    void save(std::ostream& os) const;
    static std::unique_ptr<LayerNorm> load(std::istream& is);

    // Accessors
    const Vector& get_gamma() const { return gamma_; }
    const Vector& get_beta() const { return beta_; }
    Vector& get_gamma() { return gamma_; }
    Vector& get_beta() { return beta_; }
    float get_eps() const { return eps_; }
};
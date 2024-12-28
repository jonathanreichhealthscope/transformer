#pragma once
#include "components.hpp"

class LayerNorm {
private:
    Vector gamma;
    Vector beta;
    float eps;

public:
    LayerNorm(size_t hidden_size, float eps = 1e-5);
    Matrix forward(const Matrix& x);
    Matrix forward_cuda(const Matrix& x);
    void save(std::ostream& os) const;
    static std::unique_ptr<LayerNorm> load(std::istream& is);
}; 
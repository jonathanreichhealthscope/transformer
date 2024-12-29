#pragma once
#include "components.hpp"
#include <vector>
#ifdef USE_CUDA
#include "cuda/cuda_utils.cuh"
#endif

class Quantizer {
private:
    size_t bits;
    float scale;
    float zero_point;

public:
    explicit Quantizer(size_t num_bits = 8);
    Matrix quantize(const Matrix& input);
    Matrix quantize_cuda(const Matrix& input);
    Matrix dequantize(const Matrix& quantized);
    Matrix dequantize_cuda(const Matrix& quantized);
    void save(std::ostream& os) const;
    static std::unique_ptr<Quantizer> load(std::istream& is);
}; 
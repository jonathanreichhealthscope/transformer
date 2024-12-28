#pragma once
#include "components.hpp"
#include <cstdint>

enum class QuantizationType {
    INT8,
    INT4,
    FLOAT16
};

class QuantizedMatrix {
private:
    std::vector<int8_t> quantized_data;
    std::vector<float> scales;
    std::vector<float> zero_points;
    size_t rows_;
    size_t cols_;
    QuantizationType type;

public:
    QuantizedMatrix(const Matrix& matrix, QuantizationType type);
    Matrix dequantize() const;
    
    // Quantized operations
    static Matrix quantized_matmul(const QuantizedMatrix& a, const QuantizedMatrix& b);
    void apply_quantized_relu();
    
    // Serialization
    void save(std::ostream& os) const;
    static QuantizedMatrix load(std::istream& is);
}; 
#pragma once
#include "components.hpp"
#include <cstdint>
#include <cuda_fp16.h>
#include <cereal/types/vector.hpp>
#include <cereal/access.hpp>

enum class QuantizationType {
    INT8,
    INT4,
    FLOAT16
};

class QuantizedMatrix {
private:
    size_t rows_;
    size_t cols_;
    union {
        std::vector<int8_t> int8_data;
        std::vector<uint8_t> int4_data;
        std::vector<__half> fp16_data;
    };
    std::vector<float> scales;
    std::vector<float> zero_points;
    QuantizationType type;

    // Add cereal serialization
    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar) {
        ar(rows_, cols_, type);
        ar(scales, zero_points);
        
        switch (type) {
            case QuantizationType::INT8:
                ar(int8_data);
                break;
            case QuantizationType::INT4:
                ar(int4_data);
                break;
            case QuantizationType::FLOAT16:
                ar(fp16_data);
                break;
        }
    }

public:
    // Default constructor
    QuantizedMatrix() : rows_(0), cols_(0), type(QuantizationType::FLOAT16) {
        new (&fp16_data) std::vector<__half>();
    }
    
    // Constructor for direct half-precision data
    QuantizedMatrix(size_t rows, size_t cols, std::vector<__half>&& data)
        : rows_(rows), cols_(cols), type(QuantizationType::FLOAT16) {
        new (&fp16_data) std::vector<__half>(std::move(data));
    }
    
    // Constructor for quantization from full precision
    QuantizedMatrix(const Matrix& matrix, QuantizationType type);
    
    // Destructor
    ~QuantizedMatrix() {
        switch (type) {
            case QuantizationType::INT8:
                int8_data.~vector();
                break;
            case QuantizationType::INT4:
                int4_data.~vector();
                break;
            case QuantizationType::FLOAT16:
                fp16_data.~vector();
                break;
        }
    }
    
    // Copy constructor
    QuantizedMatrix(const QuantizedMatrix& other);
    
    // Move constructor
    QuantizedMatrix(QuantizedMatrix&& other) noexcept;
    
    // Copy assignment
    QuantizedMatrix& operator=(const QuantizedMatrix& other);
    
    // Move assignment
    QuantizedMatrix& operator=(QuantizedMatrix&& other) noexcept;
    
    // Core functionality
    Matrix dequantize() const;
    static QuantizedMatrix quantize(const Matrix& input);
    
    // Quantized operations
    static Matrix quantized_matmul(const QuantizedMatrix& a, const QuantizedMatrix& b);
    void apply_quantized_relu();
    
    // Serialization
    void save(std::ostream& os) const;
    static QuantizedMatrix load(std::istream& is);
    
    // Accessors
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    const void* data() const {
        switch (type) {
            case QuantizationType::INT8:
                return int8_data.data();
            case QuantizationType::INT4:
                return int4_data.data();
            case QuantizationType::FLOAT16:
                return fp16_data.data();
        }
        return nullptr;
    }
}; 
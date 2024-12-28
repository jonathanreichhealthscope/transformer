#include "../include/quantization.hpp"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace {
    // Helper functions for quantization
    std::pair<float, float> compute_scale_zero_point(float min_val, float max_val, int bits) {
        const float qmin = 0;
        const float qmax = (1 << bits) - 1;
        const float scale = (max_val - min_val) / (qmax - qmin);
        const float zero_point = qmin - min_val / scale;
        return {scale, zero_point};
    }
    
    template<typename T>
    T quantize_value(float val, float scale, float zero_point) {
        float transformed = val / scale + zero_point;
        return static_cast<T>(std::max(0.0f, std::min(255.0f, std::round(transformed))));
    }
    
    float dequantize_value(int8_t val, float scale, float zero_point) {
        return scale * (static_cast<float>(val) - zero_point);
    }
}

QuantizedMatrix::QuantizedMatrix(const Matrix& matrix, QuantizationType type) 
    : rows_(matrix.rows()), cols_(matrix.cols()), type(type) {
    
    const size_t num_elements = rows_ * cols_;
    
    switch (type) {
        case QuantizationType::INT8: {
            // Compute per-column statistics and quantization parameters
            scales.resize(cols_);
            zero_points.resize(cols_);
            quantized_data.resize(num_elements);
            
            for (size_t col = 0; col < cols_; ++col) {
                float min_val = std::numeric_limits<float>::max();
                float max_val = std::numeric_limits<float>::lowest();
                
                // Find min/max values in this column
                for (size_t row = 0; row < rows_; ++row) {
                    float val = matrix(row, col);
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
                
                // Compute scale and zero point
                auto [scale, zero_point] = compute_scale_zero_point(min_val, max_val, 8);
                scales[col] = scale;
                zero_points[col] = zero_point;
                
                // Quantize the column
                for (size_t row = 0; row < rows_; ++row) {
                    size_t idx = row * cols_ + col;
                    quantized_data[idx] = quantize_value<int8_t>(
                        matrix(row, col), scale, zero_point
                    );
                }
            }
            break;
        }
        
        case QuantizationType::INT4: {
            // Similar to INT8 but pack two 4-bit values into one byte
            scales.resize(cols_);
            zero_points.resize(cols_);
            quantized_data.resize((num_elements + 1) / 2);
            
            for (size_t col = 0; col < cols_; ++col) {
                float min_val = std::numeric_limits<float>::max();
                float max_val = std::numeric_limits<float>::lowest();
                
                // Find min/max values
                for (size_t row = 0; row < rows_; ++row) {
                    float val = matrix(row, col);
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
                
                auto [scale, zero_point] = compute_scale_zero_point(min_val, max_val, 4);
                scales[col] = scale;
                zero_points[col] = zero_point;
                
                // Quantize and pack two 4-bit values into one byte
                for (size_t row = 0; row < rows_; row += 2) {
                    size_t idx = (row * cols_ + col) / 2;
                    uint8_t high = quantize_value<uint8_t>(
                        matrix(row, col), scale, zero_point
                    ) & 0x0F;
                    
                    uint8_t low = row + 1 < rows_ ? 
                        quantize_value<uint8_t>(
                            matrix(row + 1, col), scale, zero_point
                        ) & 0x0F : 0;
                    
                    quantized_data[idx] = (high << 4) | low;
                }
            }
            break;
        }
        
        case QuantizationType::FLOAT16: {
            // Convert to float16 using CUDA
            quantized_data.resize(num_elements * sizeof(__half));
            float* d_input;
            __half* d_output;
            
            cudaMalloc(&d_input, num_elements * sizeof(float));
            cudaMalloc(&d_output, num_elements * sizeof(__half));
            
            cudaMemcpy(d_input, matrix.data(), num_elements * sizeof(float), 
                      cudaMemcpyHostToDevice);
            
            // Launch kernel to convert float32 to float16
            const int block_size = 256;
            const int grid_size = (num_elements + block_size - 1) / block_size;
            
            convert_f32_to_f16<<<grid_size, block_size>>>(
                d_input, d_output, num_elements
            );
            
            cudaMemcpy(quantized_data.data(), d_output, 
                      num_elements * sizeof(__half), cudaMemcpyDeviceToHost);
            
            cudaFree(d_input);
            cudaFree(d_output);
            break;
        }
    }
}

Matrix QuantizedMatrix::dequantize() const {
    Matrix result(rows_, cols_);
    
    switch (type) {
        case QuantizationType::INT8: {
            for (size_t col = 0; col < cols_; ++col) {
                float scale = scales[col];
                float zero_point = zero_points[col];
                
                for (size_t row = 0; row < rows_; ++row) {
                    size_t idx = row * cols_ + col;
                    result(row, col) = dequantize_value(
                        quantized_data[idx], scale, zero_point
                    );
                }
            }
            break;
        }
        
        case QuantizationType::INT4: {
            for (size_t col = 0; col < cols_; ++col) {
                float scale = scales[col];
                float zero_point = zero_points[col];
                
                for (size_t row = 0; row < rows_; row += 2) {
                    size_t idx = (row * cols_ + col) / 2;
                    uint8_t packed = quantized_data[idx];
                    
                    result(row, col) = dequantize_value(
                        (packed >> 4) & 0x0F, scale, zero_point
                    );
                    
                    if (row + 1 < rows_) {
                        result(row + 1, col) = dequantize_value(
                            packed & 0x0F, scale, zero_point
                        );
                    }
                }
            }
            break;
        }
        
        case QuantizationType::FLOAT16: {
            // Convert back to float32 using CUDA
            const size_t num_elements = rows_ * cols_;
            __half* d_input;
            float* d_output;
            
            cudaMalloc(&d_input, num_elements * sizeof(__half));
            cudaMalloc(&d_output, num_elements * sizeof(float));
            
            cudaMemcpy(d_input, quantized_data.data(), 
                      num_elements * sizeof(__half), cudaMemcpyHostToDevice);
            
            const int block_size = 256;
            const int grid_size = (num_elements + block_size - 1) / block_size;
            
            convert_f16_to_f32<<<grid_size, block_size>>>(
                d_input, d_output, num_elements
            );
            
            cudaMemcpy(result.data(), d_output, 
                      num_elements * sizeof(float), cudaMemcpyDeviceToHost);
            
            cudaFree(d_input);
            cudaFree(d_output);
            break;
        }
    }
    
    return result;
}

void QuantizedMatrix::save(std::ostream& os) const {
    // Save metadata
    os.write(reinterpret_cast<const char*>(&rows_), sizeof(rows_));
    os.write(reinterpret_cast<const char*>(&cols_), sizeof(cols_));
    os.write(reinterpret_cast<const char*>(&type), sizeof(type));
    
    // Save quantization parameters
    size_t scales_size = scales.size();
    os.write(reinterpret_cast<const char*>(&scales_size), sizeof(scales_size));
    os.write(reinterpret_cast<const char*>(scales.data()), scales_size * sizeof(float));
    os.write(reinterpret_cast<const char*>(zero_points.data()), scales_size * sizeof(float));
    
    // Save quantized data
    size_t data_size = quantized_data.size();
    os.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
    os.write(reinterpret_cast<const char*>(quantized_data.data()), data_size);
}

QuantizedMatrix QuantizedMatrix::load(std::istream& is) {
    QuantizedMatrix result;
    
    // Load metadata
    is.read(reinterpret_cast<char*>(&result.rows_), sizeof(result.rows_));
    is.read(reinterpret_cast<char*>(&result.cols_), sizeof(result.cols_));
    is.read(reinterpret_cast<char*>(&result.type), sizeof(result.type));
    
    // Load quantization parameters
    size_t scales_size;
    is.read(reinterpret_cast<char*>(&scales_size), sizeof(scales_size));
    result.scales.resize(scales_size);
    result.zero_points.resize(scales_size);
    is.read(reinterpret_cast<char*>(result.scales.data()), scales_size * sizeof(float));
    is.read(reinterpret_cast<char*>(result.zero_points.data()), scales_size * sizeof(float));
    
    // Load quantized data
    size_t data_size;
    is.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    result.quantized_data.resize(data_size);
    is.read(reinterpret_cast<char*>(result.quantized_data.data()), data_size);
    
    return result;
} 
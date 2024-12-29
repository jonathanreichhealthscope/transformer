#include "../include/quantization.hpp"
#include <cmath>
#ifdef USE_CUDA
#include "cuda/quantization_kernels.cuh"
#endif

Quantizer::Quantizer(size_t num_bits)
    : bits(num_bits), scale(1.0f), zero_point(0.0f) {}

Matrix Quantizer::quantize(const Matrix& input) {
    // Find min and max values
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            min_val = std::min(min_val, input(i, j));
            max_val = std::max(max_val, input(i, j));
        }
    }
    
    // Calculate scale and zero point
    float range = max_val - min_val;
    scale = range / ((1 << bits) - 1);
    zero_point = -min_val / scale;
    
    // Quantize values
    Matrix quantized(input.rows(), input.cols());
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            float val = input(i, j);
            quantized(i, j) = std::round(val / scale + zero_point);
        }
    }
    
    return quantized;
}

Matrix Quantizer::dequantize(const Matrix& quantized) {
    Matrix result(quantized.rows(), quantized.cols());
    
    for (size_t i = 0; i < quantized.rows(); ++i) {
        for (size_t j = 0; j < quantized.cols(); ++j) {
            float val = quantized(i, j);
            result(i, j) = (val - zero_point) * scale;
        }
    }
    
    return result;
}

void Quantizer::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&bits), sizeof(bits));
    os.write(reinterpret_cast<const char*>(&scale), sizeof(scale));
    os.write(reinterpret_cast<const char*>(&zero_point), sizeof(zero_point));
}

std::unique_ptr<Quantizer> Quantizer::load(std::istream& is) {
    size_t bits;
    float scale, zero_point;
    
    is.read(reinterpret_cast<char*>(&bits), sizeof(bits));
    is.read(reinterpret_cast<char*>(&scale), sizeof(scale));
    is.read(reinterpret_cast<char*>(&zero_point), sizeof(zero_point));
    
    auto quantizer = std::make_unique<Quantizer>(bits);
    quantizer->scale = scale;
    quantizer->zero_point = zero_point;
    
    return quantizer;
}

Matrix Quantizer::quantize_cuda(const Matrix& input) {
#ifdef USE_CUDA
    // Find min/max using CUDA reduction kernel
    float min_val, max_val;
    find_minmax_cuda(input.data(), input.rows() * input.cols(), &min_val, &max_val);
    
    // Calculate scale and zero point
    float range = max_val - min_val;
    scale = range / ((1 << bits) - 1);
    zero_point = -min_val / scale;
    
    // Setup CUDA grid dimensions
    const int block_size = 256;
    const int grid_size = (input.rows() * input.cols() + block_size - 1) / block_size;
    
    // Quantize using CUDA kernel
    Matrix quantized(input.rows(), input.cols());
    CUDA_LAUNCH(quantize_kernel, grid_size, block_size, 0, 0,
        input.data(), quantized.data(),
        input.rows() * input.cols(),
        scale, zero_point
    );
    
    return quantized;
#else
    return quantize(input);
#endif
}

Matrix Quantizer::dequantize_cuda(const Matrix& quantized) {
#ifdef USE_CUDA
    // Setup CUDA grid dimensions
    const int block_size = 256;
    const int grid_size = (quantized.rows() * quantized.cols() + block_size - 1) / block_size;
    
    Matrix result(quantized.rows(), quantized.cols());
    CUDA_LAUNCH(dequantize_kernel, grid_size, block_size, 0, 0,
        quantized.data(), result.data(),
        quantized.rows() * quantized.cols(),
        scale, zero_point
    );
    return result;
#else
    return dequantize(quantized);
#endif
} 
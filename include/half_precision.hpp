#pragma once

#include "components.hpp"
#include "config.hpp"  // Add this include for TransformerConfig
#ifdef USE_CUDA
#include <cuda_fp16.h>
using half_type = __half;  ///< CUDA 16-bit floating point type
#else
using half_type = float;   ///< Fallback to float when CUDA is not available
#endif

/**
 * @brief Provides utilities for half-precision (FP16) training.
 * 
 * This class implements functionality for converting between 32-bit and 16-bit
 * floating point representations, enabling memory-efficient training while
 * maintaining numerical stability. Features include:
 * - FP32 to FP16 conversion
 * - FP16 to FP32 conversion
 * - CUDA support when available
 * - Automatic fallback to FP32 when CUDA is not available
 * 
 * Half-precision training can reduce memory usage and potentially improve
 * performance on hardware with FP16 support (e.g., NVIDIA Tensor Cores).
 */
class HalfPrecisionTraining {
  public:
    /**
     * @brief Initializes the HalfPrecisionTraining class with the given configuration.
     * 
     * @param config The configuration object containing memory pool size information.
     */
    static void initialize(const TransformerConfig& config);

    /**
     * @brief Converts a matrix from FP32 to FP16 format.
     * 
     * When CUDA is available, this uses CUDA's native FP16 type.
     * Otherwise, the matrix remains in FP32 format.
     * 
     * @param matrix Matrix to convert to half precision
     */
    static void convert_to_fp16(Matrix& matrix);

    /**
     * @brief Converts a matrix from FP16 back to FP32 format.
     * 
     * This conversion is necessary before operations that require
     * full precision or when saving model weights.
     * 
     * @param matrix Matrix to convert to single precision
     */
    static void convert_to_fp32(Matrix& matrix);

  private:
    static std::vector<half_type> half_data;  ///< Buffer for half-precision data
};
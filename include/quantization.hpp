#pragma once
#include "components.hpp"
#include "transformer.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#ifdef USE_CUDA
#include "cuda/cuda_utils.cuh"
#endif

/**
 * @brief Post-training quantization for model compression.
 * 
 * The Quantizer class provides functionality for quantizing model weights
 * and activations to reduced precision (e.g., 8-bit integers) after training.
 * Features include:
 * - Configurable bit width
 * - Scale and zero-point calibration
 * - CUDA acceleration support
 * - Symmetric and asymmetric quantization
 */
class Quantizer {
  private:
    size_t bits;         ///< Number of bits for quantization
    float scale;         ///< Scaling factor for quantization
    float zero_point;    ///< Zero point offset for asymmetric quantization

  public:
    /**
     * @brief Constructs a quantizer with specified precision.
     * @param num_bits Number of bits for quantization (default: 8)
     */
    explicit Quantizer(size_t num_bits = 8);

    /**
     * @brief Quantizes a floating-point matrix to reduced precision.
     * @param input Input matrix to quantize
     * @return Quantized matrix
     */
    Matrix quantize(const Matrix& input);

    /**
     * @brief CUDA-accelerated matrix quantization.
     * @param input Input matrix to quantize
     * @return Quantized matrix
     */
    Matrix quantize_cuda(const Matrix& input);

    /**
     * @brief Dequantizes a matrix back to floating-point.
     * @param quantized Quantized matrix to convert back
     * @return Dequantized floating-point matrix
     */
    Matrix dequantize(const Matrix& quantized);

    /**
     * @brief CUDA-accelerated matrix dequantization.
     * @param quantized Quantized matrix to convert back
     * @return Dequantized floating-point matrix
     */
    Matrix dequantize_cuda(const Matrix& quantized);

    /**
     * @brief Saves quantization parameters to a stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const;

    /**
     * @brief Loads quantization parameters from a stream.
     * @param is Input stream to load from
     * @return Unique pointer to loaded quantizer
     */
    static std::unique_ptr<Quantizer> load(std::istream& is);
};

/**
 * @brief Implements Quantization-Aware Training (QAT) for transformer models.
 * 
 * QAT simulates quantization effects during training, allowing the model to
 * adapt to reduced precision. Features include:
 * - Per-layer quantization parameters
 * - Calibration using representative data
 * - Symmetric and asymmetric quantization options
 * - Statistics collection for optimal scaling
 */
class QuantizationAwareTraining {
  private:
    /**
     * @brief Parameters for quantizing a single layer.
     */
    struct QuantizationParams {
        float scale;      ///< Scaling factor for the layer
        float zero_point; ///< Zero point for asymmetric quantization
        int bits;        ///< Number of bits for quantization
    };

    std::unordered_map<std::string, QuantizationParams> layer_params; ///< Per-layer quantization parameters
    bool use_symmetric_quantization;  ///< Whether to use symmetric quantization
    const int default_bits = 8;      ///< Default quantization bit width

    /**
     * @brief Collects statistics for calibrating quantization parameters.
     * 
     * Analyzes weights and calibration data to determine optimal
     * scaling factors and zero points for each layer.
     * 
     * @param weights Layer weights to analyze
     * @param calibration_data Representative input data
     * @param layer_name Name of the layer being analyzed
     */
    void collect_statistics(const std::vector<std::reference_wrapper<Matrix>>& weights,
                            const std::vector<Matrix>& calibration_data,
                            const std::string& layer_name);

    /**
     * @brief Performs symmetric quantization of weights.
     * 
     * Quantizes weights using symmetric scaling around zero,
     * which is often preferred for weight quantization.
     * 
     * @param weights Weights to quantize
     * @param scale Scaling factor
     * @param bits Number of quantization bits
     * @return Quantized weights
     */
    Matrix symmetric_quantize(const Matrix& weights, float scale, int bits);

  public:
    /**
     * @brief Constructs a QAT trainer.
     * @param symmetric Whether to use symmetric quantization (default: true)
     */
    QuantizationAwareTraining(bool symmetric = true) : use_symmetric_quantization(symmetric) {}

    /**
     * @brief Calibrates quantization parameters using representative data.
     * 
     * Analyzes the model and calibration data to determine optimal
     * quantization parameters for each layer.
     * 
     * @param model Transformer model to analyze
     * @param calibration_data Representative input data for calibration
     */
    void calibrate(const Transformer& model, const std::vector<Matrix>& calibration_data);

    /**
     * @brief Quantizes weights of a specific layer.
     * 
     * Applies quantization using the calibrated parameters
     * for the specified layer.
     * 
     * @param weights Weights to quantize
     * @param layer_name Name of the layer
     * @return Quantized weights
     */
    Matrix quantize_weights(const Matrix& weights, const std::string& layer_name);
};
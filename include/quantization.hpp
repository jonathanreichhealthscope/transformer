#pragma once
#include "components.hpp"
#include "transformer.hpp"
#include <string>
#include <unordered_map>
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
  Matrix quantize(const Matrix &input);
  Matrix quantize_cuda(const Matrix &input);
  Matrix dequantize(const Matrix &quantized);
  Matrix dequantize_cuda(const Matrix &quantized);
  void save(std::ostream &os) const;
  static std::unique_ptr<Quantizer> load(std::istream &is);
};

class QuantizationAwareTraining {
private:
  struct QuantizationParams {
    float scale;
    float zero_point;
    int bits;
  };

  std::unordered_map<std::string, QuantizationParams> layer_params;
  bool use_symmetric_quantization;
  const int default_bits = 8;

  void
  collect_statistics(const std::vector<std::reference_wrapper<Matrix>> &weights,
                     const std::vector<Matrix> &calibration_data,
                     const std::string &layer_name) {
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();

    // Collect min/max from weights
    for (const auto &weight_ref : weights) {
      const Matrix &weight = weight_ref.get();
      for (size_t i = 0; i < weight.rows(); ++i) {
        for (size_t j = 0; j < weight.cols(); ++j) {
          min_val = std::min(min_val, weight(i, j));
          max_val = std::max(max_val, weight(i, j));
        }
      }
    }

    // Collect min/max from calibration data
    for (const auto &data : calibration_data) {
      for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
          min_val = std::min(min_val, data(i, j));
          max_val = std::max(max_val, data(i, j));
        }
      }
    }

    // Store the statistics in layer_params
    float range = max_val - min_val;
    layer_params[layer_name] = {
        range / ((1 << default_bits) - 1), // scale
        -min_val,                          // zero_point
        default_bits                       // bits
    };
  }

  Matrix symmetric_quantize(const Matrix &weights, float scale, int bits) {
    Matrix quantized(weights.rows(), weights.cols());
    float max_val = float((1 << (bits - 1)) - 1);

    for (size_t i = 0; i < weights.rows(); ++i) {
      for (size_t j = 0; j < weights.cols(); ++j) {
        float val = weights(i, j);
        quantized(i, j) = std::round(val / scale) * scale;
        quantized(i, j) = std::clamp(quantized(i, j), -max_val, max_val);
      }
    }
    return quantized;
  }

public:
  QuantizationAwareTraining(bool symmetric = true)
      : use_symmetric_quantization(symmetric) {}

  void calibrate(const Transformer &model,
                 const std::vector<Matrix> &calibration_data) {
    auto all_weights = model.get_layer_weights();
    for (size_t i = 0; i < all_weights.size(); ++i) {
      std::string layer_name = "layer_" + std::to_string(i);
      collect_statistics(all_weights[i], calibration_data, layer_name);
    }
  }

  Matrix quantize_weights(const Matrix &weights,
                          const std::string &layer_name) {
    const auto &params = layer_params[layer_name];
    return symmetric_quantize(weights, params.scale, params.bits);
  }
};
#pragma once

#include <cmath>
#include <iostream>
#include "../matrix.hpp"

/**
 * @brief Handles dynamic loss scaling for mixed precision training.
 * 
 * This class implements dynamic loss scaling to prevent gradient underflow
 * in FP16 training while maintaining stability. It automatically adjusts
 * the scaling factor based on the presence of inf/nan values.
 */
class DynamicLossScaler {
public:
    DynamicLossScaler(float initial_scale = 65536.0f,
                      float scale_factor = 2.0f,
                      float scale_window = 2000,
                      float min_scale = 1.0f,
                      float max_scale = 65536.0f)
        : current_scale_(initial_scale)
        , scale_factor_(scale_factor)
        , scale_window_(scale_window)
        , min_scale_(min_scale)
        , max_scale_(max_scale)
        , stable_steps_(0) {}

    /**
     * @brief Get the current loss scale.
     */
    float get_scale() const { return current_scale_; }

    // Specialization for Matrix
    bool has_inf_or_nan(const Matrix& tensor) const {
        for (size_t i = 0; i < tensor.rows(); ++i) {
            for (size_t j = 0; j < tensor.cols(); ++j) {
                if (std::isinf(tensor(i, j)) || std::isnan(tensor(i, j))) {
                    return true;
                }
            }
        }
        return false;
    }

    // Specialization for Vector
    bool has_inf_or_nan(const Vector& tensor) const {
        for (size_t i = 0; i < tensor.size(); ++i) {
            if (std::isinf(tensor[i]) || std::isnan(tensor[i])) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Update the loss scale based on gradient behavior.
     * 
     * @param has_inf_or_nan Whether the current step had inf/nan values
     * @return bool Whether the current step should be skipped
     */
    bool update_scale(bool has_inf_or_nan) {
        if (has_inf_or_nan) {
            // Decrease scale on inf/nan
            current_scale_ = std::max(current_scale_ / scale_factor_, min_scale_);
            stable_steps_ = 0;
            std::cout << "Loss scale decreased to: " << current_scale_ << std::endl;
            return false;  // Skip this step
        } else {
            stable_steps_++;
            if (stable_steps_ >= scale_window_) {
                // Increase scale after window of stability
                float new_scale = std::min(current_scale_ * scale_factor_, max_scale_);
                if (new_scale != current_scale_) {
                    std::cout << "Loss scale increased to: " << new_scale << std::endl;
                }
                current_scale_ = new_scale;
                stable_steps_ = 0;
            }
            return true;  // Continue with this step
        }
    }

private:
    float current_scale_;
    float scale_factor_;
    float scale_window_;
    float min_scale_;
    float max_scale_;
    int stable_steps_;
}; 
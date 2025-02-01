#include "../../include/training/loss_tracker.hpp"
#include "../../include/tensor.hpp"
#include <numeric>  // For std::accumulate
#include <cmath>   // For std::log

void LossTracker::add_loss(float loss) {
    if (std::isfinite(loss)) {
        loss_history.push_back(loss);
        if (loss_history.size() > WINDOW_SIZE) {
            loss_history.pop_front();
        }
        update_statistics();
    }
}

bool LossTracker::should_adjust_lr() const {
    return loss_history.size() >= MIN_SAMPLES;
}

float LossTracker::get_trend() const {
    if (loss_history.size() < MIN_SAMPLES) return 1.0f;
    return recent_average / (overall_average + 1e-8f);
}

void LossTracker::update_statistics() {
    size_t n = loss_history.size();
    if (n == 0) return;

    size_t recent_window = std::max(size_t(1), std::min(n/4, size_t(10)));
    
    // Compute recent average (last 25% of samples)
    recent_average = std::accumulate(
        loss_history.end() - recent_window, 
        loss_history.end(), 
        0.0f) / recent_window;

    // Compute overall average
    overall_average = std::accumulate(
        loss_history.begin(), 
        loss_history.end(), 
        0.0f) / n;
}

float LossTracker::compute_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.size() != targets.size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }

    const size_t batch_size = predictions.rows();
    const size_t vocab_size = predictions.cols();
    float total_loss = 0.0f;

    // Compute cross-entropy loss for each item in the batch
    #pragma omp parallel for reduction(+:total_loss)
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < vocab_size; ++j) {
            if (targets(i, j) > 0.0f) {  // Only compute loss for actual targets
                // Add small epsilon to prevent log(0)
                const float epsilon = 1e-10f;
                float pred = std::clamp(predictions(i, j), epsilon, 1.0f - epsilon);
                total_loss -= targets(i, j) * std::log(pred);
            }
        }
    }

    // Average the loss over the batch
    float avg_loss = total_loss / static_cast<float>(batch_size);

    // Check for NaN or Inf
    if (!std::isfinite(avg_loss)) {
        throw std::runtime_error("Loss computation resulted in non-finite value");
    }

    return avg_loss;
}
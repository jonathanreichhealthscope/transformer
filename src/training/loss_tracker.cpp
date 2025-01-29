#include "../../include/training/loss_tracker.hpp"
#include "../../include/tensor.hpp"
#include <numeric>  // For std::accumulate

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
    float loss = 0.0f;
    // TODO: Implement actual loss computation
    return loss;
}
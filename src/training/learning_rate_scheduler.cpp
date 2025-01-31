#include "../../include/training/learning_rate_scheduler.hpp"

float LearningRateScheduler::get_learning_rate(const TrainingMetrics& metrics) {
    step_count++;
    float scale = compute_scale_factor(metrics);
    
    // Apply warmup schedule
    if (step_count < WARMUP_STEPS) {
        scale *= static_cast<float>(step_count) / WARMUP_STEPS;
    }
    
    current_lr = std::clamp(
        base_lr * scale,
        MIN_LEARNING_RATE,
        MAX_LEARNING_RATE
    );
    
    return current_lr;
}

float LearningRateScheduler::compute_scale_factor(const TrainingMetrics& metrics) {
    float loss_factor = compute_loss_factor(metrics.loss_trend);
    float grad_factor = compute_gradient_factor(metrics.grad_stats);
    float progress_factor = compute_progress_factor(metrics.epoch, metrics.step);
    
    return loss_factor * grad_factor * progress_factor;
}

float LearningRateScheduler::compute_loss_factor(float loss_trend) {
    if (loss_trend > 1.1f) return 0.5f;  // Loss increasing significantly
    if (loss_trend < 0.9f) return 1.2f;  // Loss decreasing significantly
    return 1.0f;  // Loss stable
}

float LearningRateScheduler::compute_gradient_factor(const RunningStatistics& grad_stats) {
    float grad_magnitude = std::sqrt(grad_stats.variance) + std::abs(grad_stats.mean);
    if (grad_magnitude > MAX_LEARNING_RATE) {
        return MAX_LEARNING_RATE / grad_magnitude;
    }
    return 1.0f;
}

float LearningRateScheduler::compute_progress_factor(size_t epoch, size_t step) {
    // Cosine decay schedule
    float progress = static_cast<float>(step) / (epoch + 1);
    return 0.5f * (1.0f + std::cos(progress * M_PI));
} 
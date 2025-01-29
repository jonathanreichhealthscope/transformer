#pragma once
#include <cmath>
#include "gradient_manager.hpp"
#include "training_metrics.hpp"

class LearningRateScheduler {
public:
    static constexpr float MIN_LEARNING_RATE = 1e-6f;
    static constexpr float MAX_LEARNING_RATE = 1e-2f;
    static constexpr float WARMUP_STEPS = 1000;

    explicit LearningRateScheduler(float initial_lr) 
        : current_lr(initial_lr), base_lr(initial_lr) {}

    float get_learning_rate(const TrainingMetrics& metrics);
    float get_current_lr() const { return current_lr; }

private:
    float current_lr;
    float base_lr;
    size_t step_count = 0;

    float compute_scale_factor(const TrainingMetrics& metrics);
    float compute_loss_factor(float loss_trend);
    float compute_gradient_factor(const RunningStatistics& grad_stats);
    float compute_progress_factor(size_t epoch, size_t step);
}; 
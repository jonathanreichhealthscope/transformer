#pragma once
#include "loss_tracker.hpp"
#include "gradient_manager.hpp"
#include "training_metrics.hpp"
#include <string>

class TrainingMonitor {
public:
    static constexpr float DIVERGENCE_THRESHOLD = 3.0f;
    static constexpr size_t MAX_NAN_OCCURRENCES = 5;
    static constexpr size_t MAX_EPOCHS = 1000;

    void log_metrics(const TrainingMetrics& metrics);
    bool should_stop_training();

private:
    LossTracker loss_tracker;
    GradientManager gradient_manager;
    size_t nan_counter = 0;
    size_t current_epoch = 0;

    bool detect_divergence();
    bool reached_convergence();
    bool exceeded_max_epochs();
    void update_running_statistics(const TrainingMetrics& metrics);
    void detect_anomalies(const TrainingMetrics& metrics);
    void log_to_tensorboard(const TrainingMetrics& metrics);
}; 
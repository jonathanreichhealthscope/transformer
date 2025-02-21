#pragma once
#include "loss_tracker.hpp"
#include "gradient_manager.hpp"
#include "learning_rate_scheduler.hpp"
#include "training_metrics.hpp"
#include <memory>

class TrainingStateManager {
public:
    static constexpr float INSTABILITY_THRESHOLD = 2.0f;

    TrainingStateManager(float initial_lr = 0.001f)
        : loss_tracker(std::make_unique<LossTracker>()),
          gradient_manager(std::make_unique<GradientManager>()),
          lr_scheduler(std::make_unique<LearningRateScheduler>(initial_lr)) {}

    void update_state(const TrainingMetrics& metrics);
    float get_learning_rate() const { return lr_scheduler->get_current_lr(); }
    bool is_stable() const { return !detect_instability(); }

private:
    std::unique_ptr<LossTracker> loss_tracker;
    std::unique_ptr<GradientManager> gradient_manager;
    std::unique_ptr<LearningRateScheduler> lr_scheduler;

    bool detect_instability() const;
    void recover_from_instability();
}; 
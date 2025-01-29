#pragma once
#include "../matrix.hpp"
#include "gradient_manager.hpp"

struct TrainingMetrics {
    float loss;
    Matrix gradients;
    size_t epoch;
    size_t step;
    float loss_trend;
    const RunningStatistics& grad_stats;

    // Constructor
    TrainingMetrics(float l, Matrix g, size_t e, size_t s, float lt, const RunningStatistics& gs)
        : loss(l), gradients(std::move(g)), epoch(e), step(s), loss_trend(lt), grad_stats(gs) {}
}; 
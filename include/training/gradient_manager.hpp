#pragma once
#include "../matrix.hpp"
#include <cmath>

struct RunningStatistics {
    float mean = 0.0f;
    float variance = 0.0f;
    size_t count = 0;

    void update(float new_mean, float new_variance) {
        float delta = new_mean - mean;
        count++;
        mean += delta / count;
        variance = ((count - 1) * variance + delta * (new_mean - mean)) / count;
    }
};

class GradientManager {
public:
    void process_gradients(Matrix& gradients);
    bool explosion_detected() const { return explosion_count > EXPLOSION_THRESHOLD; }
    const RunningStatistics& get_statistics() const { return grad_stats; }

    static constexpr float EXPLOSION_THRESHOLD = 5;
    static constexpr float MAX_GRAD_VALUE = 100.0f;

private:
    RunningStatistics grad_stats;
    size_t explosion_count = 0;

    void update_statistics(const Matrix& gradients);
    void clip_gradients(Matrix& gradients);
    bool detect_explosion(const Matrix& gradients);
    void recover_from_explosion(Matrix& gradients);
    void compute_running_statistics(const Matrix& gradients, float& mean, float& variance);
}; 
#ifndef LOSS_TRACKER_HPP
#define LOSS_TRACKER_HPP

#include <deque>
#include <vector>
#include "../../include/tensor.hpp"  // Just include tensor.hpp directly

class LossTracker {
public:
    void add_loss(float loss);
    bool should_adjust_lr() const;
    float get_trend() const;
    float compute_loss(const Tensor& predictions, const Tensor& targets);  // Declaration

private:
    void update_statistics();
    std::deque<float> loss_history;
    float recent_average = 0.0f;
    float overall_average = 0.0f;

    static const size_t WINDOW_SIZE = 100;
    static const size_t MIN_SAMPLES = 10;
};

#endif // LOSS_TRACKER_HPP 
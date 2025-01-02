#include "../include/components.hpp"
#include <iostream>
#include <cmath>

void print_matrix_stats(const Matrix& m) {
    if (m.empty()) {
        std::cout << "Matrix is empty" << std::endl;
        return;
    }

    float min_val = m.data()[0];
    float max_val = m.data()[0];
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (size_t i = 0; i < m.size(); i++) {
        float val = m.data()[i];
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
        sum_sq += val * val;
    }

    float mean = sum / m.size();
    float variance = (sum_sq / m.size()) - (mean * mean);
    float std_dev = std::sqrt(variance);

    std::cout << "  Range: [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Std Dev: " << std_dev << std::endl;
} 
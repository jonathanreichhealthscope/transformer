#include "../include/components.hpp"
#include <iostream>
#include <cmath>
#include <omp.h>

void print_matrix_stats(const Matrix& m) {
    if (m.empty()) {
        std::cout << "Matrix is empty" << std::endl;
        return;
    }

    float min_val = m.data()[0];
    float max_val = m.data()[0];
    float sum = 0.0f;
    float sum_sq = 0.0f;

    #pragma omp parallel
    {
        float local_min = std::numeric_limits<float>::max();
        float local_max = std::numeric_limits<float>::lowest();
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;

        #pragma omp for simd nowait
        for (size_t i = 0; i < m.size(); i++) {
            float val = m.data()[i];
            local_min = std::min(local_min, val);
            local_max = std::max(local_max, val);
            local_sum += val;
            local_sum_sq += val * val;
        }

        #pragma omp critical
        {
            min_val = std::min(min_val, local_min);
            max_val = std::max(max_val, local_max);
            sum += local_sum;
            sum_sq += local_sum_sq;
        }
    }

    float mean = sum / m.size();
    float variance = (sum_sq / m.size()) - (mean * mean);
    float std_dev = std::sqrt(variance);

    std::cout << "  Range: [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Std Dev: " << std_dev << std::endl;
} 
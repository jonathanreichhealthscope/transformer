#pragma once
#include "../matrix.hpp"
#include <vector>
#include "kernel_declarations.cuh"

namespace cuda {
    void topk(const std::vector<float>& scores, Matrix& output_scores, 
              std::vector<int>& output_indices, int k);
              
    void beam_search_step(const Matrix& current_scores, const Matrix& next_scores,
                         Matrix& output_scores, std::vector<int>& output_indices, int beam_width);
} 
#pragma once

#include "components.hpp"
#ifdef USE_CUDA
#include <cuda_fp16.h>
using half_type = __half;
#else
using half_type = float;  // Fallback to float when CUDA is not available
#endif

class HalfPrecisionTraining {
public:
    static void convert_to_fp16(Matrix& matrix);
    
    static void convert_to_fp32(Matrix& matrix);

private:
    static std::vector<half_type> half_data;
}; 
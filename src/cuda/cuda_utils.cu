#include "../../include/cuda/cuda_utils.cuh"
#include "../../include/cuda/cuda_utils_impl.cuh"

namespace cuda_kernels {
    // Feed forward kernels - explicit instantiations
    template void launch_cuda_kernel<decltype(&feed_forward_backward_kernel_1)>(
        decltype(&feed_forward_backward_kernel_1),
        dim3, dim3, size_t, cudaStream_t,
        const float*, const float*, float*, int, int, int);
} 
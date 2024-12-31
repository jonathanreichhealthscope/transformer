#pragma once
#include "kernel_traits.cuh"

// This header should only be included by .cu files
namespace cuda_kernels {
// Trait specialization definitions
template <>
struct cuda_kernel_traits<decltype(&feed_forward_backward_kernel_1)> {
  using arg_tuple =
      std::tuple<const float *, const float *, float *, int, int, int>;
};

template <> struct cuda_kernel_traits<decltype(&gelu_backward_kernel)> {
  using arg_tuple = std::tuple<const float *, float *, int>;
};

template <>
struct cuda_kernel_traits<decltype(&feed_forward_backward_kernel_2)> {
  using arg_tuple =
      std::tuple<const float *, const float *, float *, int, int, int>;
};
} // namespace cuda_kernels
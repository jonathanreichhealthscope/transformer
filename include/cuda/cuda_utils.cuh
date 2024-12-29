#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#ifdef __CUDACC__
// Kernel launch macro - only available in CUDA source files
#define CUDA_LAUNCH(kernel, gridSize, blockSize, sharedMem, stream, ...)       \
  do {                                                                         \
    kernel<<<gridSize, blockSize, sharedMem, stream>>>(__VA_ARGS__);           \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)
#else
// Stub for non-CUDA source files
#define CUDA_LAUNCH(kernel, gridSize, blockSize, sharedMem, stream, ...)       \
  do {                                                                         \
    fprintf(stderr, "CUDA_LAUNCH called from non-CUDA code\n");                \
    exit(EXIT_FAILURE);                                                        \
  } while (0)
#endif

#endif
#include "../../include/cuda/quantization_kernels.cuh"
#include "../../include/cuda/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <cfloat>  // For FLT_MAX
#include <cstdio>  // For fprintf and stderr

// Helper function for parallel reduction to find min/max
__device__ void warp_reduce_minmax(volatile float* smin, volatile float* smax, int tid) {
    smin[tid] = min(smin[tid], smin[tid + 32]);
    smax[tid] = max(smax[tid], smax[tid + 32]);
    smin[tid] = min(smin[tid], smin[tid + 16]);
    smax[tid] = max(smax[tid], smax[tid + 16]);
    smin[tid] = min(smin[tid], smin[tid + 8]);
    smax[tid] = max(smax[tid], smax[tid + 8]);
    smin[tid] = min(smin[tid], smin[tid + 4]);
    smax[tid] = max(smax[tid], smax[tid + 4]);
    smin[tid] = min(smin[tid], smin[tid + 2]);
    smax[tid] = max(smax[tid], smax[tid + 2]);
    smin[tid] = min(smin[tid], smin[tid + 1]);
    smax[tid] = max(smax[tid], smax[tid + 1]);
}

__global__ void find_minmax_kernel(const float* input, size_t size, float* min_out, float* max_out) {
    __shared__ float smin[256];
    __shared__ float smax[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize with first values or neutral elements
    smin[tid] = gid < size ? input[gid] : FLT_MAX;
    smax[tid] = gid < size ? input[gid] : -FLT_MAX;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            smin[tid] = min(smin[tid], smin[tid + s]);
            smax[tid] = max(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }
    
    // Final warp reduction
    if (tid < 32) warp_reduce_minmax(smin, smax, tid);
    
    // Write result for this block
    if (tid == 0) {
        min_out[blockIdx.x] = smin[0];
        max_out[blockIdx.x] = smax[0];
    }
}

void find_minmax_cuda(const float* input, size_t size, float* min_val, float* max_val) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    // Allocate temporary device memory for block results
    float *d_min, *d_max;
    CUDA_CHECK(cudaMalloc(&d_min, grid_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max, grid_size * sizeof(float)));
    
    // First pass: find min/max per block
    find_minmax_kernel<<<grid_size, block_size>>>(input, size, d_min, d_max);
    
    // Second pass: reduce block results
    find_minmax_kernel<<<1, block_size>>>(d_min, grid_size, min_val, max_val);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_min));
    CUDA_CHECK(cudaFree(d_max));
}

__global__ void quantize_kernel(
    const float* input,
    float* output,
    size_t size,
    float scale,
    float zero_point
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = roundf(val / scale + zero_point);
    }
}

__global__ void dequantize_kernel(
    const float* input,
    float* output,
    size_t size,
    float scale,
    float zero_point
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = (val - zero_point) * scale;
    }
} 
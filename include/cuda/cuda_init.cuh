#pragma once
#include <cublas_v2.h>

// Global cuBLAS handle
extern cublasHandle_t cublas_handle;

// CUDA initialization functions
void initialize_cuda();
void cleanup_cuda(); 
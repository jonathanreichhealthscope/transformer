#pragma once
#include <cublas_v2.h>
#include <stdio.h>

// cuBLAS error checking macro
#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t status = call;                                                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                     \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status);            \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
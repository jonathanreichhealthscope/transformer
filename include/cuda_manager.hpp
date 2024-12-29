#pragma once
#include <cuda_runtime.h>

class CudaManager {
public:
  CudaManager();
  ~CudaManager();

  void synchronize();
  void *allocate(size_t size);
  void deallocate(void *ptr);
};
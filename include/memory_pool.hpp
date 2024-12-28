#pragma once
#include <vector>
#include <memory>
#include <mutex>
#include "components.hpp"

class MemoryPool {
private:
    struct Block {
        float* data;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks;
    std::mutex mutex;
    
    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }
    
public:
    static float* allocate(size_t size);
    static void deallocate(float* ptr);
    static void clear_pool();
    
    // CUDA memory pool
    static float* cuda_allocate(size_t size);
    static void cuda_deallocate(float* ptr);
};

// Smart pointer for automatic memory management
template<typename T>
class PooledPtr {
private:
    T* ptr;
    bool is_cuda;
    
public:
    explicit PooledPtr(size_t size, bool cuda = false);
    ~PooledPtr();
    
    T* get() { return ptr; }
    const T* get() const { return ptr; }
}; 
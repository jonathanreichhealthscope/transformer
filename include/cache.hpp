#pragma once
#include "matrix.hpp"

struct KVCache {
    Matrix key_cache;   // Single matrix for keys
    Matrix value_cache; // Single matrix for values

    // Constructor with size
    explicit KVCache(size_t max_len = 0);

    // Clear the cache
    void clear();

    // Check if cache is empty
    bool empty() const {
        return key_cache.empty() || value_cache.empty();
    }

    // Update cache with new keys and values
    void update(const Matrix& new_keys, const Matrix& new_values);

    // Get cached key-value pairs
    std::pair<Matrix, Matrix> get_cached_kv() const;
};
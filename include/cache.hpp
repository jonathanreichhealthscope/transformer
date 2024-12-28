#pragma once
#include "components.hpp"
#include <deque>

class KVCache {
private:
    std::deque<Matrix> key_cache;
    std::deque<Matrix> value_cache;
    size_t max_seq_length;
    
public:
    explicit KVCache(size_t max_length);
    void update(const Matrix& new_keys, const Matrix& new_values);
    std::pair<Matrix, Matrix> get_cached_kv() const;
    void clear();
}; 
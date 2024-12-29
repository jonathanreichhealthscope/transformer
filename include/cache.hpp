#pragma once
#include "components.hpp"
#include <vector>
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>

class KVCache {
private:
    std::vector<Matrix> key_cache;
    std::vector<Matrix> value_cache;
    size_t max_length;
    size_t current_length;

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & ar) {
        ar(key_cache, value_cache, max_length, current_length);
    }

public:
    KVCache() : max_length(0), current_length(0) {}
    explicit KVCache(size_t max_len);
    
    void update(const Matrix& new_keys, const Matrix& new_values);
    std::pair<Matrix, Matrix> get_cached_kv() const;
    void clear();
}; 
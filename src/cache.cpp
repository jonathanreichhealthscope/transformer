#include "../include/cache.hpp"

KVCache::KVCache(size_t max_len) : max_length(max_len), current_length(0) {
    clear();
}

void KVCache::update(const Matrix& new_keys, const Matrix& new_values) {
    // Add new keys and values
    key_cache.push_back(new_keys);
    value_cache.push_back(new_values);
    current_length += new_keys.rows();
    
    // Remove old entries if we exceed max length
    while (current_length > max_length) {
        size_t rows_to_remove = key_cache.front().rows();
        key_cache.erase(key_cache.begin());
        value_cache.erase(value_cache.begin());
        current_length -= rows_to_remove;
    }
}

std::pair<Matrix, Matrix> KVCache::get_cached_kv() const {
    if (key_cache.empty()) {
        return {Matrix(), Matrix()};
    }
    
    // Calculate total rows
    size_t total_rows = 0;
    for (const auto& k : key_cache) {
        total_rows += k.rows();
    }
    
    // Create concatenated matrices
    Matrix concatenated_keys(total_rows, key_cache[0].cols());
    Matrix concatenated_values(total_rows, value_cache[0].cols());
    
    // Copy data
    size_t current_row = 0;
    for (size_t i = 0; i < key_cache.size(); ++i) {
        const Matrix& k = key_cache[i];
        const Matrix& v = value_cache[i];
        
        for (size_t row = 0; row < k.rows(); ++row) {
            for (size_t col = 0; col < k.cols(); ++col) {
                concatenated_keys(current_row + row, col) = k(row, col);
            }
            for (size_t col = 0; col < v.cols(); ++col) {
                concatenated_values(current_row + row, col) = v(row, col);
            }
        }
        current_row += k.rows();
    }
    
    return {concatenated_keys, concatenated_values};
}

void KVCache::clear() {
    key_cache.clear();
    value_cache.clear();
    current_length = 0;
} 
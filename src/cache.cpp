#include "../include/cache.hpp"

KVCache::KVCache(size_t max_len) : max_length(max_len), current_length(0) {
  clear();
}

void KVCache::update(const Matrix &new_keys, const Matrix &new_values) {
  // Add new keys and values
  key_cache.push_back(new_keys);
  value_cache.push_back(new_values);
  current_length += new_keys.rows();

  // Remove old entries if we exceed max length
  while (current_length > max_length) {
    current_length -= key_cache.front().rows();
    key_cache.erase(key_cache.begin());
    value_cache.erase(value_cache.begin());
  }
}

std::pair<Matrix, Matrix> KVCache::get_cached_kv() const {
  if (key_cache.empty()) {
    return {Matrix(), Matrix()};
  }

  size_t total_rows = current_length;
  size_t cols = key_cache[0].cols();

  Matrix keys(total_rows, cols);
  Matrix values(total_rows, cols);

  size_t current_row = 0;
  for (size_t i = 0; i < key_cache.size(); ++i) {
    for (size_t row = 0; row < key_cache[i].rows(); ++row) {
      for (size_t col = 0; col < cols; ++col) {
        keys(current_row + row, col) = key_cache[i](row, col);
        values(current_row + row, col) = value_cache[i](row, col);
      }
    }
    current_row += key_cache[i].rows();
  }

  return {keys, values};
}

void KVCache::clear() {
  key_cache.clear();
  value_cache.clear();
  current_length = 0;
}
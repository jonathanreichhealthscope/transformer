#pragma once
#include "components.hpp"
#include <algorithm>
#include <chrono>
#include <optional>
#include <unordered_map>
#include <vector>

enum class CacheReplacementPolicy { LRU, LFU, ARC };

template <typename T> class AdvancedCache {
private:
  struct CacheEntry {
    T data;
    size_t frequency;
    std::chrono::steady_clock::time_point last_access;
  };

  std::unordered_map<size_t, CacheEntry> cache;
  size_t capacity;
  CacheReplacementPolicy policy;

  void evict_lru() {
    auto oldest = std::min_element(
        cache.begin(), cache.end(), [](const auto &a, const auto &b) {
          return a.second.last_access < b.second.last_access;
        });
    if (oldest != cache.end()) {
      cache.erase(oldest);
    }
  }

  void evict_lfu() {
    auto least_used = std::min_element(
        cache.begin(), cache.end(), [](const auto &a, const auto &b) {
          return a.second.frequency < b.second.frequency;
        });
    if (least_used != cache.end()) {
      cache.erase(least_used);
    }
  }

  void evict_adaptive() {
    // Fallback to LRU for now
    evict_lru();
  }

  void evict() {
    switch (policy) {
    case CacheReplacementPolicy::LRU:
      evict_lru();
      break;
    case CacheReplacementPolicy::LFU:
      evict_lfu();
      break;
    case CacheReplacementPolicy::ARC:
      evict_adaptive();
      break;
    }
  }

public:
  AdvancedCache(size_t capacity_ = 1024,
                CacheReplacementPolicy policy_ = CacheReplacementPolicy::LRU)
      : capacity(capacity_), policy(policy_) {}

  void put(size_t key, const T &value) {
    if (cache.size() >= capacity) {
      evict();
    }
    cache[key] = {value, 1, std::chrono::steady_clock::now()};
  }

  std::optional<T> get(size_t key) {
    auto it = cache.find(key);
    if (it != cache.end()) {
      it->second.frequency++;
      it->second.last_access = std::chrono::steady_clock::now();
      return it->second.data;
    }
    return std::nullopt;
  }
};

class KVCache {
private:
  std::vector<Matrix> key_cache;
  std::vector<Matrix> value_cache;
  size_t max_length;
  size_t current_length;

public:
  KVCache() : max_length(0), current_length(0) {}
  explicit KVCache(size_t max_len);

  void update(const Matrix &new_keys, const Matrix &new_values);
  std::pair<Matrix, Matrix> get_cached_kv() const;
  void clear();
};
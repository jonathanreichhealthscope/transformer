#pragma once
#include <algorithm>
#include <chrono>
#include <list>
#include <memory>
#include <unordered_map>

enum class CacheReplacementPolicy {
  LRU,
  LFU,
  ARC // Adaptive Replacement Cache
};

template <typename T> class TensorCache {
private:
  struct Node {
    T data;
    size_t key;
    size_t frequency;
    std::chrono::steady_clock::time_point last_access;
    bool is_dirty;
  };

  std::unordered_map<size_t, typename std::list<Node>::iterator> cache_map;
  std::list<Node> cache_list;
  size_t capacity;
  CacheReplacementPolicy policy;

  void write_back(const T &data, size_t key) {
    // Implement persistence logic here
  }

  typename std::list<Node>::iterator select_victim() {
    switch (policy) {
    case CacheReplacementPolicy::LRU:
      return select_lru_victim();
    case CacheReplacementPolicy::LFU:
      return select_lfu_victim();
    case CacheReplacementPolicy::ARC:
      return select_arc_victim();
    default:
      return cache_list.begin();
    }
  }

  typename std::list<Node>::iterator select_lru_victim() {
    return std::min_element(cache_list.begin(), cache_list.end(),
                            [](const Node &a, const Node &b) {
                              return a.last_access < b.last_access;
                            });
  }

  typename std::list<Node>::iterator select_lfu_victim() {
    return std::min_element(
        cache_list.begin(), cache_list.end(),
        [](const Node &a, const Node &b) { return a.frequency < b.frequency; });
  }

  typename std::list<Node>::iterator select_arc_victim() {
    // Implement ARC victim selection
    return select_lru_victim(); // Fallback to LRU for now
  }

public:
  TensorCache(size_t capacity_ = 1024,
              CacheReplacementPolicy policy_ = CacheReplacementPolicy::ARC)
      : capacity(capacity_), policy(policy_) {}

  void put(size_t key, const T &value) {
    auto it = cache_map.find(key);
    if (it != cache_map.end()) {
      // Update existing entry
      it->second->data = value;
      it->second->frequency++;
      it->second->last_access = std::chrono::steady_clock::now();
      it->second->is_dirty = true;
    } else {
      // Add new entry
      if (cache_list.size() >= capacity) {
        evict();
      }

      Node new_node{value, key, 1, std::chrono::steady_clock::now(), true};

      cache_list.push_front(new_node);
      cache_map[key] = cache_list.begin();
    }
  }

  std::optional<T> get(size_t key) {
    auto it = cache_map.find(key);
    if (it != cache_map.end()) {
      it->second->frequency++;
      it->second->last_access = std::chrono::steady_clock::now();
      return it->second->data;
    }
    return std::nullopt;
  }

  void evict() {
    if (cache_list.empty())
      return;

    auto victim = select_victim();
    if (victim->is_dirty) {
      write_back(victim->data, victim->key);
    }
    cache_map.erase(victim->key);
    cache_list.erase(victim);
  }

  void clear() {
    while (!cache_list.empty()) {
      evict();
    }
  }
};
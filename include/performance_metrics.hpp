#pragma once
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

class PerformanceMetrics {
private:
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
    std::unordered_map<std::string, double> accumulated_times;
    std::unordered_map<std::string, size_t> call_counts;
    
    // Memory tracking
    size_t peak_memory_usage;
    std::vector<size_t> memory_samples;

public:
    // Timing methods
    void start_timer(const std::string& name);
    void stop_timer(const std::string& name);
    double get_average_time(const std::string& name) const;
    
    // Memory tracking
    void record_memory_usage(size_t bytes_used);
    size_t get_peak_memory() const;
    double get_average_memory() const;
    
    // Attention specific metrics
    void record_attention_flops(size_t seq_length, size_t num_heads, size_t head_dim);
    double get_attention_gflops() const;
    
    // Reporting
    void print_metrics() const;
    void reset();
}; 
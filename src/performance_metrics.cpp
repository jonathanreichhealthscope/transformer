#include "../include/performance_metrics.hpp"
#include <iostream>
#include <numeric>

void PerformanceMetrics::start_timer(const std::string& name) {
    start_times[name] = std::chrono::high_resolution_clock::now();
}

void PerformanceMetrics::stop_timer(const std::string& name) {
    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_times[name])
            .count() /
        1000.0; // Convert to milliseconds

    accumulated_times[name] += duration;
    call_counts[name]++;
}

double PerformanceMetrics::get_average_time(const std::string& name) const {
    if (call_counts.find(name) == call_counts.end() || call_counts.at(name) == 0) {
        return 0.0;
    }
    return accumulated_times.at(name) / call_counts.at(name);
}

void PerformanceMetrics::record_memory_usage(size_t bytes_used) {
    memory_samples.push_back(bytes_used);
    peak_memory_usage = std::max(peak_memory_usage, bytes_used);
}

size_t PerformanceMetrics::get_peak_memory() const {
    return peak_memory_usage;
}

double PerformanceMetrics::get_average_memory() const {
    if (memory_samples.empty())
        return 0.0;
    return std::accumulate(memory_samples.begin(), memory_samples.end(), 0.0) /
           memory_samples.size();
}

void PerformanceMetrics::record_attention_flops(size_t seq_length, size_t num_heads,
                                                size_t head_dim) {
    // Calculate FLOPs for attention computation
    // Q*K^T: 2*seq_length^2*head_dim operations per head
    // Softmax: 3*seq_length operations per head
    // Attention*V: 2*seq_length^2*head_dim operations per head
    size_t flops = num_heads * (4 * seq_length * seq_length * head_dim + 3 * seq_length);
    accumulated_times["attention_flops"] += flops;
    call_counts["attention_flops"]++;
}

double PerformanceMetrics::get_attention_gflops() const {
    if (call_counts.find("attention_flops") == call_counts.end() ||
        call_counts.at("attention_flops") == 0) {
        return 0.0;
    }
    return (accumulated_times.at("attention_flops") / 1e9) /
           (accumulated_times.at("attention_computation") / 1000.0); // GFLOPS
}

void PerformanceMetrics::print_metrics() const {
    std::cout << "\n=== Performance Metrics ===\n";

    // Print timing metrics
    for (const auto& [name, total_time] : accumulated_times) {
        if (name != "attention_flops") {
            double avg_time = get_average_time(name);
            std::cout << name << ": " << avg_time << "ms (avg), " << call_counts.at(name)
                      << " calls\n";
        }
    }

    // Print memory metrics
    std::cout << "\nMemory Usage:\n"
              << "Peak: " << (get_peak_memory() / 1024.0 / 1024.0) << " MB\n"
              << "Average: " << (get_average_memory() / 1024.0 / 1024.0) << " MB\n";

    // Print attention metrics
    if (call_counts.find("attention_flops") != call_counts.end()) {
        std::cout << "\nAttention Performance:\n"
                  << "GFLOPS: " << get_attention_gflops() << "\n";
    }
}

void PerformanceMetrics::reset() {
    start_times.clear();
    accumulated_times.clear();
    call_counts.clear();
    peak_memory_usage = 0;
    memory_samples.clear();
}
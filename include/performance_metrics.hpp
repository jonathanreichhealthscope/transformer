#pragma once
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief Performance monitoring and profiling for transformer operations.
 * 
 * The PerformanceMetrics class provides comprehensive performance tracking
 * capabilities for transformer model operations, including:
 * - High-resolution timing measurements
 * - Memory usage tracking
 * - FLOPS calculations for attention operations
 * - Statistical aggregation of metrics
 */
class PerformanceMetrics {
  private:
    /// Stores start times for active timing operations
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
    
    /// Accumulates total time spent in each operation
    std::unordered_map<std::string, double> accumulated_times;
    
    /// Tracks number of calls to each operation
    std::unordered_map<std::string, size_t> call_counts;

    // Memory tracking
    size_t peak_memory_usage;        ///< Maximum memory usage observed
    std::vector<size_t> memory_samples; ///< Historical memory usage data

  public:
    /**
     * @brief Starts timing an operation.
     * 
     * Records the start time for a named operation. Must be paired
     * with a corresponding stop_timer call.
     * 
     * @param name Identifier for the operation being timed
     */
    void start_timer(const std::string& name);

    /**
     * @brief Stops timing an operation.
     * 
     * Calculates elapsed time since the corresponding start_timer
     * call and updates statistics.
     * 
     * @param name Identifier for the operation being timed
     * @throws std::runtime_error if no matching start_timer call exists
     */
    void stop_timer(const std::string& name);

    /**
     * @brief Gets average execution time for an operation.
     * 
     * Calculates the mean execution time across all recorded
     * instances of the named operation.
     * 
     * @param name Identifier for the operation
     * @return Average time in milliseconds
     */
    double get_average_time(const std::string& name) const;

    /**
     * @brief Records current memory usage.
     * 
     * Updates peak memory usage if necessary and stores
     * the sample for statistical analysis.
     * 
     * @param bytes_used Current memory usage in bytes
     */
    void record_memory_usage(size_t bytes_used);

    /**
     * @brief Gets peak memory usage.
     * @return Maximum memory usage observed in bytes
     */
    size_t get_peak_memory() const;

    /**
     * @brief Gets average memory usage.
     * @return Mean memory usage across all samples in bytes
     */
    double get_average_memory() const;

    /**
     * @brief Records FLOPs for an attention operation.
     * 
     * Calculates and accumulates floating point operations
     * for attention computation based on input dimensions.
     * 
     * @param seq_length Sequence length
     * @param num_heads Number of attention heads
     * @param head_dim Dimension of each head
     */
    void record_attention_flops(size_t seq_length, size_t num_heads, size_t head_dim);

    /**
     * @brief Gets attention computation performance.
     * @return Attention computation speed in GFLOPS
     */
    double get_attention_gflops() const;

    /**
     * @brief Prints all collected metrics.
     * 
     * Outputs a formatted report of:
     * - Timing statistics for all operations
     * - Memory usage statistics
     * - Computational performance metrics
     */
    void print_metrics() const;

    /**
     * @brief Resets all metrics.
     * 
     * Clears all accumulated statistics and resets counters
     * to their initial state.
     */
    void reset();
};
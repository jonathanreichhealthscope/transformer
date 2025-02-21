#pragma once

#include "transformer.hpp"
#include "utils.hpp"
#include <vector>
#include <random>
#include <memory>

// Structure to hold hyperparameter ranges
struct HyperparameterRanges {
    // Architecture ranges
    std::vector<size_t> num_layers_range{2, 4, 6, 8};
    std::vector<size_t> num_heads_range{4, 8, 12, 16};
    std::vector<size_t> hidden_size_range{256, 512, 768, 1024};
    std::vector<size_t> intermediate_size_range{512, 1024, 2048, 4096};
    std::vector<size_t> head_dim_range{32, 64, 96, 128};
    
    // Learning rate parameters
    std::vector<float> initial_lr_range{1e-5f, 5e-5f, 1e-4f, 5e-4f};
    std::vector<float> peak_lr_range{1e-4f, 5e-4f, 1e-3f, 5e-3f};
    std::vector<size_t> warmup_steps_range{50, 100, 200, 500};
    std::vector<float> decay_factor_range{0.95f, 0.97f, 0.98f, 0.99f};
    
    // Training parameters
    std::vector<float> dropout_rate_range{0.0f, 0.1f, 0.2f, 0.3f};
    std::vector<float> weight_decay_range{0.0f, 0.01f, 0.1f, 0.2f};
    std::vector<size_t> early_stopping_patience_range{2, 3, 4, 5};
    std::vector<float> early_stopping_threshold_range{1.2f, 1.5f, 1.8f, 2.0f};
    std::vector<float> gradient_clip_threshold_range{1.0f, 3.0f, 5.0f, 10.0f};
    std::vector<float> layer_norm_epsilon_range{1e-6f, 1e-5f, 1e-4f};
    
    // Memory and optimization
    std::vector<size_t> memory_pool_size_range{1024, 2048, 4096, 8192};
    std::vector<size_t> gradient_accumulation_steps_range{1, 2, 4, 8};
};

// Structure to hold a specific hyperparameter configuration
struct HyperparameterConfig {
    // Architecture parameters
    size_t num_layers;
    size_t num_heads;
    size_t hidden_size;
    size_t intermediate_size;
    size_t head_dim;
    
    // Learning rate parameters
    float initial_lr;
    float peak_lr;
    size_t warmup_steps;
    float decay_factor;
    
    // Training parameters
    float dropout_rate;
    float weight_decay;
    size_t early_stopping_patience;
    float early_stopping_threshold;
    float gradient_clip_threshold;
    float layer_norm_epsilon;
    
    // Memory and optimization
    size_t memory_pool_size;
    size_t gradient_accumulation_steps;
    
    // Convert to TransformerConfig
    TransformerConfig to_transformer_config() const;
    
    // Serialization
    void save(const std::string& path) const;
    static HyperparameterConfig load(const std::string& path);
};

// Results from a single hyperparameter evaluation
struct TuningResult {
    HyperparameterConfig config;
    float mean_validation_loss;
    float validation_loss_std;
    size_t early_stops;
    std::vector<float> fold_scores;
    
    // For sorting results - inline implementation
    bool operator<(const TuningResult& other) const {
        // Primary sort by mean validation loss
        if (mean_validation_loss != other.mean_validation_loss) {
            return mean_validation_loss < other.mean_validation_loss;
        }
        // Secondary sort by validation loss standard deviation
        if (validation_loss_std != other.validation_loss_std) {
            return validation_loss_std < other.validation_loss_std;
        }
        // Tertiary sort by number of early stops (fewer is better)
        return early_stops < other.early_stops;
    }
};

class HyperparameterTuner {
public:
    HyperparameterTuner(const HyperparameterRanges& ranges, 
                        size_t num_trials = 50,
                        size_t num_folds = 5,
                        unsigned int random_seed = 42);
    
    // Main tuning function
    std::vector<TuningResult> tune(const std::vector<std::pair<std::string, std::string>>& training_data,
                                  const Tokenizer& tokenizer);
    
    // Get best configuration
    HyperparameterConfig get_best_config() const;
    
    // Save/load results
    void save_results(const std::string& path) const;
    void load_results(const std::string& path);

private:
    // Internal helper functions
    HyperparameterConfig sample_random_config();
    TuningResult evaluate_config(const HyperparameterConfig& config,
                               const std::vector<std::pair<std::string, std::string>>& data,
                               const Tokenizer& tokenizer);
    
    // Member variables
    HyperparameterRanges ranges_;
    size_t num_trials_;
    size_t num_folds_;
    std::mt19937 rng_;
    std::vector<TuningResult> results_;
    
    // Validation helpers
    bool validate_config(const HyperparameterConfig& config) const;
    void log_trial_progress(size_t current_trial, const TuningResult& result) const;
};

// Utility functions
namespace HyperparameterUtils {
    // Random sampling helpers with explicit declarations
    template<typename T>
    T sample_from_range(const std::vector<T>& range, std::mt19937& rng);
    
    // Explicit declarations for the template function
    extern template float sample_from_range<float>(const std::vector<float>&, std::mt19937&);
    extern template size_t sample_from_range<size_t>(const std::vector<size_t>&, std::mt19937&);
    
    // Validation helpers
    bool is_valid_architecture(const HyperparameterConfig& config);
    bool is_valid_learning_rate(const HyperparameterConfig& config);
    bool is_valid_training_params(const HyperparameterConfig& config);
    
    // Logging helpers
    void log_config(const HyperparameterConfig& config);
    void log_result(const TuningResult& result);
} 
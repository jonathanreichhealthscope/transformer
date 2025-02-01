#include "../include/hyperparameter_tuner.hpp"
#include "../include/transformer.hpp"
#include "../include/utils.hpp"
#include "../include/tokenizer.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <random>
#include <nlohmann/json.hpp>

// Explicit template instantiations for sample_from_range
template float HyperparameterUtils::sample_from_range<float>(const std::vector<float>&, std::mt19937&);
template size_t HyperparameterUtils::sample_from_range<size_t>(const std::vector<size_t>&, std::mt19937&);

// Convert HyperparameterConfig to TransformerConfig
TransformerConfig HyperparameterConfig::to_transformer_config() const {
    TransformerConfig config;
    
    // Architecture parameters
    config.num_layers = num_layers;
    config.num_heads = num_heads;
    config.hidden_size = hidden_size;
    config.intermediate_size = intermediate_size;
    config.head_dim = head_dim;
    
    // Training parameters
    config.dropout_rate = dropout_rate;
    config.weight_decay = weight_decay;
    
    // Memory and optimization
    config.memory_pool_size = memory_pool_size;
    config.layer_norm_epsilon = layer_norm_epsilon;
    
    return config;
}

// Serialization
void HyperparameterConfig::save(const std::string& path) const {
    nlohmann::json j;
    
    // Architecture parameters
    j["num_layers"] = num_layers;
    j["num_heads"] = num_heads;
    j["hidden_size"] = hidden_size;
    j["intermediate_size"] = intermediate_size;
    j["head_dim"] = head_dim;
    
    // Learning rate parameters
    j["initial_lr"] = initial_lr;
    j["peak_lr"] = peak_lr;
    j["warmup_steps"] = warmup_steps;
    j["decay_factor"] = decay_factor;
    
    // Training parameters
    j["dropout_rate"] = dropout_rate;
    j["weight_decay"] = weight_decay;
    j["early_stopping_patience"] = early_stopping_patience;
    j["early_stopping_threshold"] = early_stopping_threshold;
    j["gradient_clip_threshold"] = gradient_clip_threshold;
    j["layer_norm_epsilon"] = layer_norm_epsilon;
    
    // Memory and optimization
    j["memory_pool_size"] = memory_pool_size;
    j["gradient_accumulation_steps"] = gradient_accumulation_steps;
    
    std::ofstream file(path);
    file << j.dump(4);
}

HyperparameterConfig HyperparameterConfig::load(const std::string& path) {
    std::ifstream file(path);
    nlohmann::json j;
    file >> j;
    
    HyperparameterConfig config;
    
    // Architecture parameters
    config.num_layers = j["num_layers"];
    config.num_heads = j["num_heads"];
    config.hidden_size = j["hidden_size"];
    config.intermediate_size = j["intermediate_size"];
    config.head_dim = j["head_dim"];
    
    // Learning rate parameters
    config.initial_lr = j["initial_lr"];
    config.peak_lr = j["peak_lr"];
    config.warmup_steps = j["warmup_steps"];
    config.decay_factor = j["decay_factor"];
    
    // Training parameters
    config.dropout_rate = j["dropout_rate"];
    config.weight_decay = j["weight_decay"];
    config.early_stopping_patience = j["early_stopping_patience"];
    config.early_stopping_threshold = j["early_stopping_threshold"];
    config.gradient_clip_threshold = j["gradient_clip_threshold"];
    config.layer_norm_epsilon = j["layer_norm_epsilon"];
    
    // Memory and optimization
    config.memory_pool_size = j["memory_pool_size"];
    config.gradient_accumulation_steps = j["gradient_accumulation_steps"];
    
    return config;
}

// HyperparameterTuner implementation
HyperparameterTuner::HyperparameterTuner(const HyperparameterRanges& ranges,
                                         size_t num_trials,
                                         size_t num_folds,
                                         unsigned int seed)
    : ranges_(ranges), num_trials_(num_trials), num_folds_(num_folds), rng_(seed) {
    std::cout << "Initializing HyperparameterTuner with:"
              << "\n- Number of trials: " << num_trials
              << "\n- Number of folds: " << num_folds
              << "\n- Random seed: " << seed << std::endl;
}

template<typename T>
T HyperparameterUtils::sample_from_range(const std::vector<T>& range, std::mt19937& rng) {
    std::uniform_int_distribution<size_t> dist(0, range.size() - 1);
    return range[dist(rng)];
}

HyperparameterConfig HyperparameterTuner::sample_random_config() {
    HyperparameterConfig config;
    
    // Sample architecture parameters
    config.num_layers = HyperparameterUtils::sample_from_range(ranges_.num_layers_range, rng_);
    config.num_heads = HyperparameterUtils::sample_from_range(ranges_.num_heads_range, rng_);
    config.hidden_size = HyperparameterUtils::sample_from_range(ranges_.hidden_size_range, rng_);
    config.intermediate_size = HyperparameterUtils::sample_from_range(ranges_.intermediate_size_range, rng_);
    config.head_dim = HyperparameterUtils::sample_from_range(ranges_.head_dim_range, rng_);
    
    // Sample learning rate parameters
    config.initial_lr = HyperparameterUtils::sample_from_range(ranges_.initial_lr_range, rng_);
    config.peak_lr = HyperparameterUtils::sample_from_range(ranges_.peak_lr_range, rng_);
    config.warmup_steps = HyperparameterUtils::sample_from_range(ranges_.warmup_steps_range, rng_);
    config.decay_factor = HyperparameterUtils::sample_from_range(ranges_.decay_factor_range, rng_);
    
    // Sample training parameters
    config.dropout_rate = HyperparameterUtils::sample_from_range(ranges_.dropout_rate_range, rng_);
    config.weight_decay = HyperparameterUtils::sample_from_range(ranges_.weight_decay_range, rng_);
    config.early_stopping_patience = HyperparameterUtils::sample_from_range(ranges_.early_stopping_patience_range, rng_);
    config.early_stopping_threshold = HyperparameterUtils::sample_from_range(ranges_.early_stopping_threshold_range, rng_);
    config.gradient_clip_threshold = HyperparameterUtils::sample_from_range(ranges_.gradient_clip_threshold_range, rng_);
    config.layer_norm_epsilon = HyperparameterUtils::sample_from_range(ranges_.layer_norm_epsilon_range, rng_);
    
    // Sample memory and optimization parameters
    config.memory_pool_size = HyperparameterUtils::sample_from_range(ranges_.memory_pool_size_range, rng_);
    config.gradient_accumulation_steps = HyperparameterUtils::sample_from_range(ranges_.gradient_accumulation_steps_range, rng_);
    
    return config;
}

bool HyperparameterUtils::is_valid_architecture(const HyperparameterConfig& config) {
    // Check basic size requirements
    if (config.hidden_size % config.num_heads != 0) {
        std::cout << "Invalid architecture: hidden_size must be divisible by num_heads" << std::endl;
        return false;
    }
    
    // Check head dimension compatibility
    if (config.hidden_size / config.num_heads != config.head_dim) {
        std::cout << "Invalid architecture: hidden_size/num_heads must equal head_dim" << std::endl;
        return false;
    }
    
    // Check intermediate size is reasonable multiple of hidden size
    if (config.intermediate_size < config.hidden_size) {
        std::cout << "Invalid architecture: intermediate_size should be >= hidden_size" << std::endl;
        return false;
    }
    
    return true;
}

bool HyperparameterUtils::is_valid_learning_rate(const HyperparameterConfig& config) {
    // Check learning rate ordering
    if (config.peak_lr <= config.initial_lr) {
        std::cout << "Invalid learning rates: peak_lr must be greater than initial_lr" << std::endl;
        return false;
    }
    
    // Check decay factor range
    if (config.decay_factor <= 0.0f || config.decay_factor >= 1.0f) {
        std::cout << "Invalid decay factor: must be between 0 and 1" << std::endl;
        return false;
    }
    
    return true;
}

bool HyperparameterUtils::is_valid_training_params(const HyperparameterConfig& config) {
    // Check probability ranges
    if (config.dropout_rate < 0.0f || config.dropout_rate >= 1.0f) {
        std::cout << "Invalid dropout rate: must be between 0 and 1" << std::endl;
        return false;
    }
    
    // Check positive parameters
    if (config.weight_decay < 0.0f || 
        config.gradient_clip_threshold <= 0.0f || 
        config.layer_norm_epsilon <= 0.0f) {
        std::cout << "Invalid training parameters: must be positive" << std::endl;
        return false;
    }
    
    return true;
}

bool HyperparameterTuner::validate_config(const HyperparameterConfig& config) const {
    return HyperparameterUtils::is_valid_architecture(config) &&
           HyperparameterUtils::is_valid_learning_rate(config) &&
           HyperparameterUtils::is_valid_training_params(config);
}

TuningResult HyperparameterTuner::evaluate_config(
    const HyperparameterConfig& config,
    const std::vector<std::pair<std::string, std::string>>& data,
    const Tokenizer& tokenizer) {
    
    TuningResult result;
    result.config = config;
    
    // Create cross-validation folds
    auto folds = Utils::create_cross_validation_folds(data, num_folds_);
    
    // Evaluate each fold
    float total_loss = 0.0f;
    size_t early_stops = 0;
    
    for (size_t fold = 0; fold < folds.size(); fold++) {
        const auto& [train_data, val_data] = folds[fold];
        
        // Create transformer with current config
        auto transformer_config = config.to_transformer_config();
        Transformer transformer(transformer_config);
        
        // Train and evaluate
        float fold_loss = Utils::perform_cross_validation(
            transformer, tokenizer, train_data, 1, config.early_stopping_threshold);
        
        result.fold_scores.push_back(fold_loss);
        total_loss += fold_loss;
        
        // Check for early stopping
        if (fold_loss > config.early_stopping_threshold) {
            early_stops++;
        }
    }
    
    // Compute statistics
    result.mean_validation_loss = total_loss / num_folds_;
    
    float variance = 0.0f;
    for (float score : result.fold_scores) {
        variance += (score - result.mean_validation_loss) * (score - result.mean_validation_loss);
    }
    result.validation_loss_std = std::sqrt(variance / num_folds_);
    result.early_stops = early_stops;
    
    return result;
}

void HyperparameterTuner::log_trial_progress(size_t current_trial, const TuningResult& result) const {
    std::cout << "\n=== Trial " << current_trial + 1 << "/" << num_trials_ << " ===\n"
              << "Mean validation loss: " << result.mean_validation_loss << "\n"
              << "Validation loss std: " << result.validation_loss_std << "\n"
              << "Early stops: " << result.early_stops << "/" << num_folds_ << std::endl;
    
    HyperparameterUtils::log_config(result.config);
}

void HyperparameterUtils::log_config(const HyperparameterConfig& config) {
    std::cout << "\nConfiguration:"
              << "\nArchitecture:"
              << "\n- Layers: " << config.num_layers
              << "\n- Heads: " << config.num_heads
              << "\n- Hidden size: " << config.hidden_size
              << "\n- Intermediate size: " << config.intermediate_size
              << "\n- Head dimension: " << config.head_dim
              << "\n\nLearning rate:"
              << "\n- Initial: " << config.initial_lr
              << "\n- Peak: " << config.peak_lr
              << "\n- Warmup steps: " << config.warmup_steps
              << "\n- Decay factor: " << config.decay_factor
              << "\n\nTraining parameters:"
              << "\n- Dropout rate: " << config.dropout_rate
              << "\n- Weight decay: " << config.weight_decay
              << "\n- Early stopping patience: " << config.early_stopping_patience
              << "\n- Early stopping threshold: " << config.early_stopping_threshold
              << "\n- Gradient clip threshold: " << config.gradient_clip_threshold
              << "\n- Layer norm epsilon: " << config.layer_norm_epsilon
              << "\n\nOptimization:"
              << "\n- Memory pool size: " << config.memory_pool_size
              << "\n- Gradient accumulation steps: " << config.gradient_accumulation_steps
              << std::endl;
}

void HyperparameterUtils::log_result(const TuningResult& result) {
    std::cout << "\nTrial Results:"
              << "\n- Mean validation loss: " << result.mean_validation_loss
              << "\n- Validation loss std: " << result.validation_loss_std
              << "\n- Early stops: " << result.early_stops
              << "\n\nFold scores:";
    
    for (size_t i = 0; i < result.fold_scores.size(); i++) {
        std::cout << "\n- Fold " << i + 1 << ": " << result.fold_scores[i];
    }
    std::cout << std::endl;
}

std::vector<TuningResult> HyperparameterTuner::tune(
    const std::vector<std::pair<std::string, std::string>>& training_data,
    const Tokenizer& tokenizer) {
    
    results_.clear();  // Clear any previous results
    
    for (size_t trial = 0; trial < num_trials_; ++trial) {
        // Sample a random configuration
        auto config = sample_random_config();
        
        // Skip invalid configurations
        if (!validate_config(config)) {
            std::cout << "Skipping invalid configuration in trial " << trial + 1 << std::endl;
            continue;
        }
        
        // Evaluate the configuration
        auto result = evaluate_config(config, training_data, tokenizer);
        results_.push_back(result);
        
        // Log progress
        log_trial_progress(trial + 1, result);
    }
    
    // Sort results by performance
    std::sort(results_.begin(), results_.end());
    
    return results_;
}

HyperparameterConfig HyperparameterTuner::get_best_config() const {
    if (results_.empty()) {
        throw std::runtime_error("No tuning results available. Run tune() first.");
    }
    
    // Return the config with the best performance (first after sorting)
    return results_[0].config;
}

void HyperparameterTuner::save_results(const std::string& path) const {
    if (results_.empty()) {
        throw std::runtime_error("No results to save. Run tune() first.");
    }
    
    nlohmann::json j;
    j["num_trials"] = num_trials_;
    j["num_folds"] = num_folds_;
    
    // Save all results
    nlohmann::json results_array = nlohmann::json::array();
    for (const auto& result : results_) {
        nlohmann::json result_json;
        
        // Save config
        result_json["config"] = {
            {"num_layers", result.config.num_layers},
            {"num_heads", result.config.num_heads},
            {"hidden_size", result.config.hidden_size},
            {"intermediate_size", result.config.intermediate_size},
            {"head_dim", result.config.head_dim},
            {"initial_lr", result.config.initial_lr},
            {"peak_lr", result.config.peak_lr},
            {"warmup_steps", result.config.warmup_steps},
            {"decay_factor", result.config.decay_factor},
            {"dropout_rate", result.config.dropout_rate},
            {"weight_decay", result.config.weight_decay},
            {"early_stopping_patience", result.config.early_stopping_patience},
            {"early_stopping_threshold", result.config.early_stopping_threshold},
            {"gradient_clip_threshold", result.config.gradient_clip_threshold},
            {"layer_norm_epsilon", result.config.layer_norm_epsilon},
            {"memory_pool_size", result.config.memory_pool_size},
            {"gradient_accumulation_steps", result.config.gradient_accumulation_steps}
        };
        
        // Save metrics
        result_json["mean_validation_loss"] = result.mean_validation_loss;
        result_json["validation_loss_std"] = result.validation_loss_std;
        result_json["early_stops"] = result.early_stops;
        result_json["fold_scores"] = result.fold_scores;
        
        results_array.push_back(result_json);
    }
    
    j["results"] = results_array;
    
    // Save to file
    std::ofstream file(path);
    file << j.dump(4);
} 
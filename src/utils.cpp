#include "../include/utils.hpp"
#include "../include/beam_search.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <queue>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <set>
#include <unordered_set>
#include "../include/data_augmentation.hpp"

// Initialize static members
std::random_device Utils::rd;
std::mt19937 Utils::random_generator;
std::atomic<uint64_t> Utils::prediction_counter(0);

bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && 
           str.compare(0, prefix.size(), prefix) == 0;
}

float Utils::adjust_learning_rate(float current_lr, float loss_ratio, size_t step) {
    const size_t WARMUP_STEPS = 50;
    const float PEAK_LR = 1e-3f;  // Increased from 5e-4f
    const float MIN_LR = 1e-5f;
    const float MAX_RATIO = 1.5f;  // Maximum allowed val/train loss ratio

    // During warmup, increase learning rate linearly
    if (step < WARMUP_STEPS) {
        return MIN_LR + (PEAK_LR - MIN_LR) * (static_cast<float>(step) / WARMUP_STEPS);
    }

    // After warmup, adjust based on validation performance
    const size_t DECAY_STEPS = 2000;  // Reduced from 5000 for faster adaptation
    float progress = static_cast<float>(step - WARMUP_STEPS) / DECAY_STEPS;
    progress = std::min(1.0f, progress);

    // Compute adaptive learning rate based on loss ratio
    float ratio_factor = std::min(loss_ratio, MAX_RATIO) / MAX_RATIO;
    float adaptive_factor = 1.0f - (ratio_factor * progress);
    
    // More aggressive early in training
    if (step < WARMUP_STEPS * 2) {
        adaptive_factor *= 1.2f;  // Boost learning rate in early stages
    }

    float new_lr = PEAK_LR * adaptive_factor;
    new_lr = std::max(MIN_LR, std::min(PEAK_LR, new_lr));

    std::cout << "Learning rate adjustment:"
              << "\n- Step: " << step
              << "\n- Loss ratio: " << loss_ratio
              << "\n- Progress: " << progress
              << "\n- Adaptive factor: " << adaptive_factor
              << "\n- Old LR: " << current_lr
              << "\n- New LR: " << new_lr << std::endl;

    return new_lr;
}

Matrix Utils::create_batch_target_distribution(const std::vector<std::vector<int>>& target_tokens,
                                               const Tokenizer& tokenizer, size_t vocab_size,
                                               size_t input_max_seq_len) {
    // Calculate total size based on input sequence length
    size_t batch_size = target_tokens.size();
    size_t total_tokens = batch_size * input_max_seq_len;
    
    std::cout << "Creating target distribution with dimensions: " << total_tokens << "x" << vocab_size << std::endl;
    
    // Create target distribution for all token positions
    Matrix target_distribution(total_tokens, vocab_size, 0.0f);
    
    // Set target distribution for each token in each sequence
    size_t current_pos = 0;
    for (size_t seq = 0; seq < target_tokens.size(); seq++) {
        const auto& sequence = target_tokens[seq];
        
        // Find the start of the noun phrase at the end
        size_t noun_phrase_start = sequence.size();
        for (size_t i = sequence.size(); i > 0; --i) {
            std::string token = tokenizer.decode({sequence[i-1]});
            if (!tokenizer.is_noun(token)) {
                break;
            }
            noun_phrase_start = i-1;
        }
        
        // Set actual tokens with higher weight for noun phrase tokens
        for (size_t i = 0; i < sequence.size(); i++) {
            // Ensure token ID is within vocab_size bounds
            int token_id = sequence[i];
            if (token_id >= 0 && static_cast<size_t>(token_id) < vocab_size) {  // Use the passed vocab_size
                float weight = (i >= noun_phrase_start) ? 1.0f : 0.5f;
                target_distribution(current_pos, token_id) = weight;
            } else {
                std::cout << "Warning: Token ID " << token_id << " is outside vocabulary size " << vocab_size << std::endl;
            }
            current_pos++;
        }
        
        // Pad remaining positions with pad token
        int pad_token = tokenizer.get_pad_token_id();
        if (pad_token >= 0 && static_cast<size_t>(pad_token) < vocab_size) {  // Use the passed vocab_size
            for (size_t i = sequence.size(); i < input_max_seq_len; i++) {
                target_distribution(current_pos, pad_token) = 1.0f;
                current_pos++;
            }
        }
    }
    
    // Normalize the target distributions
    for (size_t i = 0; i < total_tokens; i++) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < vocab_size; j++) {
            row_sum += target_distribution(i, j);
        }
        if (row_sum > 0.0f) {
            for (size_t j = 0; j < vocab_size; j++) {
                target_distribution(i, j) /= row_sum;
            }
        }
    }
    
    std::cout << "Final target distribution shape: " 
              << target_distribution.rows() << "x" << target_distribution.cols() << std::endl;
    std::cout << "Final current_pos: " << current_pos << "\n";
    std::cout << "=== Target Distribution Creation Complete ===\n\n";
    
    return target_distribution;
}

float Utils::compute_batch_loss(const Matrix& logits, const Matrix& target_distribution, const Tokenizer& tokenizer) {
    // Input validation with detailed error messages
    if (logits.empty() || target_distribution.empty()) {
        std::cout << "Logits shape: " << (logits.empty() ? "empty" : 
                  (std::to_string(logits.rows()) + "x" + std::to_string(logits.cols()))) << std::endl;
        std::cout << "Target distribution shape: " << (target_distribution.empty() ? "empty" : 
                  (std::to_string(target_distribution.rows()) + "x" + std::to_string(target_distribution.cols()))) << std::endl;
        throw std::runtime_error("Empty logits or target distribution in compute_batch_loss");
    }

    // Detailed dimension check
    std::cout << "Logits shape: " << logits.rows() << "x" << logits.cols() << std::endl;
    std::cout << "Target distribution shape: " << target_distribution.rows() << "x" << target_distribution.cols() << std::endl;

    if (logits.rows() != target_distribution.rows() || logits.cols() != target_distribution.cols()) {
        throw std::runtime_error("Dimension mismatch between logits and target distribution");
    }

    float total_loss = 0.0f;
    const size_t batch_size = logits.rows();
    const size_t vocab_size = logits.cols();
    const float epsilon = 1e-7f;  // Increased epsilon for better stability
    const float max_loss_per_token = 100.0f;  // Cap individual token losses

    // Pre-compute max logits for numerical stability
    std::vector<float> max_logits(batch_size, -std::numeric_limits<float>::infinity());
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < vocab_size; ++j) {
            max_logits[i] = std::max(max_logits[i], logits(i, j));
        }
    }

    // Compute loss with improved numerical stability
    #pragma omp parallel for reduction(+:total_loss)
    for (size_t i = 0; i < batch_size; ++i) {
        float sequence_loss = 0.0f;
        float sum_exp = 0.0f;

        // First pass: compute denominator for softmax
        for (size_t j = 0; j < vocab_size; ++j) {
            float shifted_logit = logits(i, j) - max_logits[i];
            sum_exp += std::exp(shifted_logit);
        }
        sum_exp = std::max(sum_exp, epsilon);  // Prevent division by zero

        // Second pass: compute cross-entropy loss
        for (size_t j = 0; j < vocab_size; ++j) {
            if (target_distribution(i, j) > 0.0f) {
                float shifted_logit = logits(i, j) - max_logits[i];
                float log_prob = shifted_logit - std::log(sum_exp);
                
                // Clip log probability for stability
                log_prob = std::max(log_prob, -max_loss_per_token);
                
                float token_loss = -target_distribution(i, j) * log_prob;
                
                // Apply loss scaling and clipping
                token_loss = std::min(token_loss, max_loss_per_token);
                sequence_loss += token_loss;
            }
        }
        
        // Add sequence loss to total with stability check
        if (std::isfinite(sequence_loss)) {
            total_loss += sequence_loss;
        } else {
            std::cout << "Warning: Non-finite sequence loss detected at position " << i << std::endl;
            total_loss += max_loss_per_token;  // Use max loss as fallback
        }
    }

    // Compute average loss with stability checks
    float avg_loss = total_loss / static_cast<float>(batch_size);
    
    // Final stability check
    if (!std::isfinite(avg_loss)) {
        std::cout << "Warning: Non-finite average loss detected. Using fallback value." << std::endl;
        return max_loss_per_token;
    }

    // Add minimum loss floor to prevent vanishing
    const float min_loss = 1e-4f;
    avg_loss = std::max(avg_loss, min_loss);

    std::cout << "Batch loss computation:"
              << "\n- Total loss: " << total_loss
              << "\n- Average loss: " << avg_loss
              << "\n- Batch size: " << batch_size << std::endl;

    return avg_loss;
}

TransformerConfig Utils::load_config(const std::string& config_path) {
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_path);
    }

    nlohmann::json j;
    config_file >> j;

    TransformerConfig config;

    // Load tokenizer config first
    if (j.contains("tokenizer")) {
        const auto& tok = j["tokenizer"];
        config.tokenizer.use_subword = tok.value("use_subword", true);
        // Only set vocab_size if it exists in the config
        if (tok.contains("vocab_size")) {
            size_t vocab_size = tok["vocab_size"].get<size_t>();
            config.tokenizer.vocab_size = vocab_size;
            // Ensure model vocab size matches tokenizer vocab size
            if (j.contains("model")) {
                j["model"]["vocab_size"] = vocab_size;
            }
            std::cout << "Setting vocabulary size from config: " << vocab_size << std::endl;
        }
        config.tokenizer.model_path = tok.value("model_path", "model/tokenizer.model");
    }

    // Parse model settings
    if (j.contains("model")) {
        auto& model = j["model"];
        // Use tokenizer vocab size if available, otherwise use model vocab size
        config.vocab_size = config.tokenizer.vocab_size > 0 ? 
                           config.tokenizer.vocab_size : 
                           model.value("vocab_size", 32000);
        
        std::cout << "Final vocabulary size in config: " << config.vocab_size << std::endl;
        
        // Load other model settings
        config.hidden_size = model["hidden_size"];
        config.num_heads = model["num_heads"];
        config.num_layers = model["num_layers"];
        config.head_dim = model["head_dim"];
        config.intermediate_size = model["intermediate_size"];
    }

    // Parse training settings
    auto& training = j["training"];
    config.batch_size = training["batch_size"];
    config.num_epochs = training["num_epochs"];
    config.dropout_rate = training["dropout_rate"];
    config.weight_decay = training["weight_decay"];
    
    // Parse learning rate settings
    if (j.contains("learning_rate")) {
        auto& lr = j["learning_rate"];
        config.initial_lr = lr.value("initial_lr", 1e-4f);
        config.peak_lr = lr.value("peak_lr", 1e-3f);
        config.warmup_steps = lr.value("warmup_steps", 100);
        config.decay_factor = lr.value("decay_factor", 0.98f);
    }
    
    // Parse early stopping settings
    if (j.contains("early_stopping")) {
        auto& es = j["early_stopping"];
        config.early_stopping_patience = es.value("patience", 3);
        config.early_stopping_threshold = es.value("threshold", 1.5f);
    }
    
    // Parse optimization settings
    if (j.contains("optimization")) {
        auto& opt = j["optimization"];
        config.gradient_clip_threshold = opt.value("gradient_clip_threshold", 5.0f);
        config.layer_norm_epsilon = opt.value("layer_norm_epsilon", 1e-5f);
        config.gradient_accumulation_steps = opt.value("gradient_accumulation_steps", 4);
        config.use_gradient_checkpointing = opt.value("use_gradient_checkpointing", false);
        config.use_fp16 = opt.value("use_fp16", false);
        config.memory_pool_size = opt.value("memory_pool_size", 1024);
    }

    // Parse paths
    if (j.contains("paths")) {
        auto& paths = j["paths"];
        config.paths.save_directory = paths["save_directory"];
        config.paths.model_name = paths["model_name"];
        config.paths.checkpoint_frequency = paths["checkpoint_frequency"];
    }

    // Parse attention settings
    auto& attention = j["attention"];
    config.use_flash_attention = attention["use_flash_attention"];
    config.use_rope = attention["use_rope"];
    config.use_sliding_window = attention["use_sliding_window"];
    config.window_size = attention["window_size"];
    if (attention.contains("use_gqa")) {
        config.use_gqa = attention["use_gqa"].get<bool>();
        if (config.use_gqa) {
            if (attention.contains("num_kv_heads")) {
                config.num_kv_heads = attention["num_kv_heads"].get<size_t>();
            } else {
                config.num_kv_heads = config.num_heads / 2; // Default to half the heads
            }
        } else {
            config.num_kv_heads = config.num_heads;
        }
    }

    // Parse beam search settings
    if (j.contains("beam_search")) {
        auto& beam = j["beam_search"];
        config.beam_search.use_beam_search = beam.value("use_beam_search", true);
        config.beam_search.beam_size = beam["beam_size"];
        config.beam_search.beams_per_group = beam.value("beams_per_group", 4);
        config.beam_search.num_groups = beam.value("num_groups", 3);
        config.beam_search.length_penalty = beam["length_penalty"];
        config.beam_search.temperature = beam["temperature"];
        config.beam_search.top_p = beam["top_p"];
        config.beam_search.max_length = beam["max_length"];
        config.beam_search.initial_temperature = beam.value("initial_temperature", 3.0f);
        config.beam_search.initial_noise_scale = beam.value("initial_noise_scale", 0.8f);
        config.beam_search.diversity_strength = beam.value("diversity_strength", 4.0f);
        config.beam_search.top_k = beam.value("top_k", 100);
        config.beam_search.token_noise_scale = beam.value("token_noise_scale", 0.1f);
    }

    // Parse checkpoint loading settings
    if (j.contains("load_from_checkpoint")) {
        config.load_from_checkpoint = j["load_from_checkpoint"].get<bool>();
        if (config.load_from_checkpoint && j.contains("checkpoint_to_load")) {
            config.checkpoint_to_load = j["checkpoint_to_load"].get<std::string>();
        }
    }

    // Parse token prediction settings
    if (j.contains("token_prediction")) {
        auto& tp = j["token_prediction"];
        config.token_prediction.temperature = tp.value("temperature", 1.0f);
        config.token_prediction.top_k = tp.value("top_k", 5);
        config.token_prediction.top_p = tp.value("top_p", 0.9f);
        config.token_prediction.frequency_penalty = tp.value("frequency_penalty", 0.1f);
        config.token_prediction.presence_penalty = tp.value("presence_penalty", 0.0f);
        config.token_prediction.min_token_prob = tp.value("min_token_prob", 0.05f);
        
        if (tp.contains("category_bonus")) {
            auto& cb = tp["category_bonus"];
            config.token_prediction.category_bonus.verb = cb.value("verb", 0.2f);
            config.token_prediction.category_bonus.adjective = cb.value("adjective", 0.2f);
            config.token_prediction.category_bonus.noun = cb.value("noun", 0.3f);
        }
    }

    return config;
}

std::vector<std::pair<std::string, std::string>> Utils::create_training_data() {
    std::vector<std::pair<std::string, std::string>> training_pairs;
    std::filesystem::path exe_path = std::filesystem::current_path().parent_path();
    std::filesystem::path data_dir = exe_path / "data";
    std::filesystem::path file_path = data_dir / "training_pairs.txt";

    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open training data file: " + file_path.string());
    }

    std::string line;
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> category_pairs;
    std::unordered_set<std::string> seen_pairs;
    
    // First, read all pairs and categorize them
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        // Normalize separators
        std::string normalized_line = line;
        std::replace(normalized_line.begin(), normalized_line.end(), '#', '|');
        std::replace(normalized_line.begin(), normalized_line.end(), '*', '|');
        
        size_t delimiter_pos = normalized_line.find('|');
        if (delimiter_pos != std::string::npos) {
            std::string input = normalized_line.substr(0, delimiter_pos);
            std::string output = normalized_line.substr(delimiter_pos + 1);
            
            // Trim whitespace
            input = std::regex_replace(input, std::regex("^\\s+|\\s+$"), "");
            output = std::regex_replace(output, std::regex("^\\s+|\\s+$"), "");
            
            // Create unique key to detect duplicates
            std::string pair_key = input + "|" + output;
            if (seen_pairs.find(pair_key) != seen_pairs.end()) {
                continue;  // Skip duplicates
            }
            seen_pairs.insert(pair_key);
            
            // Categorize the pair
            std::string category;
            if (input.length() > 50) {
                category = "complex";
            } else if (input.find("is") != std::string::npos || 
                      input.find("looks") != std::string::npos || 
                      input.find("feels") != std::string::npos) {
                category = "adjective";
            } else if (input.find("to") != std::string::npos) {
                category = "verb";
            } else {
                category = "other";
            }
            
            category_pairs[category].push_back({input, output});
        }
    }
    
    // Print initial statistics
    std::cout << "\nInitial Training Data Statistics:" << std::endl;
    size_t total_pairs = 0;
    size_t max_category_size = 0;
    for (const auto& [category, pairs] : category_pairs) {
        std::cout << category << " pairs: " << pairs.size() << std::endl;
        total_pairs += pairs.size();
        max_category_size = std::max(max_category_size, pairs.size());
    }
    std::cout << "Total pairs before balancing: " << total_pairs << std::endl;

    // Instead of taking the minimum, we'll take up to 80% of the largest category size
    size_t target_size = static_cast<size_t>(max_category_size * 0.8);
    std::cout << "Target size per category: " << target_size << std::endl;

    // Random number generator for shuffling
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    
    // Sample from each category
    for (const auto& [category, pairs] : category_pairs) {
        std::vector<size_t> indices(pairs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Take up to target_size samples from each category
        size_t samples_to_take = std::min(target_size, pairs.size());
        for (size_t i = 0; i < samples_to_take; ++i) {
            training_pairs.push_back(pairs[indices[i]]);
        }
    }
    
    // Shuffle the final training pairs
    std::shuffle(training_pairs.begin(), training_pairs.end(), gen);
    
    std::cout << "\nFinal Training Data Statistics:" << std::endl;
    std::cout << "Total pairs after balancing: " << training_pairs.size() << std::endl;
    
    return training_pairs;
}

void Utils::analyze_token_mappings(
    const std::vector<std::pair<std::string, std::string>>& training_data,
    const Tokenizer& tokenizer) {
    std::cout << "\n=== Analyzing Token Mappings ===\n";
    size_t total_words = 0;
    size_t unknown_tokens = 0;
    std::unordered_map<std::string, int> unknown_words;

    for (const auto& pair : training_data) {
        std::string processed_input = pair.first;
        tokenizer.preprocess_text(processed_input);
        std::vector<int> tokens = tokenizer.encode(processed_input);

        for (int token : tokens) {
            if (!tokenizer.is_special_token(token)) {
                total_words++;
                if (tokenizer.decode({token}) == " ") {
                    unknown_tokens++;
                    unknown_words[tokenizer.decode({token})]++;
                }
            }
        }
    }

    std::cout << "Token Mapping Statistics:\n"
              << "Total words: " << total_words << "\n"
              << "Unknown tokens: " << unknown_tokens << " ("
              << (100.0f * unknown_tokens / total_words) << "%)\n";
}

std::vector<std::pair<std::string, std::string>> Utils::load_validation_data() {
    std::vector<std::pair<std::string, std::string>> validation_pairs;
    std::filesystem::path exe_path = std::filesystem::current_path().parent_path();
    std::filesystem::path data_dir = exe_path / "data";
    std::filesystem::path file_path = data_dir / "validation_pairs.txt";

    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open validation data file: " + file_path.string());
    }

    std::string line;
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> category_pairs;
    std::unordered_set<std::string> seen_pairs;
    
    while (std::getline(file, line)) {
        // Normalize separators
        std::string normalized_line = line;
        std::replace(normalized_line.begin(), normalized_line.end(), '#', '|');
        std::replace(normalized_line.begin(), normalized_line.end(), '*', '|');
        
        size_t delimiter_pos = normalized_line.find('|');
        if (delimiter_pos != std::string::npos) {
            std::string input = normalized_line.substr(0, delimiter_pos);
            std::string output = normalized_line.substr(delimiter_pos + 1);
            
            // Trim whitespace
            input = std::regex_replace(input, std::regex("^\\s+|\\s+$"), "");
            output = std::regex_replace(output, std::regex("^\\s+|\\s+$"), "");
            
            // Create unique key to detect duplicates
            std::string pair_key = input + "|" + output;
            if (seen_pairs.find(pair_key) != seen_pairs.end()) {
                continue;  // Skip duplicates
            }
            seen_pairs.insert(pair_key);
            
            // Categorize the pair
            std::string category;
            if (input.length() > 50) {
                category = "complex";
            } else if (input.find("is") != std::string::npos || 
                      input.find("looks") != std::string::npos || 
                      input.find("feels") != std::string::npos) {
                category = "adjective";
            } else if (input.find("to") != std::string::npos) {
                category = "verb";
            } else {
                category = "other";
            }
            
            category_pairs[category].push_back({input, output});
        }
    }
    
    // Balance categories but keep more validation samples
    size_t min_category_size = std::numeric_limits<size_t>::max();
    for (const auto& [category, pairs] : category_pairs) {
        min_category_size = std::min(min_category_size, pairs.size());
    }
    
    // Use up to 20% of training size for validation
    min_category_size = std::min(min_category_size, static_cast<size_t>(min_category_size * 0.2));
    
    // Sample evenly from each category
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    for (const auto& [category, pairs] : category_pairs) {
        std::vector<size_t> indices(pairs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        
        // Take min_category_size samples from each category
        for (size_t i = 0; i < min_category_size; ++i) {
            validation_pairs.push_back(pairs[indices[i]]);
        }
    }
    
    std::cout << "\nValidation Data Statistics:" << std::endl;
    std::cout << "Total pairs after balancing: " << validation_pairs.size() << std::endl;
    for (const auto& [category, pairs] : category_pairs) {
        std::cout << category << " pairs: " << pairs.size() 
                  << " (used " << min_category_size << ")" << std::endl;
    }
    
    return validation_pairs;
}

bool Utils::validate_input_sequence(const std::vector<int>& tokens, size_t vocab_size,
                                    size_t max_seq_length) {
    // For target sequences, we allow empty sequences
    if (tokens.empty()) {
        return true;  // Empty sequences are valid for targets
    }

    // For non-empty sequences, check length if max_seq_length is specified
    if (max_seq_length > 0 && tokens.size() > max_seq_length) {
        std::cout << "Invalid sequence: too long (length: " << tokens.size() 
                  << ", max: " << max_seq_length << ")" << std::endl;
        return false;
    }

    // Validate each token
    for (int token : tokens) {
        if (token < 0 || static_cast<size_t>(token) >= vocab_size) {
            std::cout << "Invalid token " << token << " (vocab size: " << vocab_size << ")" << std::endl;
            return false;
        }
    }
    return true;
}

void Utils::print_matrix(const Matrix& m, const std::string& name, size_t max_rows,
                         size_t max_cols) {
    std::cout << "\n" << name << " (" << m.rows() << "x" << m.cols() << "):\n";
    for (size_t i = 0; i < std::min(max_rows, m.rows()); ++i) {
        for (size_t j = 0; j < std::min(max_cols, m.cols()); ++j) {
            std::cout << std::fixed << std::setprecision(4) << m(i, j) << " ";
        }
        std::cout << (m.cols() > max_cols ? "..." : "") << "\n";
    }
    if (m.rows() > max_rows) {
        std::cout << "...\n";
    }
}

// Helper function to get multi-token predictions
std::vector<std::pair<std::string, float>> Utils::get_multi_token_predictions(
    const Matrix& logits, const Tokenizer& tokenizer, int beam_width) {
    
    const int last_pos = logits.rows() - 1;
    std::vector<std::pair<std::string, float>> predictions;
    
    // Get top tokens and their probabilities
    std::vector<std::pair<float, int>> token_probs;
    for (int j = 0; j < logits.cols(); j++) {
        token_probs.push_back({logits(last_pos, j), j});
    }
    
    // Sort by probability
    std::sort(token_probs.begin(), token_probs.end(), std::greater<>());
    
    // Take top beam_width tokens
    for (int i = 0; i < std::min(beam_width, static_cast<int>(token_probs.size())); i++) {
        int token_id = token_probs[i].second;
        float prob = token_probs[i].first;
        
        // Skip special tokens
        if (tokenizer.is_special_token(token_id)) continue;
        
        // Decode token
        std::vector<int> token_seq = {token_id};
        std::string decoded = tokenizer.decode(token_seq);
        
        if (!decoded.empty()) {
            predictions.push_back({decoded, prob});
        }
    }
    
    return predictions;
}

TokenCategories Utils::analyze_token_categories(const std::vector<std::pair<std::string, std::string>>& training_data) {
    TokenCategories categories;
    
    for (const auto& [input, target] : training_data) {
        size_t sep_pos;
        if ((sep_pos = target.find('#')) != std::string::npos) {
            // This is a verb ending
            std::string verb = target.substr(sep_pos + 1);
            Utils::trim(verb);
            categories.verb_tokens.insert(verb);
        } else if ((sep_pos = target.find('*')) != std::string::npos) {
            // This is an adjective ending
            std::string adj = target.substr(sep_pos + 1);
            Utils::trim(adj);
            categories.adjective_tokens.insert(adj);
        } else if ((sep_pos = target.find('|')) != std::string::npos) {
            // This is a noun ending
            std::string noun = target.substr(sep_pos + 1);
            Utils::trim(noun);
            categories.noun_tokens.insert(noun);
        }
    }
    
    std::cout << "\nToken Category Analysis:\n";
    std::cout << "Unique Verbs: " << categories.verb_tokens.size() << "\n";
    std::cout << "Unique Adjectives: " << categories.adjective_tokens.size() << "\n";
    std::cout << "Unique Nouns: " << categories.noun_tokens.size() << "\n";
    
    return categories;
}

// Function to determine the category of a token
std::string Utils::get_token_category(const std::string& token, const TokenCategories& categories) {
    if (categories.verb_tokens.find(token) != categories.verb_tokens.end()) {
        return "VERB";
    } else if (categories.adjective_tokens.find(token) != categories.adjective_tokens.end()) {
        return "ADJ";
    } else if (categories.noun_tokens.find(token) != categories.noun_tokens.end()) {
        return "NOUN";
    }
    return "UNKNOWN";
}

// Modify the print_top_predictions function to show token categories
void Utils::print_top_predictions(const Matrix& logits, const Tokenizer& tokenizer, 
                                Transformer& transformer, int k) {
    const auto& config = transformer.getConfig();
    const auto& tp_config = config.token_prediction;

    // Get the last row of logits (predictions for the next token)
    std::vector<float> last_logits;
    for (size_t i = 0; i < logits.cols(); i++) {
        last_logits.push_back(logits(logits.rows() - 1, i));
    }

    // Apply temperature scaling
    float temperature = tp_config.temperature;
    for (auto& logit : last_logits) {
        logit /= temperature;
    }

    // Apply softmax to get probabilities
    float max_logit = *std::max_element(last_logits.begin(), last_logits.end());
    std::vector<float> probabilities(last_logits.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < last_logits.size(); i++) {
        probabilities[i] = std::exp(last_logits[i] - max_logit);
        sum_exp += probabilities[i];
    }
    for (float& prob : probabilities) {
        prob /= sum_exp;
    }

    // Apply frequency penalty
    const auto& frequencies = transformer.get_lm_head()->get_token_frequencies();
    if (!frequencies.empty()) {
        float max_freq = *std::max_element(frequencies.begin(), frequencies.end());
        for (size_t i = 0; i < probabilities.size(); i++) {
            if (i < frequencies.size()) {
                float penalty = tp_config.frequency_penalty * (frequencies[i] / max_freq);
                probabilities[i] *= (1.0f - penalty);
            }
        }
    }

    // Apply category bonuses
    for (size_t i = 0; i < probabilities.size(); i++) {
        std::string token = tokenizer.decode({static_cast<int>(i)});
        if (tokenizer.is_verb(token)) {
            probabilities[i] *= (1.0f + tp_config.category_bonus.verb);
        } else if (tokenizer.is_adjective(token)) {
            probabilities[i] *= (1.0f + tp_config.category_bonus.adjective);
        } else if (tokenizer.is_noun(token)) {
            probabilities[i] *= (1.0f + tp_config.category_bonus.noun);
        }
    }

    // Re-normalize probabilities
    sum_exp = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
    for (float& prob : probabilities) {
        prob /= sum_exp;
    }

    // Apply top-p (nucleus) sampling
    if (tp_config.top_p < 1.0f) {
        std::vector<std::pair<float, size_t>> prob_idx;
        for (size_t i = 0; i < probabilities.size(); i++) {
            prob_idx.push_back({probabilities[i], i});
        }
        std::sort(prob_idx.begin(), prob_idx.end(), std::greater<>());

        float cumsum = 0.0f;
        std::vector<size_t> nucleus_indices;
        for (const auto& [prob, idx] : prob_idx) {
            if (cumsum >= tp_config.top_p) break;
            nucleus_indices.push_back(idx);
            cumsum += prob;
        }

        // Zero out probabilities outside the nucleus
        std::vector<bool> in_nucleus(probabilities.size(), false);
        for (size_t idx : nucleus_indices) {
            in_nucleus[idx] = true;
        }
        for (size_t i = 0; i < probabilities.size(); i++) {
            if (!in_nucleus[i]) {
                probabilities[i] = 0.0f;
            }
        }
    }

    // Apply minimum probability threshold
    for (float& prob : probabilities) {
        if (prob < tp_config.min_token_prob) {
            prob = 0.0f;
        }
    }

    // Create index vector for top-k selection
    std::vector<size_t> indices(probabilities.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by probability
    std::partial_sort(indices.begin(), 
                     indices.begin() + std::min(k, static_cast<int>(indices.size())),
                     indices.end(),
                     [&probabilities](size_t a, size_t b) {
                         return probabilities[a] > probabilities[b];
                     });

    // Print top k predictions
    std::cout << "\nTop " << k << " predictions:" << std::endl;
    for (int i = 0; i < k && i < static_cast<int>(indices.size()); i++) {
        size_t idx = indices[i];
        if (probabilities[idx] > 0.0f) {  // Only show non-zero probability tokens
            std::string token = tokenizer.decode({static_cast<int>(idx)});
            std::string category = "";
            if (tokenizer.is_verb(token)) category = " (VERB)";
            else if (tokenizer.is_adjective(token)) category = " (ADJ)";
            else if (tokenizer.is_noun(token)) category = " (NOUN)";
            
            std::cout << i + 1 << ". \"" << token << "\"" << category << " (p=" 
                      << std::fixed << std::setprecision(4) << probabilities[idx] << ")" << std::endl;
        }
    }
}

float Utils::evaluate_validation(
    Transformer& transformer, const Tokenizer& tokenizer,
    const std::vector<std::pair<std::string, std::string>>& validation_data) {
    std::cout << "\n=== Evaluating Validation Data ===\n";

    float total_loss = 0.0f;
    size_t correct_predictions = 0;
    size_t total_predictions = 0;

    // Get token prediction config
    const auto& config = transformer.getConfig();
    const auto& tp_config = config.token_prediction;

    // Validate we have data to process
    if (validation_data.empty()) {
        std::cout << "Warning: Empty validation data\n";
        return 0.0f;
    }

    transformer.set_training(false); // Set model to evaluation mode

    for (const auto& pair : validation_data) {
        try {
            // Preprocess input
            std::string processed_input = pair.first;
            std::cout << "Processing input: '" << processed_input << "'\n";
            tokenizer.preprocess_text(processed_input);
            std::vector<int> input_tokens = tokenizer.encode(processed_input);
            
            // Forward pass to get hidden states (this should return hidden states, not logits)
            Matrix hidden_states = transformer.forward(input_tokens, processed_input, tokenizer);
            std::cout << "Hidden states dimensions: " << hidden_states.rows() << "x" << hidden_states.cols() << std::endl;
            
            // Project through language model head to get logits
            auto lm_head = transformer.get_lm_head();
            if (!lm_head) {
                throw std::runtime_error("Language model head is not initialized");
            }
            Matrix logits = lm_head->forward(hidden_states);
            std::cout << "Logits dimensions: " << logits.rows() << "x" << logits.cols() << std::endl;
            
            // Get the last token's logits
            Vector last_logits = logits.row(logits.rows() - 1);
            
            // Get predicted token
            int predicted_token = 0;
            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < last_logits.size(); i++) {
                if (last_logits[i] > max_logit) {
                    max_logit = last_logits[i];
                    predicted_token = i;
                }
            }

            // Get target
            std::string processed_target = pair.second;
            tokenizer.preprocess_text(processed_target);
            std::vector<int> target_tokens = tokenizer.encode(processed_target);

            // Create target distribution
            Matrix target_distribution(1, tokenizer.vocab_size(), 0.0f);
            if (!target_tokens.empty()) {
                target_distribution(0, target_tokens.back()) = 1.0f;
            }

            // Update logits matrix with modified values
            Matrix last_token_logits(1, logits.cols());
            for (size_t i = 0; i < last_logits.size(); i++) {
                last_token_logits(0, i) = last_logits[i];
            }

            // Verify dimensions before computing loss
            if (last_token_logits.cols() != target_distribution.cols()) {
                throw std::runtime_error("Dimension mismatch: logits cols (" + 
                    std::to_string(last_token_logits.cols()) + ") != target cols (" + 
                    std::to_string(target_distribution.cols()) + ")");
            }

            float loss = compute_batch_loss(last_token_logits, target_distribution, tokenizer);
            total_loss += loss;

            // Check if prediction matches target
            if (!target_tokens.empty() && predicted_token == target_tokens.back()) {
                correct_predictions++;
            }
            total_predictions++;

        } catch (const std::exception& e) {
            std::cout << "Error evaluating validation: " << e.what() << "\n";
            continue;  // Skip this example and continue with the next
        }
    }

    // Print evaluation metrics
    if (total_predictions > 0) {
        float accuracy = static_cast<float>(correct_predictions) / total_predictions;
        float avg_loss = total_loss / total_predictions;
        std::cout << "\nValidation Results:\n"
                  << "Average Loss: " << avg_loss << "\n"
                  << "Accuracy: " << (accuracy * 100.0f) << "%\n"
                  << "Correct Predictions: " << correct_predictions << "/" << total_predictions << "\n";
    }

    transformer.set_training(true); // Reset to training mode
    return total_predictions > 0 ? total_loss / total_predictions : 0.0f;
}

void Utils::apply_sampling_parameters(std::vector<float>& logits, float temperature, float top_p) {
    // Apply temperature scaling first
    if (temperature != 1.0f) {
        for (auto& logit : logits) {
            logit /= temperature;
        }
    }

    // Apply top-p (nucleus) sampling if enabled
    if (top_p < 1.0f) {
        // Convert logits to probabilities
        std::vector<std::pair<float, size_t>> probs_with_indices;
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;

        for (size_t i = 0; i < logits.size(); i++) {
            float prob = std::exp(logits[i] - max_logit);
            sum_exp += prob;
            probs_with_indices.push_back({prob, i});
        }

        // Normalize probabilities
        for (auto& pair : probs_with_indices) {
            pair.first /= sum_exp;
        }

        // Sort by probability in descending order
        std::sort(probs_with_indices.begin(), probs_with_indices.end(),
                  std::greater<std::pair<float, size_t>>());

        // Find cutoff index for top-p
        float cumsum = 0.0f;
        size_t cutoff_idx = probs_with_indices.size() - 1;
        for (size_t i = 0; i < probs_with_indices.size(); i++) {
            cumsum += probs_with_indices[i].first;
            if (cumsum > top_p) {
                cutoff_idx = i;
                break;
            }
        }

        // Create mask for filtered tokens
        std::vector<bool> keep_token(logits.size(), false);
        for (size_t i = 0; i <= cutoff_idx; i++) {
            keep_token[probs_with_indices[i].second] = true;
        }

        // Apply mask to logits
        for (size_t i = 0; i < logits.size(); i++) {
            if (!keep_token[i]) {
                logits[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }
}

std::vector<std::string>& Utils::get_vocabulary(const Tokenizer& tokenizer) {
    static std::vector<std::string> vocabulary;
    if (vocabulary.empty()) {
        vocabulary.reserve(tokenizer.vocab_size());
        
        // Fill vocabulary with all possible token strings
        for (size_t i = 0; i < tokenizer.vocab_size(); i++) {
            vocabulary.push_back(tokenizer.decode({static_cast<int>(i)}));
        }
        std::cout << "Loaded vocabulary with " << vocabulary.size() << " tokens" << std::endl;
    }
    return vocabulary;
}

std::vector<size_t> topKSampling(const std::vector<float>& probabilities, size_t k) {
    std::vector<std::pair<float, size_t>> prob_idx;
    for (size_t i = 0; i < probabilities.size(); i++) {
        prob_idx.push_back({probabilities[i], i});
    }
    
    // Sort by probability in descending order
    std::partial_sort(prob_idx.begin(), prob_idx.begin() + k, prob_idx.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Return top k indices
    std::vector<size_t> result;
    for (size_t i = 0; i < k; i++) {
        result.push_back(prob_idx[i].second);
    }
    return result;
}

std::vector<size_t> nucleusSampling(const std::vector<float>& probabilities, float p) {
    std::vector<std::pair<float, size_t>> sorted_probs;
    for (size_t i = 0; i < probabilities.size(); i++) {
        sorted_probs.push_back({probabilities[i], i});
    }
    
    std::sort(sorted_probs.begin(), sorted_probs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    float cumsum = 0.0f;
    std::vector<size_t> result;
    for (const auto& pair : sorted_probs) {
        cumsum += pair.first;
        result.push_back(pair.second);
        if (cumsum >= p) break;
    }
    return result;
}

void from_json(const nlohmann::json& j, TokenizerConfig& t) {
    if (j.contains("use_subword")) {
        t.use_subword = j["use_subword"].get<bool>();
    }
    if (j.contains("vocab_size")) {
        t.vocab_size = j["vocab_size"].get<size_t>();
    }
    if (j.contains("model_path")) {
        t.model_path = j["model_path"].get<std::string>();
    }
    if (j.contains("special_tokens")) {
        t.special_tokens = j["special_tokens"].get<std::vector<std::string>>();
    }
}

void to_json(nlohmann::json& j, const TokenizerConfig& t) {
    j = nlohmann::json{
        {"use_subword", t.use_subword},
        {"vocab_size", t.vocab_size},
        {"model_path", t.model_path},
        {"special_tokens", t.special_tokens}
    };
}

void Utils::trim(std::string& s) {
    // Trim from start
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));

    // Trim from end
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// Add new cross-validation function
std::vector<std::pair<std::vector<std::pair<std::string, std::string>>, 
                     std::vector<std::pair<std::string, std::string>>>> 
Utils::create_cross_validation_folds(const std::vector<std::pair<std::string, std::string>>& data, size_t num_folds) {
    std::vector<std::pair<std::vector<std::pair<std::string, std::string>>,
                         std::vector<std::pair<std::string, std::string>>>> folds;
    
    // Create a copy of the data that we can shuffle
    std::vector<std::pair<std::string, std::string>> shuffled_data = data;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), g);
    
    // Calculate fold size
    size_t fold_size = shuffled_data.size() / num_folds;
    
    // Create folds
    for (size_t i = 0; i < num_folds; i++) {
        std::vector<std::pair<std::string, std::string>> validation_fold;
        std::vector<std::pair<std::string, std::string>> training_fold;
        
        // Calculate start and end indices for current validation fold
        size_t start_idx = i * fold_size;
        size_t end_idx = (i == num_folds - 1) ? shuffled_data.size() : (i + 1) * fold_size;
        
        // Split data into training and validation
        for (size_t j = 0; j < shuffled_data.size(); j++) {
            if (j >= start_idx && j < end_idx) {
                validation_fold.push_back(shuffled_data[j]);
            } else {
                training_fold.push_back(shuffled_data[j]);
            }
        }
        
        folds.push_back({training_fold, validation_fold});
    }
    
    return folds;
}

float Utils::perform_cross_validation(Transformer& transformer, const Tokenizer& tokenizer,
                                    const std::vector<std::pair<std::string, std::string>>& data,
                                    size_t num_folds, float early_stopping_threshold) {
    std::cout << "\n=== Starting " << num_folds << "-fold Cross Validation ===\n";
    std::cout << "Total data size: " << data.size() << " pairs\n";
    
    auto folds = create_cross_validation_folds(data, num_folds);
    std::vector<float> fold_scores;
    float total_val_loss = 0.0f;
    size_t early_stops = 0;
    
    // Get configuration parameters
    const auto& config = transformer.getConfig();
    const float initial_lr = config.initial_lr;
    const float peak_lr = config.peak_lr;
    const size_t warmup_steps = config.warmup_steps;
    const float decay_factor = config.decay_factor;
    const size_t patience = config.early_stopping_patience;
    const float gradient_clip = config.gradient_clip_threshold;
    const size_t grad_accum_steps = config.gradient_accumulation_steps;
    
    float current_lr = initial_lr;
    
    // Store the current gradients and tokens for accumulation
    std::vector<int> current_input_tokens;
    Matrix current_loss_gradients;
    
    for (size_t fold = 0; fold < folds.size(); fold++) {
        std::cout << "\n>>> Processing Fold " << fold + 1 << "/" << num_folds << std::endl;
        
        const auto& [train_data, val_data] = folds[fold];
        std::cout << "Train data size: " << train_data.size() << ", Val data size: " << val_data.size() << std::endl;
        
        // Track validation loss history for early stopping
        std::deque<float> val_loss_history;
        float best_val_loss = std::numeric_limits<float>::max();
        size_t no_improvement_count = 0;
        size_t global_step = 0;
        
        // Gradient accumulation variables
        size_t accum_step = 0;
        float accumulated_loss = 0.0f;
        
        // Training loop for this fold
        for (size_t epoch = 0; epoch < config.num_epochs; epoch++) {
            std::cout << "\n>> Starting Epoch " << epoch + 1 << "/" << config.num_epochs << std::endl;
            
            float train_loss = 0.0f;
            size_t processed_examples = 0;
            
            std::cout << "Processing training data..." << std::endl;
            for (const auto& pair : train_data) {
                try {
                    std::cout << "\rProcessing example " << processed_examples + 1 << "/" << train_data.size() 
                              << " (Fold " << fold + 1 << "/" << num_folds 
                              << ", Epoch " << epoch + 1 << "/" << config.num_epochs << ")" << std::flush;
                    
                    // Forward pass
                    current_input_tokens = tokenizer.encode(pair.first);
                    Matrix output = transformer.forward(current_input_tokens, "", tokenizer);
                    Matrix target_distribution = create_batch_target_distribution(
                        {tokenizer.encode(pair.second)}, tokenizer, tokenizer.vocab_size(), current_input_tokens.size());
                    
                    // Compute loss and gradients
                    float example_loss = compute_batch_loss(output, target_distribution, tokenizer);
                    current_loss_gradients = compute_loss_gradient(output, target_distribution);
                    
                    // Apply gradient clipping
                    float grad_norm = 0.0f;
                    for (size_t i = 0; i < current_loss_gradients.rows(); ++i) {
                        for (size_t j = 0; j < current_loss_gradients.cols(); ++j) {
                            grad_norm += current_loss_gradients(i, j) * current_loss_gradients(i, j);
                        }
                    }
                    grad_norm = std::sqrt(grad_norm);
                    
                    if (grad_norm > gradient_clip) {
                        float scale = gradient_clip / grad_norm;
                        for (size_t i = 0; i < current_loss_gradients.rows(); ++i) {
                            for (size_t j = 0; j < current_loss_gradients.cols(); ++j) {
                                current_loss_gradients(i, j) *= scale;
                            }
                        }
                    }
                    
                    // Update learning rate with warmup and decay
                    if (global_step < warmup_steps) {
                        current_lr = initial_lr + (peak_lr - initial_lr) * (float)global_step / warmup_steps;
                    } else {
                        current_lr *= decay_factor;
                    }
                    
                    // Gradient accumulation
                    accumulated_loss += example_loss;
                    accum_step++;
                    
                    // Only update parameters after accumulating enough gradients
                    if (accum_step >= grad_accum_steps) {
                        // Backward pass and parameter update with accumulated gradients
                        transformer.backward(current_loss_gradients, current_input_tokens, current_lr);
                        transformer.update_parameters(current_lr);
                        
                        // Reset accumulation
                        accumulated_loss = 0.0f;
                        accum_step = 0;
                        current_loss_gradients = Matrix();  // Clear gradients
                        current_input_tokens.clear();       // Clear tokens
                    }
                    
                    train_loss += example_loss;
                    processed_examples++;
                    global_step++;
                    
                    if (processed_examples % 10 == 0) {
                        std::cout << "\nBatch " << processed_examples/10 << " stats:"
                                 << "\n- Current example: " << processed_examples << "/" << train_data.size()
                                 << "\n- Fold: " << fold + 1 << "/" << num_folds
                                 << "\n- Epoch: " << epoch + 1 << "/" << config.num_epochs
                                 << "\n- Average loss: " << (train_loss / processed_examples)
                                 << "\n- Learning rate: " << current_lr
                                 << "\n- Example loss: " << example_loss
                                 << "\n- Gradient norm: " << grad_norm << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "\nError processing example " << processed_examples + 1 
                              << " in fold " << fold + 1 << ", epoch " << epoch + 1 
                              << ": " << e.what() << std::endl;
                    throw;
                }
            }
            
            // Process any remaining accumulated gradients
            if (accum_step > 0 && !current_input_tokens.empty()) {
                transformer.backward(current_loss_gradients, current_input_tokens, current_lr);
                transformer.update_parameters(current_lr);
            }
            
            train_loss /= train_data.size();
            std::cout << "\nEpoch " << epoch + 1 << " training complete. Average train loss: " << train_loss << std::endl;
            
            // Evaluate on validation data
            float val_loss = evaluate_validation(transformer, tokenizer, val_data);
            val_loss_history.push_back(val_loss);
            if (val_loss_history.size() > patience) {
                val_loss_history.pop_front();
            }
            
            // Early stopping check
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                no_improvement_count = 0;
            } else {
                no_improvement_count++;
                if (no_improvement_count >= patience) {
                    std::cout << "Early stopping triggered on fold " << fold + 1 << std::endl;
                    early_stops++;
                    break;
                }
            }
            
            // Check for overfitting using configured threshold
            float loss_ratio = val_loss / train_loss;
            if (loss_ratio > config.early_stopping_threshold) {
                std::cout << "Overfitting detected (val/train ratio: " << loss_ratio 
                          << " > " << config.early_stopping_threshold << ")" << std::endl;
                early_stops++;
                break;
            }
        }
        
        fold_scores.push_back(best_val_loss);
        total_val_loss += best_val_loss;
        std::cout << "\nCompleted Fold " << fold + 1 << "/" << num_folds 
                  << "\n- Best validation loss: " << best_val_loss 
                  << "\n- Early stops: " << early_stops << std::endl;
    }
    
    // Compute and print statistics
    float mean_val_loss = total_val_loss / num_folds;
    float variance = 0.0f;
    for (float score : fold_scores) {
        variance += (score - mean_val_loss) * (score - mean_val_loss);
    }
    variance /= num_folds;
    
    std::cout << "\nCross Validation Results:" << std::endl;
    std::cout << "Mean Validation Loss: " << mean_val_loss << std::endl;
    std::cout << "Standard Deviation: " << std::sqrt(variance) << std::endl;
    std::cout << "Early Stops: " << early_stops << "/" << num_folds << std::endl;
    
    return mean_val_loss;
}

void Utils::generate_predictions(Transformer& transformer, const std::string& input_text, Tokenizer* tokenizer) {
    std::cout << "\n=== Generating predictions for: '" << input_text << "' ===" << std::endl;
    
    // Preprocess input
    std::string processed_input = input_text;
    tokenizer->preprocess_text(processed_input);
    std::vector<int> input_tokens = tokenizer->encode(processed_input);
    
    // Get model prediction
    transformer.set_training(false);  // Set to evaluation mode
    Matrix hidden_states = transformer.forward(input_tokens, processed_input, *tokenizer);
    Matrix logits = transformer.get_lm_head()->forward(hidden_states);
    
    // Get probabilities for last token
    Vector last_logits = logits.row(logits.rows() - 1);
    std::vector<std::pair<float, int>> token_probs;
    
    // Convert logits to probabilities using softmax
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < last_logits.size(); i++) {
        max_logit = std::max(max_logit, last_logits[i]);
    }
    
    float sum_exp = 0.0f;
    for (size_t i = 0; i < last_logits.size(); i++) {
        float prob = std::exp(last_logits[i] - max_logit);
        sum_exp += prob;
        token_probs.push_back({prob, static_cast<int>(i)});
    }
    
    // Normalize probabilities
    for (auto& pair : token_probs) {
        pair.first /= sum_exp;
    }
    
    // Sort by probability
    std::sort(token_probs.begin(), token_probs.end(),
              std::greater<std::pair<float, int>>());
    
    // Print top 5 predictions
    std::cout << "Top 5 predictions:" << std::endl;
    for (int i = 0; i < std::min(5, static_cast<int>(token_probs.size())); i++) {
        float prob = token_probs[i].first;
        int token_id = token_probs[i].second;
        std::string token = tokenizer->decode({token_id});
        
        // Add token type annotation
        std::string token_type = "";
        if (tokenizer->is_verb(token)) token_type = " (VERB)";
        else if (tokenizer->is_adjective(token)) token_type = " (ADJ)";
        else if (tokenizer->is_noun(token)) token_type = " (NOUN)";
        
        std::cout << i + 1 << ". \"" << token << "\"" << token_type 
                  << " (p=" << std::fixed << std::setprecision(4) << prob * 100 << "%)" << std::endl;
    }
    
    std::cout << "===" << std::endl;
    transformer.set_training(true);  // Reset to training mode
}
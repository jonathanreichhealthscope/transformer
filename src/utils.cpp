#include "../include/utils.hpp"
#include <random>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <iomanip>
#include <iostream>

float Utils::adjust_learning_rate(float current_lr, float loss_ratio, size_t step) {
    const size_t WARMUP_STEPS = 50;
    const float PEAK_LR = 5e-4;
    const float MIN_LR = 1e-5;
    
    if (step < WARMUP_STEPS) {
        return MIN_LR + (PEAK_LR - MIN_LR) * (static_cast<float>(step) / WARMUP_STEPS);
    }
    
    const size_t DECAY_STEPS = 5000;
    float progress = static_cast<float>(step - WARMUP_STEPS) / DECAY_STEPS;
    progress = std::min(1.0f, progress);
    
    float decay_factor = 0.5f * (1.0f + std::cos(progress * M_PI));
    float lr = MIN_LR + (PEAK_LR - MIN_LR) * decay_factor;
    
    const float LOSS_SPIKE_THRESHOLD = 1.5f;
    if (loss_ratio > LOSS_SPIKE_THRESHOLD) {
        lr *= 0.1f;
    }
    
    return std::clamp(lr, MIN_LR, PEAK_LR);
}

Matrix Utils::create_batch_target_distribution(const std::vector<std::vector<int>>& target_tokens,
                                              const Tokenizer& tokenizer, size_t vocab_size) {
    Matrix target_distribution(target_tokens.size(), vocab_size, 0.0f);
    for (size_t i = 0; i < target_tokens.size(); i++) {
        if (!target_tokens[i].empty()) {
            target_distribution(i, target_tokens[i].back()) = 1.0f;
        }
    }
    return target_distribution;
}

float Utils::compute_batch_loss(const Matrix& logits, const Matrix& target_distribution) {
    float loss = 0.0f;
    const float epsilon = 1e-10f;
    
    for (size_t i = 0; i < logits.rows(); i++) {
        // Find max logit for numerical stability
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < logits.cols(); j++) {
            max_logit = std::max(max_logit, logits(i, j));
        }
        
        // Compute softmax with improved numerical stability
        float sum_exp = 0.0f;
        std::vector<float> probs(logits.cols());
        
        for (size_t j = 0; j < logits.cols(); j++) {
            probs[j] = std::exp(logits(i, j) - max_logit);
            sum_exp += probs[j];
        }
        
        // Compute cross-entropy loss
        for (size_t j = 0; j < logits.cols(); j++) {
            probs[j] /= (sum_exp + epsilon);
            if (target_distribution(i, j) > 0.0f) {
                loss -= target_distribution(i, j) * std::log(probs[j] + epsilon);
            }
        }
    }
    
    return loss / logits.rows();
}

TransformerConfig Utils::load_config(const std::string& config_path) {
    TransformerConfig config;
    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open config file: " + config_path);
        }
        
        nlohmann::json j;
        file >> j;
        
        // Parse model settings
        auto& model = j["model"];
        config.vocab_size = model["vocab_size"];
        config.hidden_size = model["hidden_size"];
        config.num_heads = model["num_heads"];
        config.num_layers = model["num_layers"];
        config.head_dim = model["head_dim"];
        config.intermediate_size = model["intermediate_size"];
        
        // Parse training settings
        auto& training = j["training"];
        config.batch_size = training["batch_size"];
        config.num_epochs = training["num_epochs"];
        config.dropout_rate = training["dropout_rate"];
        config.weight_decay = training["weight_decay"];
        
        // Parse attention settings
        auto& attention = j["attention"];
        config.use_flash_attention = attention["use_flash_attention"];
        config.use_rope = attention["use_rope"];
        config.use_sliding_window = attention["use_sliding_window"];
        config.window_size = attention["window_size"];
        config.use_gqa = attention["use_gqa"];
        config.num_kv_heads = attention["num_kv_heads"];
        
        // Parse optimization settings
        auto& optimization = j["optimization"];
        config.use_cuda = optimization["use_cuda"];
        config.use_fp16 = optimization["use_fp16"];
        config.use_gradient_checkpointing = optimization["use_gradient_checkpointing"];
        config.memory_pool_size = optimization["memory_pool_size"];
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Error parsing config file: " + std::string(e.what()));
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
    while (std::getline(file, line)) {
        size_t delimiter_pos = line.find('|');
        if (delimiter_pos != std::string::npos) {
            std::string input = line.substr(0, delimiter_pos);
            std::string output = line.substr(delimiter_pos + 1);
            training_pairs.emplace_back(input, output);
        }
    }
    return training_pairs;
}

void Utils::analyze_token_mappings(const std::vector<std::pair<std::string, std::string>>& training_data, 
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
                if (tokenizer.decode({token}) == "<unk>") {
                    unknown_tokens++;
                    unknown_words[tokenizer.decode({token})]++;
                }
            }
        }
    }
    
    std::cout << "Token Mapping Statistics:\n"
              << "Total words: " << total_words << "\n"
              << "Unknown tokens: " << unknown_tokens 
              << " (" << (100.0f * unknown_tokens / total_words) << "%)\n";
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
    while (std::getline(file, line)) {
        size_t delimiter_pos = line.find('|');
        if (delimiter_pos != std::string::npos) {
            validation_pairs.emplace_back(line.substr(0, delimiter_pos),
                                        line.substr(delimiter_pos + 1));
        }
    }
    return validation_pairs;
}

bool Utils::validate_input_sequence(const std::vector<int>& tokens, 
                                  size_t vocab_size, 
                                  size_t max_seq_length) {
    if (tokens.empty() || tokens.size() > max_seq_length) {
        return false;
    }
    
    for (int token : tokens) {
        if (token < 0 || static_cast<size_t>(token) >= vocab_size) {
            return false;
        }
    }
    return true;
}

void Utils::print_matrix(const Matrix& m, const std::string& name,
                        size_t max_rows, size_t max_cols) {
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

void Utils::print_top_predictions(const Matrix& logits, const Tokenizer& tokenizer,
                                size_t k) {
    std::vector<float> last_logits;
    for (size_t i = 0; i < logits.cols(); ++i) {
        last_logits.push_back(logits(logits.rows() - 1, i));
    }

    float max_logit = *std::max_element(last_logits.begin(), last_logits.end());
    std::vector<float> probs(last_logits.size());
    float sum_exp = 0.0f;
    
    for (size_t i = 0; i < last_logits.size(); ++i) {
        probs[i] = std::exp(last_logits[i] - max_logit);
        sum_exp += probs[i];
    }
    
    for (float& prob : probs) {
        prob /= sum_exp;
    }

    std::vector<std::pair<float, int>> scores;
    for (size_t i = 0; i < probs.size(); ++i) {
        if (tokenizer.get_vocabulary().is_noun(tokenizer.decode({static_cast<int>(i)}))) {
            scores.push_back({probs[i], static_cast<int>(i)});
        }
    }

    if (!scores.empty()) {
        std::partial_sort(
            scores.begin(), 
            scores.begin() + std::min(k, scores.size()), 
            scores.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    std::cout << "\nTop " << k << " noun predictions:\n";
    for (size_t i = 0; i < std::min(k, scores.size()); ++i) {
        std::string token = tokenizer.decode({scores[i].second});
        std::cout << i + 1 << ". \"" << token << "\" (probability: " << std::fixed
                  << std::setprecision(4) << scores[i].first << ")\n";
    }
}

float Utils::evaluate_validation(Transformer& transformer, const Tokenizer& tokenizer,
                                const std::vector<std::pair<std::string, std::string>>& validation_data) {
    float total_loss = 0.0f;
    size_t correct_predictions = 0;
    size_t total_predictions = 0;

    transformer.set_training(false);  // Set model to evaluation mode

    for (const auto& pair : validation_data) {
        // Preprocess input
        std::string processed_input = pair.first;
        tokenizer.preprocess_text(processed_input);
        std::vector<int> input_tokens = tokenizer.encode(processed_input);

        // Get model prediction
        Matrix output = transformer.forward(input_tokens);
        Matrix logits = transformer.get_lm_head()->forward(output);

        // Get target
        std::string processed_target = pair.second;
        tokenizer.preprocess_text(processed_target);
        std::vector<int> target_tokens = tokenizer.encode(processed_target);

        // Create target distribution
        Matrix target_distribution(1, tokenizer.vocab_size(), 0.0f);
        if (!target_tokens.empty()) {
            target_distribution(0, target_tokens.back()) = 1.0f;
        }

        // Compute loss
        float loss = compute_batch_loss(logits, target_distribution);
        total_loss += loss;

        // Check if prediction matches target
        std::vector<float> last_logits;
        for (size_t i = 0; i < logits.cols(); ++i) {
            last_logits.push_back(logits(logits.rows() - 1, i));
        }
        int predicted_token = std::max_element(last_logits.begin(), last_logits.end()) - last_logits.begin();
        
        if (!target_tokens.empty() && predicted_token == target_tokens.back()) {
            correct_predictions++;
        }
        total_predictions++;
    }

    transformer.set_training(true);  // Set model back to training mode
    
    float accuracy = static_cast<float>(correct_predictions) / total_predictions;
    float avg_loss = total_loss / validation_data.size();
    
    std::cout << "\nValidation Results:\n"
              << "Average Loss: " << avg_loss << "\n"
              << "Accuracy: " << (accuracy * 100.0f) << "%\n";
    
    return avg_loss;
} 
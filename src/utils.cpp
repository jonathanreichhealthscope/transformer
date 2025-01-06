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
    std::cout << "logits shape: " << logits.shape() << std::endl;
    std::cout << "target distribution shape: " << target_distribution.shape() << std::endl;
    for (size_t i = 0; i < logits.rows(); i++) {
        // Find max logit for numerical stability
        float max_logit = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < logits.cols(); j++) {
            max_logit = std::max(max_logit, logits(i, j));
        }
        std::cout << "max logit: " << max_logit << std::endl;
        // Compute softmax with improved numerical stability
        float sum_exp = 0.0f;
        std::vector<float> probs(logits.cols());
        std::cout << "probs shape: " << probs.size() << std::endl;
        for (size_t j = 0; j < logits.cols(); j++) {
            probs[j] = std::exp(logits(i, j) - max_logit);
            sum_exp += probs[j];
        }
        std::cout << "sum exp: " << sum_exp << std::endl;
        // Compute cross-entropy loss
        for (size_t j = 0; j < logits.cols(); j++) {
            probs[j] /= (sum_exp + epsilon);
            if (target_distribution(i, j) > 0.0f) {
                loss -= target_distribution(i, j) * std::log(probs[j] + epsilon);
            }
        }
        std::cout << "loss: " << loss << std::endl;
    }
    
    return loss / logits.rows();
}

TransformerConfig Utils::load_config(const std::string& config_path) {
    TransformerConfig config;
    try {
        // Try multiple possible locations
        std::vector<std::string> possible_paths = {
            config_path,
            "../" + config_path,
            "../../" + config_path,
            "./config/transformer_config.json"
        };

        std::ifstream config_file;
        for (const auto& path : possible_paths) {
            config_file.open(path);
            if (config_file.is_open()) {
                break;
            }
        }

        if (!config_file.is_open()) {
            throw std::runtime_error("Could not open config file: " + config_path);
        }

        nlohmann::json j;
        config_file >> j;
        
        // Parse top-level settings
        config.vocab_size = j["vocab_size"];
        config.max_seq_length = j["max_seq_length"];
        config.hidden_size = j["hidden_size"];
        config.num_heads = j["num_heads"];
        config.num_layers = j["num_layers"];
        config.batch_size = j["batch_size"];
        config.num_epochs = j["num_epochs"];
        config.head_dim = config.hidden_size / config.num_heads;
        config.intermediate_size = 4 * config.hidden_size;
        
        // Parse attention settings
        auto& attention = j["attention"];
        config.use_flash_attention = attention["use_flash_attention"];
        config.use_rope = attention["use_rope"];
        config.use_sliding_window = attention["use_sliding_window"];
        config.window_size = attention["window_size"];
        if (attention.contains("use_gqa")) {
            config.use_gqa = attention["use_gqa"].get<bool>();
            if (config.use_gqa) {
                config.num_kv_heads = attention["num_kv_heads"].get<size_t>();
            } else {
                config.num_kv_heads = config.num_heads;  // No GQA, use same number
            }
        }
        
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
                if (tokenizer.decode({token}) == " ") {
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
    std::cout << "\n=== Evaluating Validation Data ===\n";
    
    float total_loss = 0.0f;
    size_t correct_predictions = 0;
    size_t total_predictions = 0;

    // Validate we have data to process
    if (validation_data.empty()) {
        std::cout << "Warning: Empty validation data\n";
        return 0.0f;
    }

    transformer.set_training(false);  // Set model to evaluation mode

    for (const auto& pair : validation_data) {
        // Preprocess input
        std::string processed_input = pair.first;
        std::cout << "Processing input: '" << processed_input << "'\n";
        tokenizer.preprocess_text(processed_input);
        std::cout << "Preprocessed input: '" << processed_input << "'\n";
        
        std::vector<int> input_tokens = tokenizer.encode(processed_input);
        std::cout << "Encoded input tokens: ";
        for (int token : input_tokens) {
            std::cout << token << " ";
        }
        std::cout << "\n";
        
        // Skip empty sequences
        if (input_tokens.empty()) {
            std::cout << "Warning: Empty input tokens, skipping\n";
            continue;
        }

        // Validate input tokens
        if (!Utils::validate_input_sequence(input_tokens, tokenizer.vocab_size())) {
            std::cout << "Warning: Invalid input sequence, skipping\n";
            continue;
        }

        try {
            // Get model prediction
            std::cout << "Calling transformer.forward with " << input_tokens.size() << " tokens\n";
            Matrix output = transformer.forward(input_tokens);
            std::cout << "Forward pass output shape: " << output.rows() << "x" << output.cols() << "\n";

            if (output.rows() == 0 || output.cols() == 0) {
                std::cout << "Warning: Empty output from transformer, skipping\n";
                continue;
            }

            auto lm_head = transformer.get_lm_head();
            if (!lm_head) {
                std::cerr << "Error: Language model head not initialized. Initializing now...\n";
                std::cout << "Error: Null language model head\n";
                continue;
            }

            Matrix logits = lm_head->forward(output);
            std::cout << "Logits shape: " << logits.rows() << "x" << logits.cols() << "\n";

            // Ensure we have valid output dimensions
            if (logits.rows() == 0 || logits.cols() != tokenizer.vocab_size()) {
                continue;
            }
            std::cout << "\n=== get target ===\n";
            // Get target
            std::string processed_target = pair.second;
            tokenizer.preprocess_text(processed_target);
            std::cout <<"encode target" << std::endl;
            std::vector<int> target_tokens = tokenizer.encode(processed_target);
            std::cout << "target tokens: " << target_tokens.size() << std::endl;
            // Create target distribution
            Matrix target_distribution(1, tokenizer.vocab_size(), 0.0f);
            if (!target_tokens.empty()) {
                target_distribution(0, target_tokens.back()) = 1.0f;
            }
            std::cout << "target distribution: " << target_distribution.rows() << "x" << target_distribution.cols() << std::endl;
            // Compute loss using only the last token's prediction
            Matrix last_token_logits(1, logits.cols());
            for (size_t i = 0; i < logits.cols(); ++i) {
                last_token_logits(0, i) = logits(logits.rows() - 1, i);
            }
            std::cout << "last token logits: " << last_token_logits.rows() << "x" << last_token_logits.cols() << std::endl;
            float loss = compute_batch_loss(last_token_logits, target_distribution);
            total_loss += loss;

            // Check if prediction matches target
            int predicted_token = -1;
            float max_logit = -std::numeric_limits<float>::infinity();
            std::cout << "iterating through logits" << std::endl;
            for (size_t i = 0; i < logits.cols(); ++i) {
                float val = logits(logits.rows() - 1, i);
                if (val > max_logit) {
                    max_logit = val;
                    predicted_token = i;
                }
            }
            std::cout << "predicted token: " << predicted_token << std::endl;

            if (!target_tokens.empty() && predicted_token == target_tokens.back()) {
                correct_predictions++;
            }
            total_predictions++;
        } catch (const std::exception& e) {
            std::cout << "Error evaluating validation: " << e.what() << "\n";
        }
    }
    std::cout << "total loss: " << total_loss << std::endl;
    std::cout << "total predictions: " << total_predictions << std::endl;
    std::cout << "correct predictions: " << correct_predictions << std::endl;
    transformer.set_training(true);  // Reset to training mode
    return total_predictions > 0 ? total_loss / total_predictions : 0.0f;
} 
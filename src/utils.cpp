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

bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && 
           str.compare(0, prefix.size(), prefix) == 0;
}

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
                                               const Tokenizer& tokenizer, size_t vocab_size,
                                               size_t input_max_seq_len) {
    // Calculate total size based on input sequence length
    size_t batch_size = target_tokens.size();
    size_t total_tokens = batch_size * input_max_seq_len;
    
    // Create target distribution for all token positions
    Matrix target_distribution(total_tokens, vocab_size, 0.0f);
    
    // Set target distribution for each token in each sequence
    size_t current_pos = 0;
    for (size_t seq = 0; seq < target_tokens.size(); seq++) {
        
        // Set actual tokens
        for (size_t i = 0; i < target_tokens[seq].size(); i++) {
            target_distribution(current_pos, target_tokens[seq][i]) = 1.0f;
            current_pos++;
        }
        
        // Pad remaining positions with pad token
        for (size_t i = target_tokens[seq].size(); i < input_max_seq_len; i++) {
            target_distribution(current_pos, tokenizer.get_pad_token_id()) = 1.0f;
            current_pos++;
        }
    }
    
    std::cout << "Final target distribution shape: " 
              << target_distribution.rows() << "x" << target_distribution.cols() << std::endl;
    std::cout << "Final current_pos: " << current_pos << "\n";
    std::cout << "=== Target Distribution Creation Complete ===\n\n";
    
    return target_distribution;
}

float Utils::compute_batch_loss(const Matrix& logits, const Matrix& target_distribution) {
    if (logits.empty() || target_distribution.empty()) {
        return 0.0f;
    }

    float total_loss = 0.0f;
    const size_t batch_size = logits.rows();
    const size_t vocab_size = logits.cols();

    // Pre-compute max logits and sums for numerical stability
    std::vector<float> max_logits(batch_size, -std::numeric_limits<float>::infinity());
    std::vector<float> sums(batch_size, 0.0f);

    #pragma omp parallel for reduction(+:total_loss)
    for (size_t i = 0; i < batch_size; ++i) {
        // Find max logit for numerical stability
        for (size_t j = 0; j < vocab_size; ++j) {
            max_logits[i] = std::max(max_logits[i], logits(i, j));
        }

        // Compute sum of exp(logits - max_logit)
        for (size_t j = 0; j < vocab_size; ++j) {
            sums[i] += std::exp(logits(i, j) - max_logits[i]);
        }

        // Compute cross entropy loss only for non-zero target probabilities
        for (size_t j = 0; j < vocab_size; ++j) {
            if (target_distribution(i, j) > 0.0f) {
                float log_prob = logits(i, j) - max_logits[i] - std::log(sums[i]);
                total_loss -= target_distribution(i, j) * log_prob;
            }
        }
    }

    return total_loss / batch_size;
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
            std::cout << "Loaded use_gqa from config: " << config.use_gqa << std::endl;
            if (config.use_gqa) {
                if (attention.contains("num_kv_heads")) {
                    config.num_kv_heads = attention["num_kv_heads"].get<size_t>();
                } else {
                    config.num_kv_heads = config.num_heads / 2; // Default to half the heads
                }
                std::cout << "Using GQA with num_heads=" << config.num_heads
                          << " and num_kv_heads=" << config.num_kv_heads << std::endl;
            } else {
                config.num_kv_heads = config.num_heads; // No GQA, use same number
            }
        }

        // Parse optimization settings
        auto& optimization = j["optimization"];
        config.use_fp16 = optimization["use_fp16"];
        config.use_gradient_checkpointing = optimization["use_gradient_checkpointing"];
        config.memory_pool_size = optimization["memory_pool_size"];

        // Parse beam search settings
        if (j.contains("beam_search")) {
            auto& beam = j["beam_search"];
            config.use_beam_search = beam.value("use_beam_search", true);  // Default to true for backward compatibility
            config.beam_size = beam["beam_size"];
            config.length_penalty = beam["length_penalty"];
            config.temperature = beam["temperature"];
            config.top_p = beam["top_p"];
            config.max_length = beam["max_length"];
        } else {
            // Default values if not specified
            config.use_beam_search = true;  // Default to true for backward compatibility
            config.beam_size = 5;
            config.length_penalty = 0.6f;
            config.temperature = 1.0f;
            config.top_p = 0.9f;
            config.max_length = 20;
        }

        // Add checkpoint loading settings
        if (j.contains("load_from_checkpoint")) {
            config.load_from_checkpoint = j["load_from_checkpoint"].get<bool>();
            if (config.load_from_checkpoint && j.contains("checkpoint_to_load")) {
                config.checkpoint_to_load = j["checkpoint_to_load"].get<std::string>();
                std::cout << "Will load checkpoint from: " << config.checkpoint_to_load
                          << std::endl;
            }
        }

        // Load tokenizer settings
        if (j.contains("tokenizer")) {
            const auto& tok = j["tokenizer"];
            config.tokenizer.use_subword = tok.value("use_subword", true);
            config.tokenizer.vocab_size = tok.value("vocab_size", 32000);
            config.tokenizer.model_path = tok.value("model_path", "model/tokenizer.model");
            config.tokenizer.special_tokens = tok.value("special_tokens", 
                std::vector<std::string>{"<pad>", "<s>", "</s>", "<unk>", "<mask>"});
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

    // Apply data augmentation
    DataAugmentation augmenter(0.3f, 0.3f);
    auto augmented_pairs = augmenter.augmentDataset(training_pairs);
    
    std::cout << "Original dataset size: " << training_pairs.size() << std::endl;
    std::cout << "Augmented dataset size: " << augmented_pairs.size() << std::endl;
    
    return augmented_pairs;
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
    while (std::getline(file, line)) {
        size_t delimiter_pos = line.find('|');
        if (delimiter_pos != std::string::npos) {
            validation_pairs.emplace_back(line.substr(0, delimiter_pos),
                                          line.substr(delimiter_pos + 1));
        }
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

void Utils::print_top_predictions(const Matrix& logits, const Tokenizer& tokenizer, Transformer& transformer, int k) {
    // Initialize beam search with parameters tuned for diversity
    BeamSearch beam_search(k,       // beam width
                          1.0f,     // length penalty (higher favors longer sequences)
                          1.2f,     // temperature (higher = more diversity) 
                          2.0f,     // diversity strength (much higher for stronger diversity)
                          50,       // top_k (reduced to focus on more likely tokens)
                          0.9f);    // top_p (nucleus sampling threshold)
    
    // Convert logits matrix to vector for beam search
    std::vector<float> initial_logits;
    initial_logits.reserve(logits.cols());  // Pre-reserve space
    for (int j = 0; j < logits.cols(); j++) {
        initial_logits.push_back(logits(logits.rows() - 1, j));
    }
    
    // Store original input tokens before we start generating
    std::vector<int> original_input = transformer.get_last_input();
    std::string original_query = transformer.get_last_query();
    size_t input_length = original_input.size();
    
    // Create next token function that uses the transformer with caching
    auto next_token_fn = [&transformer, &tokenizer](const std::vector<int>& tokens) -> std::vector<float> {
        try {
            // Only forward the last token, using the cached key-value pairs
            if (tokens.empty()) return std::vector<float>();
            std::vector<int> last_token = {tokens.back()};
            
            // Forward pass with empty query since we don't need it for generation
            Matrix next_hidden = transformer.forward(last_token, "", tokenizer, true);
            if (next_hidden.empty()) return std::vector<float>();
            
            Matrix next_logits = transformer.get_lm_head()->project_to_vocab(next_hidden);
            if (next_logits.empty()) return std::vector<float>();
            
            // Convert logits to vector
            std::vector<float> logits_vec;
            logits_vec.reserve(next_logits.cols());  // Pre-reserve space
            for (int j = 0; j < next_logits.cols(); j++) {
                logits_vec.push_back(next_logits(next_logits.rows() - 1, j));
            }
            return logits_vec;
        } catch (const std::exception& e) {
            std::cerr << "Error in next_token_fn: " << e.what() << std::endl;
            return std::vector<float>();
        }
    };
    
    // Clear the transformer's key-value cache before starting
    transformer.clear_kv_cache();
    
    // Initialize the cache with the input sequence
    if (!original_input.empty()) {
        Matrix initial_hidden = transformer.forward(original_input, original_query, tokenizer, true);
    }
    
    // Perform beam search with shorter max length to prevent runaway generation
    std::vector<BeamSearch::Hypothesis> hypotheses = beam_search.search(
        initial_logits, next_token_fn, 10, tokenizer.get_eos_token_id());
    
    // Clear the cache after we're done
    transformer.clear_kv_cache();
    
    // Print predictions
    std::cout << "\nQuery: \"" << original_query << "\"" << std::endl;
    std::cout << "Top " << k << " predicted sequences:" << std::endl;
    int predictions_shown = 0;
    
    for (const auto& hyp : hypotheses) {
        if (predictions_shown >= k) break;
        
        try {
            // Get just the generated part by removing the input tokens
            if (hyp.tokens.size() <= input_length) continue;
            
            std::vector<int> generated_tokens;
            generated_tokens.reserve(hyp.tokens.size() - input_length);  // Pre-reserve space
            for (size_t i = input_length; i < hyp.tokens.size(); i++) {
                generated_tokens.push_back(hyp.tokens[i]);
            }
            
            // Decode only the generated tokens
            std::string decoded = tokenizer.decode(generated_tokens);
            
            // Skip empty sequences
            if (decoded.empty()) continue;
            
            // Print the prediction with its score
            std::cout << predictions_shown + 1 << ". \"" << decoded << "\" (score=" 
                      << hyp.score << ")" << std::endl;
            predictions_shown++;
        } catch (const std::exception& e) {
            std::cerr << "Error processing hypothesis: " << e.what() << std::endl;
            continue;
        }
    }
    
    if (predictions_shown < k) {
        std::cout << "(Not enough valid predictions found)" << std::endl;
    }
}

float Utils::evaluate_validation(
    Transformer& transformer, const Tokenizer& tokenizer,
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

    transformer.set_training(false); // Set model to evaluation mode

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
            Matrix output = transformer.forward(input_tokens, "", tokenizer);
            std::cout << "Forward pass output shape: " << output.rows() << "x" << output.cols()
                      << "\n";

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

            // For single token prediction, we don't need beam search
            // Just get the highest probability token directly
            int predicted_token = -1;
            float max_logit = -std::numeric_limits<float>::infinity();
            
            for (size_t i = 0; i < logits.cols(); ++i) {
                float val = logits(logits.rows() - 1, i);
                if (val > max_logit) {
                    max_logit = val;
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

            // Compute loss using only the last token's prediction
            Matrix last_token_logits(1, logits.cols());
            for (size_t i = 0; i < logits.cols(); ++i) {
                last_token_logits(0, i) = logits(logits.rows() - 1, i);
            }

            float loss = compute_batch_loss(last_token_logits, target_distribution);
            total_loss += loss;

            // Check if prediction matches target
            if (!target_tokens.empty() && predicted_token == target_tokens.back()) {
                correct_predictions++;
            }
            total_predictions++;

        } catch (const std::exception& e) {
            std::cout << "Error evaluating validation: " << e.what() << "\n";
        }
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

void from_json(const nlohmann::json& j, TransformerConfig::TokenizerConfig& t) {
    j.at("use_subword").get_to(t.use_subword);
    j.at("vocab_size").get_to(t.vocab_size);
    j.at("model_path").get_to(t.model_path);
    j.at("special_tokens").get_to(t.special_tokens);
}
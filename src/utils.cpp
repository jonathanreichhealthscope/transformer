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
            float weight = (i >= noun_phrase_start) ? 1.0f : 0.5f;  // Higher weight for noun phrase tokens
            target_distribution(current_pos, sequence[i]) = weight;
            current_pos++;
        }
        
        // Pad remaining positions with pad token
        for (size_t i = sequence.size(); i < input_max_seq_len; i++) {
            target_distribution(current_pos, tokenizer.get_pad_token_id()) = 1.0f;
            current_pos++;
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
    if (logits.empty() || target_distribution.empty()) {
        return 0.0f;
    }

    float total_loss = 0.0f;
    const size_t batch_size = logits.rows();
    const size_t vocab_size = logits.cols();

    // Pre-compute which tokens are nouns - do this once for the whole vocabulary
    static std::vector<bool> is_noun_cache;
    static bool cache_initialized = false;
    
    if (!cache_initialized) {
        is_noun_cache.resize(vocab_size);
        for (size_t j = 0; j < vocab_size; ++j) {
            std::string token = tokenizer.decode({static_cast<int>(j)});
            is_noun_cache[j] = tokenizer.is_noun(token);
        }
        cache_initialized = true;
    }

    // Pre-compute max logits and sums for numerical stability
    std::vector<float> max_logits(batch_size, -std::numeric_limits<float>::infinity());
    std::vector<float> sums(batch_size, 0.0f);

    #pragma omp parallel for reduction(+:total_loss)
    for (size_t i = 0; i < batch_size; ++i) {
        // Find max logit for numerical stability, but only consider noun tokens
        for (size_t j = 0; j < vocab_size; ++j) {
            if (is_noun_cache[j]) {
                max_logits[i] = std::max(max_logits[i], logits(i, j));
            }
        }

        // Compute sum of exp(logits - max_logit) only for noun tokens
        for (size_t j = 0; j < vocab_size; ++j) {
            if (is_noun_cache[j]) {
                sums[i] += std::exp(logits(i, j) - max_logits[i]);
            }
        }

        // Compute cross entropy loss only for non-zero target probabilities
        for (size_t j = 0; j < vocab_size; ++j) {
            if (target_distribution(i, j) > 0.0f) {
                if (is_noun_cache[j]) {
                    float log_prob = logits(i, j) - max_logits[i] - std::log(sums[i]);
                    total_loss -= target_distribution(i, j) * log_prob;
                } else {
                    // Add a small penalty for non-noun predictions when they should be nouns
                    total_loss += 0.1f * target_distribution(i, j);
                }
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
        } else {
            // Default values if not specified
            config.beam_search.use_beam_search = true;
            config.beam_search.beam_size = 5;
            config.beam_search.beams_per_group = 4;
            config.beam_search.num_groups = 3;
            config.beam_search.length_penalty = 0.6f;
            config.beam_search.temperature = 1.0f;
            config.beam_search.top_p = 0.9f;
            config.beam_search.max_length = 20;
            config.beam_search.initial_temperature = 3.0f;
            config.beam_search.initial_noise_scale = 0.8f;
            config.beam_search.diversity_strength = 4.0f;
            config.beam_search.top_k = 100;
            config.beam_search.token_noise_scale = 0.1f;
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
                std::vector<std::string>{"<pad>", "", " ", "</s>", "<mask>"});
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
    std::unordered_map<std::string, int> token_frequency;
    size_t total_tokens = 0;
    size_t unique_tokens = 0;
    
    while (std::getline(file, line)) {
        size_t delimiter_pos = line.find('|');
        if (delimiter_pos != std::string::npos) {
            std::string input = line.substr(0, delimiter_pos);
            std::string output = line.substr(delimiter_pos + 1);
            training_pairs.emplace_back(input, output);
            
            // Count tokens in both input and output
            for (const auto& text : {input, output}) {
                std::istringstream iss(text);
                std::string word;
                while (iss >> word) {
                    token_frequency[word]++;
                    total_tokens++;
                }
            }
        }
    }
    
    unique_tokens = token_frequency.size();
    std::cout << "\nTraining Data Statistics:" << std::endl;
    std::cout << "Total tokens: " << total_tokens << std::endl;
    std::cout << "Unique tokens: " << unique_tokens << std::endl;
    std::cout << "Token/sample ratio: " << (float)total_tokens / training_pairs.size() << std::endl;

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
    const auto& config = transformer.getConfig();
    size_t total_beam_width = std::max(config.beam_search.beam_size, static_cast<size_t>(k * 3));
    size_t num_groups = config.beam_search.num_groups;
    size_t beams_per_group = config.beam_search.beams_per_group;
    
    BeamSearch beam_search(
        beams_per_group,
        config.beam_search.length_penalty,
        config.beam_search.temperature,
        config.beam_search.diversity_strength,
        config.beam_search.top_k,
        config.beam_search.top_p
    );

    // Store predictions from each group
    std::vector<std::vector<BeamSearch::Hypothesis>> group_hypotheses;
    std::vector<float> initial_logits;
    initial_logits.reserve(logits.cols());
    
    // Get original input
    std::vector<int> original_input = transformer.get_last_input();
    std::string original_query = transformer.get_last_query();
    size_t input_length = original_input.size();

    // Run beam search for each group with different initial conditions
    for (size_t group = 0; group < num_groups; group++) {
        // Reset and prepare initial logits with group-specific noise
        initial_logits.clear();
        for (int j = 0; j < logits.cols(); j++) {
            float base_logit = logits(logits.rows() - 1, j) / config.beam_search.initial_temperature;
            float group_noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 
                              config.beam_search.initial_noise_scale * 
                              (1.0f + group * 0.5f);
            initial_logits.push_back(base_logit + group_noise);
        }
        
        auto next_token_fn = [&transformer, &tokenizer, group, &config](const std::vector<int>& tokens) -> std::vector<float> {
            try {
                Matrix next_hidden = transformer.forward(tokens, "", tokenizer, true);
                Matrix next_logits = transformer.get_lm_head()->project_to_vocab(next_hidden);
                
                std::vector<float> logits_vec;
                logits_vec.reserve(next_logits.cols());
                for (int j = 0; j < next_logits.cols(); j++) {
                    float logit = next_logits(next_logits.rows() - 1, j);
                    float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 
                                config.beam_search.token_noise_scale * 
                                (1.0f + group * 0.3f);
                    logits_vec.push_back(logit + noise);
                }
                return logits_vec;
            } catch (const std::exception& e) {
                std::cerr << "Error in next_token_fn: " << e.what() << std::endl;
                return std::vector<float>();
            }
        };

        const size_t MAX_LENGTH = input_length + 4;
        auto group_results = beam_search.search(
            initial_logits, next_token_fn, MAX_LENGTH, tokenizer.get_eos_token_id());
        group_hypotheses.push_back(group_results);
    }

    transformer.clear_kv_cache();

    // Store both noun and noun phrase predictions
    std::unordered_set<std::string> used_first_tokens;
    std::vector<std::pair<std::string, float>> valid_noun_phrases;
    std::vector<std::pair<std::string, float>> valid_nouns;

    // Process hypotheses from each group
    for (const auto& group_results : group_hypotheses) {
        for (const auto& hyp : group_results) {
            try {
                if (hyp.tokens.size() <= input_length) continue;
                
                std::vector<int> generated_tokens(
                    hyp.tokens.begin() + input_length,
                    hyp.tokens.end()
                );
                
                // Decode tokens and check if they form a valid noun or noun phrase
                std::string decoded;
                bool is_valid = false;
                std::string first_token;
                std::vector<std::string> words;
                
                // First pass: collect all words
                for (size_t i = 0; i < generated_tokens.size(); i++) {
                    std::string token = tokenizer.decode({generated_tokens[i]});
                    if (!token.empty()) {
                        if (!decoded.empty() && token[0] != ' ') {
                            decoded += " ";
                        }
                        
                        // Skip tokens that start with 'Ġ' or other special characters
                        if (token[0] == 'Ġ' || !std::isalpha(token[0])) continue;
                        
                        // Store first non-empty token
                        if (first_token.empty() && !token.empty() && token != " ") {
                            first_token = token;
                            words.push_back(token);
                        } else if (!token.empty() && token != " ") {
                            words.push_back(token);
                        }
                        
                        decoded += token;
                    }
                }
                
                // Validate the sequence
                if (words.size() == 1) {
                    // Single word - check if it's a noun or could be part of a noun phrase
                    is_valid = tokenizer.is_noun(words[0]) || 
                             tokenizer.is_adjective(words[0]) || 
                             tokenizer.is_determiner(words[0]);
                } else if (words.size() > 1) {
                    // Multiple words - check if it's a noun phrase
                    bool has_noun = false;
                    bool has_valid_modifier = false;
                    
                    for (const auto& word : words) {
                        if (tokenizer.is_noun(word)) {
                            has_noun = true;
                        } else if (tokenizer.is_adjective(word) || tokenizer.is_determiner(word)) {
                            has_valid_modifier = true;
                        }
                    }
                    is_valid = has_noun || (has_valid_modifier && words.size() < 3);
                }
                
                if (decoded.empty() || !is_valid) continue;
                
                // Filter out predictions with very low scores
                if (hyp.score < -20.0f) continue;  // Relaxed score threshold
                
                // Add initial space if needed
                if (decoded[0] != ' ') {
                    decoded = " " + decoded;
                }
                
                // Only add if we haven't seen this first token
                if (used_first_tokens.find(first_token) == used_first_tokens.end()) {
                    used_first_tokens.insert(first_token);
                    
                    // If it's a single token, it's a noun
                    // Otherwise, it's a noun phrase
                    if (words.size() == 1) {
                        valid_nouns.push_back({decoded, hyp.score});
                    } else {
                        valid_noun_phrases.push_back({decoded, hyp.score});
                    }
                    
                    // Break if we have enough predictions of both types
                    if (valid_nouns.size() >= k && valid_noun_phrases.size() >= k) break;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error processing hypothesis: " << e.what() << std::endl;
                continue;
            }
        }
        if (valid_nouns.size() >= k && valid_noun_phrases.size() >= k) break;
    }

    // Print predictions
    std::cout << "\nQuery: \"" << original_query << "\"" << std::endl;
    
    // Print noun predictions
    std::cout << "\nTop " << k << " predicted nouns:" << std::endl;
    for (size_t i = 0; i < std::min(valid_nouns.size(), static_cast<size_t>(k)); i++) {
        const auto& [prediction, score] = valid_nouns[i];
        std::cout << i + 1 << ". \"" << prediction << "\" (score=" 
                 << std::fixed << std::setprecision(4) << score << ")" << std::endl;
    }
    if (valid_nouns.empty()) {
        std::cout << "No valid nouns found." << std::endl;
    }

    // Print noun phrase predictions
    std::cout << "\nTop " << k << " predicted noun phrases:" << std::endl;
    for (size_t i = 0; i < std::min(valid_noun_phrases.size(), static_cast<size_t>(k)); i++) {
        const auto& [prediction, score] = valid_noun_phrases[i];
        std::cout << i + 1 << ". \"" << prediction << "\" (score=" 
                 << std::fixed << std::setprecision(4) << score << ")" << std::endl;
    }
    if (valid_noun_phrases.empty()) {
        std::cout << "No valid noun phrases found." << std::endl;
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

            float loss = compute_batch_loss(last_token_logits, target_distribution, tokenizer);
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
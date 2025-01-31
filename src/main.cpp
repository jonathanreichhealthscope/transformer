#include "../include/main.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include "../include/tokenizer.hpp"
#include "../include/utils.hpp"
#include "../include/phrase_analysis.hpp"
#include "../include/training/training.hpp"  // Include unified training header
#include "../include/hyperparameter_tuner.hpp"
#include "../include/count_vocabulary.hpp"

// Add necessary forward declarations and structures
std::unique_ptr<Tokenizer> tokenizer;
PerformanceMetrics metrics;

// Configuration constants
const float INITIAL_LEARNING_RATE = 0.001f;
float learning_rate = INITIAL_LEARNING_RATE;
float prev_loss = std::numeric_limits<float>::max();
size_t global_step = 0;

// Add training components with proper types
TrainingStateManagerPtr training_manager;
TrainingMonitorPtr training_monitor;

// Data structure for preprocessing
struct Data {
    std::vector<std::vector<float>> samples;
    std::vector<int> labels;
};

// Normalization helper function
std::vector<float> normalize(const std::vector<float>& sample) {
    if (sample.empty()) return sample;
    
    float mean = 0.0f;
    float std_dev = 0.0f;
    
    // Calculate mean
    for (float val : sample) {
        mean += val;
    }
    mean /= sample.size();
    
    // Calculate standard deviation
    for (float val : sample) {
        float diff = val - mean;
        std_dev += diff * diff;
    }
    std_dev = std::sqrt(std_dev / sample.size());
    
    // Normalize
    std::vector<float> normalized(sample.size());
    if (std_dev > 0.0f) {
        for (size_t i = 0; i < sample.size(); i++) {
            normalized[i] = (sample[i] - mean) / std_dev;
        }
    } else {
        normalized = sample;  // If std_dev is 0, keep original values
    }
    
    return normalized;
}

float compute_loss(const Matrix& logits, const std::vector<int>& target_tokens, const Tokenizer& tokenizer) {
    float loss = 0.0f;
    const int sep_token_id = tokenizer.get_sep_token_id();
    bool after_separator = false;
    const float epsilon = 1e-6f;  // Increased epsilon for better stability
    
    for (size_t i = 0; i < target_tokens.size() - 1; i++) {
        int current_token = target_tokens[i];
        int next_token = target_tokens[i + 1];
        
        // Track if we're after the separator
        if (current_token == sep_token_id) {
            after_separator = true;
        }
        
        // Get predicted probability distribution
        Vector logits_row = logits.row(i);
        std::vector<float> row_data(logits_row.begin(), logits_row.end());
        
        // Find max for numerical stability
        float max_val = *std::max_element(row_data.begin(), row_data.end());
        
        // Compute softmax with numerical stability
        std::vector<float> probs(row_data.size());
        float sum_exp = 0.0f;
        
        // First pass: compute exponentials and sum
        for (size_t j = 0; j < row_data.size(); j++) {
            float val = std::exp(row_data[j] - max_val);
            probs[j] = val;
            sum_exp += val;
        }
        
        // Second pass: normalize and ensure no zeros
        sum_exp = std::max(sum_exp, epsilon);  // Prevent division by zero
        for (float& p : probs) {
            p = std::max(p / sum_exp, epsilon);  // Ensure no probability is exactly zero
        }
        
        // Compute loss with numerical stability checks
        float prob = probs[next_token];
        if (prob <= 0.0f || !std::isfinite(prob)) {
            prob = epsilon;
        }
        float token_loss = -std::log(prob);
        
        // Check for NaN/Inf
        if (!std::isfinite(token_loss)) {
            std::cout << "Warning: Non-finite loss detected. Using fallback value." << std::endl;
            token_loss = 100.0f;  // Fallback to a high but finite loss
        }
        
        // Apply additional penalty for format violations after separator
        if (after_separator) {
            std::string next_token_str = tokenizer.decode({next_token});
            if (!next_token_str.empty() && next_token_str[0] != ' ') {
                token_loss *= 1.5f;
            }
        }
        
        loss += token_loss;
    }
    
    // Final stability check
    if (!std::isfinite(loss)) {
        std::cout << "Warning: Non-finite total loss detected. Using fallback value." << std::endl;
        return 100.0f;  // Fallback to a high but finite loss
    }
    
    return loss / std::max(static_cast<float>(target_tokens.size()), epsilon);
}

// Update the training loop
void train_epoch(Transformer& model, const std::vector<std::pair<std::string, std::string>>& training_pairs,
                float learning_rate, const Tokenizer& tokenizer) {
    if (!training_manager) {
        training_manager = std::make_unique<TrainingStateManager>(learning_rate);
    }
    if (!training_monitor) {
        training_monitor = std::make_unique<TrainingMonitor>();
    }

    for (const auto& [context, target] : training_pairs) {
        // Determine the phrase type based on the target
        PhraseType phrase_type = PhraseTypeHandler::detect_phrase_type(target);
        std::string delimiter = PhraseTypeHandler::get_delimiter(phrase_type);
        
        // Combine context and target with appropriate delimiter
        std::string full_text = context + delimiter + target;
        std::string final_phrase = PhraseTypeHandler::extract_final_phrase(full_text);
        
        // Tokenize
        std::vector<int> tokens = tokenizer.encode(full_text);
        std::vector<int> final_phrase_tokens = tokenizer.encode(final_phrase);
        
        // Forward pass
        Matrix logits = model.forward(tokens, full_text, tokenizer);
        float loss = compute_loss(logits, tokens, tokenizer);
        
        // Add type-specific penalties
        switch (phrase_type) {
            case PhraseType::VERB:
                loss += compute_verb_penalty(logits, final_phrase_tokens, tokenizer);
                break;
            case PhraseType::ADJECTIVE:
                loss += compute_adjective_penalty(logits, final_phrase_tokens, tokenizer);
                break;
            default:
                break;
        }
        
        // Create gradients
        Matrix loss_gradients(logits.rows(), logits.cols());
        for (size_t i = 0; i < logits.rows(); i++) {
            for (size_t j = 0; j < logits.cols(); j++) {
                float base_gradient = logits(i, j) - (j < tokens.size() ? 1.0f : 0.0f);
                switch (phrase_type) {
                    case PhraseType::VERB:
                        base_gradient *= verb_gradient_factor(j, tokens, tokenizer);
                        break;
                    case PhraseType::ADJECTIVE:
                        base_gradient *= adjective_gradient_factor(j, tokens, tokenizer);
                        break;
                    default:
                        break;
                }
                loss_gradients(i, j) = base_gradient;
            }
        }
        
        // Update training state
        TrainingMetrics metrics(
            loss,                                                   // loss
            loss_gradients,                                        // gradients
            global_step / training_pairs.size(),                   // epoch
            global_step,                                           // step
            0.0f,                                                  // loss_trend (Will be computed by state manager)
            RunningStatistics()                                    // grad_stats (Will be updated by state manager)
        );
        
        training_manager->update_state(metrics);
        training_monitor->log_metrics(metrics);
        
        // Check if we should stop training
        if (training_monitor->should_stop_training()) {
            std::cout << "Training stopped due to monitor conditions" << std::endl;
            return;
        }
        
        // Get current learning rate from manager
        float current_lr = training_manager->get_learning_rate();
        
        // Only proceed with update if training is stable
        if (training_manager->is_stable()) {
            model.backward(loss_gradients, tokens, current_lr);
            model.update_parameters(current_lr);
        } else {
            std::cout << "Skipping update due to instability" << std::endl;
        }
        
        global_step++;
    }
}

// Helper functions for phrase type-specific loss components
float compute_verb_penalty(const Matrix& logits, const std::vector<int>& final_tokens,
                         const Tokenizer& tokenizer) {
    // Add penalty for predictions that are unlikely to be verbs
    float penalty = 0.0f;
    // Implementation details for verb-specific penalties
    return penalty;
}

float compute_adjective_penalty(const Matrix& logits, const std::vector<int>& final_tokens,
                              const Tokenizer& tokenizer) {
    // Add penalty for predictions that are unlikely to be adjectives
    float penalty = 0.0f;
    // Implementation details for adjective-specific penalties
    return penalty;
}

float verb_gradient_factor(size_t position, const std::vector<int>& tokens,
                         const Tokenizer& tokenizer) {
    // Adjust gradients for verb-specific learning
    return 1.0f; // Default implementation
}

float adjective_gradient_factor(size_t position, const std::vector<int>& tokens,
                              const Tokenizer& tokenizer) {
    // Adjust gradients for adjective-specific learning
    return 1.0f; // Default implementation
}

// Make predictions after each batch
void generate_predictions(Transformer& transformer, const std::string& input_text, Tokenizer* tokenizer) {
    std::cout << "\n=== Batch 0 Predictions for '" << input_text << "' ===" << std::endl;
    std::cout << "--" << std::endl;
    
    // Preprocess input
    std::string processed_input = input_text;
    tokenizer->preprocess_text(processed_input);
    std::vector<int> input_tokens = tokenizer->encode(processed_input);
    
    // Clear transformer state and set to eval mode
    transformer.clear_kv_cache();
    transformer.set_training(false);
    
    // Generate prediction
    Matrix hidden_states = transformer.forward(input_tokens, input_text, *tokenizer);
    std::cout << "\nAbout to call lm_head->forward" << std::endl << std::flush;
    Matrix logits = transformer.get_lm_head()->forward(hidden_states);
    std::cout << "Finished lm_head->forward" << std::endl << std::flush;
    
    // Get probabilities for last token with proper temperature scaling
    Vector last_row = logits.row(logits.rows() - 1);
    Matrix final_logits(1, last_row.size());
    for (size_t i = 0; i < last_row.size(); ++i) {
        final_logits(0, i) = last_row[i];
    }
    
    // Create a unique random generator for this prediction
    std::random_device rd;
    std::seed_seq seq{rd(), rd(), rd(), static_cast<unsigned int>(std::time(nullptr))};
    std::mt19937 gen(seq);
    
    // Get beam search config parameters
    const auto& config = transformer.getConfig();
    const auto& beam_config = config.beam_search;
    
    // Use configured temperature with dynamic adjustment
    float base_temperature = beam_config.temperature;
    float initial_temp_boost = beam_config.initial_temperature / base_temperature;
    
    // Increase temperature for more randomness
    float effective_temperature = base_temperature * initial_temp_boost * 1.5f;
    
    // Add stronger random noise for more variation
    std::normal_distribution<float> noise_dist(0.0f, beam_config.token_noise_scale * 2.0f);
    for (size_t i = 0; i < final_logits.cols(); ++i) {
        final_logits(0, i) += noise_dist(gen);
    }
    
    // Get token frequencies from the language model head
    const auto& token_frequencies = transformer.get_lm_head()->get_token_frequencies();
    
    // Create default frequencies if none available
    std::vector<float> default_frequencies;
    const std::vector<float>& freq_ref = token_frequencies.empty() ? default_frequencies : token_frequencies;
    if (freq_ref.empty()) {
        default_frequencies.resize(final_logits.cols(), 1.0f); // Equal frequencies if not available
    }
    
    // Convert logits to probabilities with temperature scaling and frequency debiasing
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < final_logits.cols(); ++i) {
        max_logit = std::max(max_logit, final_logits(0, i));
    }
    
    // Compute exponentials with temperature scaling and frequency debiasing
    float sum_exp = 0.0f;
    std::vector<float> exps(final_logits.cols());
    
    // Calculate frequency-based penalties
    std::vector<float> freq_penalties(final_logits.cols());
    float max_freq = freq_ref.empty() ? 1.0f : *std::max_element(freq_ref.begin(), freq_ref.end());
    max_freq = std::max(max_freq, 1.0f); // Ensure non-zero max frequency
    
    #pragma omp parallel for
    for (size_t i = 0; i < final_logits.cols(); ++i) {
        if (i < freq_ref.size()) {
            // Normalize frequency to [0, 1] and apply penalty
            float norm_freq = freq_ref[i] / max_freq;
            freq_penalties[i] = std::pow(1.0f - norm_freq, 0.4f); // Adjust power for penalty strength
        } else {
            freq_penalties[i] = 1.0f;
        }
    }
    
    #pragma omp parallel for reduction(+:sum_exp)
    for (size_t i = 0; i < final_logits.cols(); ++i) {
        // Apply frequency penalty to logits
        float penalized_logit = final_logits(0, i) * freq_penalties[i];
        float scaled_logit = (penalized_logit - max_logit) / effective_temperature;
        exps[i] = std::exp(scaled_logit);
        sum_exp += exps[i];
    }
    
    // Convert to probabilities and store token-probability pairs
    std::vector<std::pair<float, int>> token_probs;
    token_probs.reserve(final_logits.cols());
    for (size_t i = 0; i < final_logits.cols(); ++i) {
        float prob = exps[i] / sum_exp;
        
        // Additional diversity boost for less common tokens
        if (i < freq_ref.size() && freq_ref[i] < max_freq * 0.1f) {
            prob *= 1.2f; // Boost rare tokens
        }
        
        token_probs.push_back({prob, static_cast<int>(i)});
    }
    
    // Sort by probability
    std::sort(token_probs.begin(), token_probs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Apply more aggressive top-k filtering to remove very common tokens
    size_t effective_top_k = beam_config.top_k * 2; // Double the top-k to consider more candidates
    if (effective_top_k > 0 && effective_top_k < token_probs.size()) {
        token_probs.resize(effective_top_k);
    }
    
    // Apply nucleus sampling with higher threshold for more diversity
    float effective_top_p = std::min(0.98f, beam_config.top_p + 0.1f); // Increase top-p slightly
    float cumsum = 0.0f;
    std::vector<std::pair<float, int>> filtered_probs;
    filtered_probs.reserve(token_probs.size());
    
    for (const auto& [prob, token_id] : token_probs) {
        if (cumsum >= effective_top_p) break;
        
        // Skip very common tokens unless they're exceptionally high probability
        if (token_id < freq_ref.size() && 
            freq_ref[token_id] > max_freq * 0.8f && 
            prob < 0.5f) {
            continue;
        }
        
        filtered_probs.push_back({prob, token_id});
        cumsum += prob;
    }
    
    // Add some rare tokens to the mix
    size_t rare_tokens_added = 0;
    for (const auto& [prob, token_id] : token_probs) {
        if (rare_tokens_added >= 3) break; // Limit the number of rare tokens
        
        if (token_id < freq_ref.size() && 
            freq_ref[token_id] < max_freq * 0.05f && // Very rare tokens
            prob > 0.01f) { // But still somewhat relevant
            filtered_probs.push_back({prob * 1.5f, token_id}); // Boost their probability
            rare_tokens_added++;
        }
    }
    
    // Renormalize probabilities after filtering
    float filtered_sum = 0.0f;
    for (const auto& [prob, _] : filtered_probs) {
        filtered_sum += prob;
    }
    for (auto& [prob, _] : filtered_probs) {
        prob /= filtered_sum;
    }
    
    // Shuffle the filtered probabilities to break any remaining frequency bias
    std::shuffle(filtered_probs.begin(), filtered_probs.end(), gen);
    
    // Sort again by probability for display
    std::sort(filtered_probs.begin(), filtered_probs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Print top 5 predictions
    std::cout << "Top 5 predictions:" << std::endl;
    for (int i = 0; i < std::min(5, static_cast<int>(filtered_probs.size())); ++i) {
        int token_id = filtered_probs[i].second;
        float prob = filtered_probs[i].first;
        std::string token = tokenizer->decode({token_id});
        std::cout << std::fixed << std::setprecision(5)
                  << "   " << token << " (" << prob * 100 << "%)" << std::endl;
    }
    
    std::cout << "--" << std::endl;
    std::cout << "Nonzero final: " << filtered_probs.size() << "/" << token_probs.size() << std::endl << std::endl;
}

void preprocess_data(Data& data) {
    // Check normalization and augmentation
    // Ensure data is varied and correctly normalized
    for (auto& sample : data.samples) {
        sample = normalize(sample); // Ensure normalization is applied correctly
    }
}

// Add this function before the main training loop
void reinitialize_batch_weights(Transformer& transformer, const TransformerConfig& config, size_t global_step) {
    // Get a random seed based on time and batch number
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Dynamic temperature scaling
    float temperature = 1.0f + (5.0f * std::exp(-global_step / 500.0f));  // Starts at 6.0, decays to 1.0
    
    // Multiple scales for different layers
    float attention_scale = 0.15f * std::exp(-global_step / 2000.0f);  // Slower decay for attention
    float ffn_scale = 0.1f * std::exp(-global_step / 1000.0f);        // Faster decay for feed-forward
    float output_scale = 0.05f * std::exp(-global_step / 800.0f);     // Even faster for output layer
    
    // Minimum scales to maintain exploration
    attention_scale = std::max(0.02f, attention_scale);
    ffn_scale = std::max(0.01f, ffn_scale);
    output_scale = std::max(0.005f, output_scale);
    
    // Get all parameters that need reinitialization
    auto& params = transformer.parameters();
    
    // Reinitialize each parameter matrix with controlled randomness
    for (size_t p = 0; p < params.size(); p++) {
        Matrix& current_param = params[p];
        const size_t rows = current_param.rows();
        const size_t cols = current_param.cols();
        
        // Choose scale based on layer type (inferred from matrix dimensions)
        float scale;
        if (cols == config.hidden_size) {
            scale = attention_scale;  // Attention layers
        } else if (cols == config.intermediate_size) {
            scale = ffn_scale;        // Feed-forward layers
        } else {
            scale = output_scale;     // Output layers
        }
        
        // Process each matrix in parallel
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                // Thread-local random generators
                std::mt19937 local_gen(rd() + i * cols + j);  // Unique seed per element
                
                // Add temperature-based sampling
                std::normal_distribution<float> dist(0.0f, scale * temperature);
                float perturbation = dist(local_gen);
                
                // Current weight magnitude affects perturbation
                float current_value = std::abs(current_param(i, j));
                float adaptive_scale = scale / (1.0f + current_value * temperature);
                
                // Apply perturbation with probability decay
                float apply_prob = std::exp(-global_step / 3000.0f);  // Probability of applying perturbation
                if (std::uniform_real_distribution<float>(0.0f, 1.0f)(local_gen) < apply_prob) {
                    current_param(i, j) += perturbation * adaptive_scale;
                }
                
                // Occasionally flip sign of small weights to explore different patterns
                if (std::abs(current_param(i, j)) < 0.01f && 
                    std::uniform_real_distribution<float>(0.0f, 1.0f)(local_gen) < 0.1f) {
                    current_param(i, j) *= -1.0f;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "entering main" << std::endl;
    Logger& logger = Logger::getInstance();
    logger.startLogging();

    try {
        // Initialize random number generation
        Utils::initialize_random();
        std::filesystem::path exe_path = std::filesystem::current_path().parent_path();

        // First, count vocabulary size from training and validation files
        std::cout << "\nCounting unique tokens in training and validation files..." << std::endl;
        size_t custom_vocab_size = transformer::VocabularyCounter::countUniqueTokens(
            exe_path.string() + "/data/training_pairs.txt",
            exe_path.string() + "/data/validation_pairs.txt"
        );
        std::cout << "Number of unique tokens found in data files: " << custom_vocab_size << std::endl;

        // Load and update config with the counted vocabulary size
        std::filesystem::path config_path = exe_path / "config" / "transformer_config.json";
        
        // Read the config file
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            throw std::runtime_error("Could not open config file: " + config_path.string());
        }
        
        nlohmann::json config_json;
        config_file >> config_json;
        config_file.close();
        
        // Update vocabulary size in config
        size_t previous_vocab_size = 0;
        if (config_json.contains("vocab_size") && !config_json["vocab_size"].is_null()) {
            previous_vocab_size = config_json["vocab_size"].get<size_t>();
        }
        std::cout << "Previous vocabulary size in config: " << (previous_vocab_size == 0 ? "Not set" : std::to_string(previous_vocab_size)) << std::endl;
        
        config_json["vocab_size"] = custom_vocab_size;
        std::cout << "Updated vocabulary size in config to: " << custom_vocab_size << std::endl;
        
        // Write updated config back to file
        std::ofstream output_config_file(config_path);
        if (!output_config_file.is_open()) {
            throw std::runtime_error("Could not open config file for writing: " + config_path.string());
        }
        output_config_file << config_json.dump(4);
        output_config_file.close();

        // Now load the updated config for transformer initialization
        TransformerConfig config = Utils::load_config(config_path.string());
        
        // Initialize random seed using hardware entropy
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Create a seed sequence using multiple entropy sources
        std::vector<std::uint32_t> entropy{
            static_cast<std::uint32_t>(std::time(nullptr)),
            rd(), rd(), rd(), rd()
        };
        std::seed_seq seq(entropy.begin(), entropy.end());
        
        // Create a new generator with the seed sequence
        std::mt19937 global_gen(seq);
        
        // Store the generator in a global context or pass it where needed
        Utils::set_random_generator(global_gen);

#ifdef CUDA_AVAILABLE
        // Initialize CUDA
        if (cudaSetDevice(0) != cudaSuccess) {
            std::cerr << "Failed to initialize CUDA device" << std::endl;
            return 1;
        }

        // Create CUDA stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        std::cout << "CUDA is available" << std::endl;
#else
        std::cout << "CUDA is not available" << std::endl;
#endif

        // Load training data first
        auto training_pairs = Utils::create_training_data();
        std::cout << "Loaded " << training_pairs.size() << " training pairs" << std::endl;
        
        // Initialize tokenizer with config
        std::cout << "\nInitializing tiktoken with encoding: gpt2" << std::endl;
        tokenizer = std::make_unique<Tokenizer>("gpt2");
        
        try {
            tokenizer->initialize("cl100k_base");
            std::cout << "Initialized tokenizer with default vocabulary size: " << tokenizer->vocab_size() << std::endl;
            std::cout << "Using custom vocabulary size from data: " << custom_vocab_size << std::endl;
            
            // Set the custom vocabulary size in the tokenizer
            tokenizer->set_vocab_size(custom_vocab_size);
            
            // Verify the vocabulary size was properly set
            size_t current_vocab_size = tokenizer->vocab_size();
            if (current_vocab_size != custom_vocab_size) {
                throw std::runtime_error("Failed to set custom vocabulary size. Expected: " + 
                                       std::to_string(custom_vocab_size) + ", Got: " + 
                                       std::to_string(current_vocab_size));
            }
            std::cout << "Successfully updated tokenizer vocabulary size to: " << current_vocab_size << std::endl;
            
            // Override config vocabulary size with our custom size
            config.vocab_size = custom_vocab_size;
            config.tokenizer.vocab_size = custom_vocab_size;
            std::cout << "Updated config vocabulary sizes:"
                      << "\n- config.vocab_size: " << config.vocab_size
                      << "\n- config.tokenizer.vocab_size: " << config.tokenizer.vocab_size << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize tokenizer: " << e.what() << std::endl;
            return 1;
        }

        // Initialize model with updated config
        std::cout << "\nInitializing transformer with custom vocabulary size: " << config.vocab_size << std::endl;
        Transformer transformer(config);
        std::cout << "\nTransformer initialized with language model head" << std::endl << std::flush;

        // Print vocabulary mappings
        std::cout << "\nPrinting vocabulary mappings:\n";
        tokenizer->print_vocabulary_mappings();

        // Training parameters
        const size_t checkpoint_frequency =
            config.paths.checkpoint_frequency; // Save checkpoint every 2 epochs

        // Initialize model saver
        ModelSaver model_saver;
        std::string save_directory = config.paths.save_directory;
        std::string model_name = config.paths.model_name;

        // After transformer initialization but before training loop
        if (config.load_from_checkpoint) {
            std::cout << "Attempting to load checkpoint from: " << config.checkpoint_to_load
                      << std::endl;

            try {
                if (!std::filesystem::exists(config.checkpoint_to_load)) {
                    std::cout << "Warning: Checkpoint file does not exist: "
                              << config.checkpoint_to_load << std::endl;
                    std::cout << "Proceeding with training from scratch..." << std::endl;
                } else {
                    // Attempt to load the checkpoint
                    if (!model_saver.loadCheckpoint(transformer, config.checkpoint_to_load)) {
                        std::cerr << "Warning: Failed to load checkpoint from: "
                                  << config.checkpoint_to_load << std::endl;
                        std::cout << "Proceeding with training from scratch..." << std::endl;
                    } else {
                        // Extract epoch number from checkpoint filename
                        std::string filename =
                            std::filesystem::path(config.checkpoint_to_load).filename().string();
                        size_t epoch_pos = filename.find("epoch_");
                        if (epoch_pos != std::string::npos) {
                            std::string epoch_str = filename.substr(epoch_pos + 6);
                            size_t end_pos = epoch_str.find_first_not_of("0123456789");
                            epoch_str = epoch_str.substr(0, end_pos);
                            global_step =
                                std::stoul(epoch_str) * (training_pairs.size() / config.batch_size);
                        }

                        std::cout << "Successfully loaded checkpoint. Resuming from global step: "
                                  << global_step << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Error during checkpoint loading: " << e.what() << std::endl;
                std::cout << "Proceeding with training from scratch..." << std::endl;
            }
        }

        // Load validation data
        auto validation_data = Utils::load_validation_data();
        std::cout << "Loaded " << validation_data.size() << " validation examples\n";

        // Combine training and validation data for cross-validation
        std::vector<std::pair<std::string, std::string>> all_data;
        all_data.insert(all_data.end(), training_pairs.begin(), training_pairs.end());
        all_data.insert(all_data.end(), validation_data.begin(), validation_data.end());
        
        // Perform initial cross-validation to establish baseline
        const size_t NUM_FOLDS = 5;
        const float EARLY_STOPPING_THRESHOLD = 1.5f;  // Ratio of val_loss/train_loss that triggers early stopping
        float initial_cv_loss = Utils::perform_cross_validation(transformer, *tokenizer, all_data, 
                                                              NUM_FOLDS, EARLY_STOPPING_THRESHOLD);
        std::cout << "Initial cross-validation loss: " << initial_cv_loss << std::endl;

        // Update any hardcoded token references
        int pad_id = tokenizer->get_pad_token_id();    // Should be 0
        std::cout << "pad_id: " << pad_id << std::endl;
        int unk_id = tokenizer->get_unk_token_id();    // Should be 1
        std::cout << "unk_id: " << unk_id << std::endl;
        int bos_id = tokenizer->get_bos_token_id();    // Should be 2
        std::cout << "bos_id: " << bos_id << std::endl;
        int eos_id = tokenizer->get_eos_token_id();    // Should be 3
        std::cout << "eos_id: " << eos_id << std::endl;
        int mask_id = tokenizer->get_mask_token_id();  // Should be 4
        std::cout << "mask_id: " << mask_id << std::endl;
        std::cout << "epochs: " << config.num_epochs << std::endl;

        float best_cv_loss = initial_cv_loss;
        size_t epochs_without_improvement = 0;
        const size_t PATIENCE = 3;

        // After loading data but before training loop
        std::cout << "\nStarting hyperparameter tuning phase..." << std::endl;
        
        // Initialize hyperparameter tuner
        HyperparameterRanges ranges;  // Now defined
        HyperparameterTuner tuner(ranges, 20, 5);  // Now defined
        
        // Run hyperparameter tuning
        std::cout << "Running hyperparameter tuning with " << training_pairs.size() 
                  << " training examples..." << std::endl;
        auto tuning_results = tuner.tune(training_pairs, *tokenizer);
        
        // Get best configuration
        auto best_config = tuner.get_best_config();
        
        // Save tuning results
        std::string tuning_results_path = save_directory + "/tuning_results.json";
        tuner.save_results(tuning_results_path);
        
        std::cout << "\nHyperparameter tuning complete!" << std::endl;
        std::cout << "Best configuration achieved validation loss: " 
                  << tuning_results[0].mean_validation_loss << std::endl;
        
        // Update transformer config with best hyperparameters
        config = best_config.to_transformer_config();
        
        // Reinitialize transformer with best config
        transformer = Transformer(config);
        std::cout << "Reinitialized transformer with best hyperparameters" << std::endl;
        
        // Update training parameters from best config
        const float initial_lr = best_config.initial_lr;
        const float peak_lr = best_config.peak_lr;
        const size_t warmup_steps = best_config.warmup_steps;
        const float decay_factor = best_config.decay_factor;
        const float gradient_clip_threshold = best_config.gradient_clip_threshold;
        const size_t early_stopping_patience = best_config.early_stopping_patience;
        const float early_stopping_threshold = best_config.early_stopping_threshold;
        
        std::cout << "\nStarting main training with best hyperparameters..." << std::endl;

        for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
            std::cout << "Epoch " << epoch + 1 << "/" << config.num_epochs << "\n";
            float epoch_loss = 0.0f;
            size_t total_batches =
                (training_pairs.size() + config.batch_size - 1) / config.batch_size;

            // Process batches
            for (size_t batch = 0; batch < total_batches; ++batch) {
                metrics.start_timer("batch_processing");

                // Add random perturbations to weights before processing batch
                reinitialize_batch_weights(transformer, config, global_step);

                size_t start_idx = batch * config.batch_size;
                size_t end_idx = std::min(start_idx + config.batch_size, training_pairs.size());
                size_t current_batch_size = end_idx - start_idx;

                // Find maximum sequence length in this batch
                size_t max_seq_len = 0;
                for (size_t j = start_idx; j < end_idx; ++j) {
                    const auto& [input_str, target_str] = training_pairs[j];
                    std::vector<int> input_tokens = tokenizer->encode(input_str);
                    std::vector<int> target_tokens = tokenizer->encode(target_str);
                    // Consider both input and target sequence lengths
                    max_seq_len = std::max({max_seq_len, input_tokens.size(), target_tokens.size()});
                }
                std::cout << "\n=== Processing Batch " << batch + 1 << " ===\n";

                // Create batch with validation
                std::vector<std::vector<int>> input_batch;
                std::vector<std::vector<int>> target_batch;  // Rename from target_tokens

                // Fill and validate batch with padding
                bool batch_valid = true;
                for (size_t j = start_idx; j < end_idx; ++j) {
                    const auto& [input_str, target_str] = training_pairs[j];

                    // Preprocess text
                    std::string processed_input = input_str;
                    std::string processed_target = target_str;
                    tokenizer->preprocess_text(processed_input);
                    tokenizer->preprocess_text(processed_target);

                    // Encode using appropriate tokenizer
                    std::vector<int> input_tokens = tokenizer->encode(processed_input);
                    std::vector<int> target_tokens = tokenizer->encode(processed_target);

                    // Validate sequences
                    if (!Utils::validate_input_sequence(input_tokens, tokenizer->vocab_size()) ||
                        !Utils::validate_input_sequence(target_tokens, tokenizer->vocab_size())) {
                        std::cerr << "Invalid sequence at position " << j << std::endl;
                        batch_valid = false;
                        break;
                    }

                    // Pad sequences to max_seq_len
                    while (input_tokens.size() < max_seq_len) {
                        input_tokens.push_back(tokenizer->get_pad_token_id());
                    }
                    while (target_tokens.size() < max_seq_len) {  // Add padding for target tokens
                        target_tokens.push_back(tokenizer->get_pad_token_id());
                    }

                    input_batch.push_back(input_tokens);
                    target_batch.push_back(target_tokens);  // Use target_batch instead of target_tokens
                }

                if (!batch_valid)
                    continue; // Skip invalid batches

                std::cout << "Input batch size: " << input_batch.size() << " sequences\n";
                std::cout << "Target batch size: " << target_batch.size() << " sequences\n";

                // First collect valid sequences
                std::vector<std::vector<int>> valid_input_batch;
                std::vector<std::vector<int>> valid_target_batch;

                for (size_t i = 0; i < input_batch.size(); i++) {
                    const auto& input_sequence = input_batch[i];
                    const auto& target_sequence = target_batch[i];
                    
                    if (input_sequence.size() != max_seq_len) {
                        std::cerr << "Error: Input sequence length mismatch. Expected " << max_seq_len 
                                  << " but got " << input_sequence.size() << std::endl;
                        continue;
                    }
                    
                    if (target_sequence.size() != max_seq_len) {
                        std::cerr << "Error: Target sequence length mismatch. Expected " << max_seq_len 
                                  << " but got " << target_sequence.size() << std::endl;
                        continue;
                    }
                    
                    valid_input_batch.push_back(input_sequence);
                    valid_target_batch.push_back(target_sequence);
                }

                if (valid_input_batch.empty()) {
                    std::cerr << "Error: No valid sequences in batch\n";
                    continue;
                }

                // Create target distribution for entire batch using only valid sequences
                Matrix target_distribution = Utils::create_batch_target_distribution(
                    valid_target_batch, *tokenizer, custom_vocab_size, max_seq_len);

                // Process the batch as a single sequence
                std::vector<int> flattened_batch;
                flattened_batch.reserve(valid_input_batch.size() * max_seq_len);

                // Flatten the batch into a single sequence
                for (const auto& sequence : valid_input_batch) {
                    flattened_batch.insert(flattened_batch.end(), sequence.begin(), sequence.end());
                }
                std::cout << "Flattened batch size: " << flattened_batch.size() << " tokens\n";

                // Forward pass through the model
                Matrix logits = transformer.forward(flattened_batch, "", *tokenizer);

                // Compute batch loss
                float batch_loss = Utils::compute_batch_loss(logits, target_distribution, *tokenizer);

                // Compute gradient norm
                Matrix loss_gradients = Utils::compute_loss_gradient(logits, target_distribution);
                float grad_norm = 0.0f;
                for (size_t i = 0; i < loss_gradients.size(); i++) {
                    grad_norm += loss_gradients.data()[i] * loss_gradients.data()[i];
                }
                grad_norm = std::sqrt(grad_norm);

                // Calculate learning rate using tuned parameters
                float current_lr;
                if (global_step < warmup_steps) {
                    // Linear warmup
                    current_lr = initial_lr + (peak_lr - initial_lr) * (float)global_step / warmup_steps;
                } else {
                    // Cosine decay with tuned decay factor
                    float steps_after_warmup = global_step - warmup_steps;
                    float decay = std::pow(decay_factor, steps_after_warmup / 1000.0f);  // Decay every 1000 steps
                    current_lr = peak_lr * decay;
                }
                
                // Ensure learning rate doesn't go below initial_lr
                current_lr = std::max(current_lr, initial_lr);

                // Apply gradient clipping
                if (grad_norm > gradient_clip_threshold) {
                    float scale = gradient_clip_threshold / (grad_norm + 1e-6f);
                    for (size_t i = 0; i < loss_gradients.size(); i++) {
                        loss_gradients.data()[i] *= scale;
                    }
                    grad_norm = gradient_clip_threshold;
                }

                // Print training statistics
                std::cout << "\nTraining Statistics (Batch " << batch + 1 << "):" << std::endl;
                std::cout << "Batch Loss: " << batch_loss << std::endl;
                std::cout << "Gradient Norm: " << grad_norm << std::endl;
                std::cout << "Learning Rate: " << current_lr << std::endl;

                // Generate predictions every 2 batches
                if ((batch + 1) % 2 == 0) {
                    // Make predictions after each batch
                    generate_predictions(transformer, "I go to", tokenizer.get());
                    generate_predictions(transformer, "The weather is", tokenizer.get());
                    generate_predictions(transformer, "I want to", tokenizer.get());
                    generate_predictions(transformer, "The cat", tokenizer.get());
                    generate_predictions(transformer, "She likes to", tokenizer.get());
                }

                // Print progress and metrics every 10 batches
                if ((batch + 1) % 10 == 0 || batch + 1 == total_batches) {
                    std::cout << "\rBatch " << batch + 1 << "/" << total_batches << " in epoch "
                              << epoch + 1 << " (Loss: " << batch_loss
                              << ", Avg Loss: " << epoch_loss / (batch + 1)
                              << ", LR: " << current_lr << ")" << std::flush;

                    // Print performance metrics
                    metrics.print_metrics();
                }

                // Update tracking variables
                prev_loss = batch_loss;
                epoch_loss += batch_loss;
                global_step++;
                
                metrics.stop_timer("batch_processing");

                // In the training loop, after processing each batch
                for (const auto& tokens : input_batch) {
                    // Add frequency decay for common tokens
                    static const float FREQ_DECAY_RATE = 0.98f;
                    static const float MIN_FREQ = 0.1f;
                    
                    // Get mean frequency for comparison
                    const auto& frequencies = transformer.get_lm_head()->get_token_frequencies();
                    float mean_freq = 0.0f;
                    if (!frequencies.empty()) {
                        mean_freq = std::accumulate(frequencies.begin(), frequencies.end(), 0.0f) / frequencies.size();
                    }
                    
                    // Create a vector of tokens with adjusted frequencies
                    std::vector<int> update_tokens;
                    
                    for (int token : tokens) {
                        if (token < frequencies.size()) {
                            float current_freq = frequencies[token];
                            
                            // Only include tokens that need frequency adjustment
                            if (current_freq > 2.0f * mean_freq) {
                                // Apply decay to very frequent tokens
                                // We'll handle these separately to reduce their frequency
                                continue;
                            } else {
                                // Add tokens that should get frequency boost
                                update_tokens.push_back(token);
                            }
                        }
                    }
                    
                    // Update frequencies for normal tokens
                    if (!update_tokens.empty()) {
                        transformer.get_lm_head()->update_token_frequencies(update_tokens);
                    }
                    
                    // For frequent tokens, we'll need to implement a separate method in LanguageModelHead
                    // to handle frequency decay. For now, we'll skip the decay to fix the compilation error.
                }
            }

            std::cout << "\nCompleted epoch " << epoch + 1 << "/" << config.num_epochs
                      << " (Loss: " << epoch_loss / total_batches << ")" << std::endl;

            // Perform cross-validation every few epochs
            if ((epoch + 1) % 3 == 0) {  // Every 3 epochs
                float cv_loss = Utils::perform_cross_validation(transformer, *tokenizer, all_data, 
                                                              NUM_FOLDS, early_stopping_threshold);
                std::cout << "Cross-validation loss after epoch " << epoch + 1 << ": " << cv_loss << std::endl;
                
                // Early stopping based on cross-validation with tuned parameters
                if (cv_loss < best_cv_loss) {
                    best_cv_loss = cv_loss;
                    epochs_without_improvement = 0;
                    
                    // Save best model
                    std::cout << "New best model found! Saving checkpoint..." << std::endl;
                    if (!model_saver.saveCheckpoint(transformer, save_directory, 
                                                  model_name + "_best", epoch + 1, cv_loss)) {
                        std::cerr << "Failed to save best model checkpoint" << std::endl;
                    }
                } else {
                    epochs_without_improvement++;
                    if (epochs_without_improvement >= early_stopping_patience) {
                        std::cout << "Early stopping triggered after " << epoch + 1 
                                 << " epochs (patience: " << early_stopping_patience << ")" << std::endl;
                        break;
                    }
                }
                
                // Check for significant overfitting using tuned threshold
                float train_val_ratio = cv_loss / (epoch_loss / total_batches);
                if (train_val_ratio > early_stopping_threshold) {
                    std::cout << "Significant overfitting detected (ratio: " << train_val_ratio 
                             << " > threshold: " << early_stopping_threshold << "). Stopping training." << std::endl;
                    break;
                }
            }

            // Save regular checkpoint
            if ((epoch + 1) % checkpoint_frequency == 0) {
                std::cout << "Attempting to save checkpoint to: " << save_directory << "/"
                          << model_name << std::endl;

                // Verify directory exists and is writable
                if (!std::filesystem::exists(save_directory)) {
                    std::cout << "Creating directory: " << save_directory << std::endl;
                    if (!std::filesystem::create_directories(save_directory)) {
                        std::cerr << "Failed to create directory: " << save_directory << std::endl;
                        // Don't exit, just skip checkpoint
                        continue;
                    }
                }

                // Try to save
                if (!model_saver.saveCheckpoint(transformer, save_directory, model_name, epoch + 1,
                                                epoch_loss)) {
                    std::cerr << "Failed to save checkpoint, but continuing training" << std::endl;
                    // Don't exit, just continue training
                }
            }

            // Test prediction on a sample input
            if ((epoch + 1) % 2 == 0) {
                std::cout << "\nTesting generation with " 
                          << (config.tokenizer.use_subword ? "subword" : "regular") 
                          << " tokenization:" << std::endl;
                
                // Test a simple input
                std::string test_input = "I go to";
                std::cout << "\n=== Processing prompt: '" << test_input << "' ===" << std::endl;
                
                // Preprocess input
                std::string processed_input = test_input;
                tokenizer->preprocess_text(processed_input);
                std::vector<int> test_tokens = tokenizer->encode(processed_input);
                
                // Get model prediction
                Matrix test_hidden = transformer.forward(test_tokens, "", *tokenizer);
                Matrix logits = transformer.get_lm_head()->forward(test_hidden);
                
                // For single token prediction, we don't need beam search
                // Just show the top predictions
                std::cout << "\nTop Predictions:\n";
                Utils::print_top_predictions(logits, *tokenizer, transformer, 5);
            }

            if ((epoch + 1) % 5 == 0) { 
                // Cache clearing removed since TiktokenTokenizer doesn't use caching
            }

            // Run validation every 3 epochs
            if ((epoch + 1) % 3 == 0) { // Validate every 3 epochs
                std::cout << "\nRunning validation after epoch " << (epoch + 1) << "...\n";
                float validation_loss =
                    Utils::evaluate_validation(transformer, *tokenizer, validation_data);

                // Log validation results
                std::cout << "Epoch " << (epoch + 1) << " Validation Loss: " << validation_loss
                          << std::endl;
            }
        }

        std::cout << "\nTraining completed!\n";

        // Final prediction test
        std::cout << "\nFinal generation test with " 
                  << (config.tokenizer.use_subword ? "subword" : "regular") 
                  << " tokenization:" << std::endl;
        
        // Test a simple input
        std::string test_input = "I go to";
        std::cout << "\n=== Processing prompt: '" << test_input << "' ===" << std::endl;
        
        // Preprocess input
        std::string processed_input = test_input;
        tokenizer->preprocess_text(processed_input);
        std::vector<int> test_tokens = tokenizer->encode(processed_input);
        
        // Get model prediction
        transformer.set_training(false);  // Set to evaluation mode
        Matrix test_hidden = transformer.forward(test_tokens, "", *tokenizer);
        Matrix logits = transformer.get_lm_head()->forward(test_hidden);
        
        // Show the top predictions
        std::cout << "\nTop Predictions:\n";
        Utils::print_top_predictions(logits, *tokenizer, transformer, 5);

        // Create directories if they don't exist
        std::filesystem::create_directories(save_directory);

        // Save the trained model
        std::cout << "\nSaving final model to " << save_directory << "/" << model_name << "...\n";
        bool save_success = model_saver.saveModel(transformer, save_directory, model_name);
        if (save_success) {
            std::cout << "Successfully saved model to " + save_directory + "/" + model_name
                      << std::endl;
            std::cout << "Model saved successfully!\n";
        } else {
            std::cout << "Failed to save model to " + save_directory + "/" + model_name
                      << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

#ifdef CUDA_AVAILABLE
    cleanup_cuda(); // Cleanup at program end
#endif
    logger.stopLogging();
    std::cout << "exiting main" << std::endl;
    return 0;
}
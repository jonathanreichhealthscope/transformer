#include "../include/main.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include "../include/tokenizer.hpp"
#include "../include/utils.hpp"
#include "../include/phrase_analysis.hpp"
#include "../include/training/training.hpp"  // Include unified training header

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
void generate_predictions(Transformer& transformer, const std::string& input_text, const Tokenizer* tokenizer) {
    std::cout << "\n=== Batch " << global_step << " Predictions for '" << input_text << "' ===" << std::endl;
    
    // Process input
    std::string processed_input = input_text;
    tokenizer->preprocess_text(processed_input);
    std::vector<int> input_tokens = tokenizer->encode(processed_input);
    
    // Generate prediction
    transformer.set_training(false);
    Matrix hidden_states = transformer.forward(input_tokens, input_text, *tokenizer);
    Matrix logits = transformer.get_lm_head()->forward(hidden_states);
    
    // Get probabilities for last token
    Vector last_row = logits.row(logits.rows() - 1);
    Matrix final_logits(1, last_row.size());  // Create 1 x N matrix
    for (size_t i = 0; i < last_row.size(); ++i) {
        final_logits(0, i) = last_row[i];
    }
    std::vector<std::pair<float, int>> token_probs;
    
    // Convert logits to probabilities with softmax
    float max_logit = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < final_logits.cols(); ++i) {
        max_logit = std::max(max_logit, final_logits(0, i));
    }
    
    float sum_exp = 0.0f;
    std::vector<float> exps(final_logits.cols());
    for (size_t i = 0; i < final_logits.cols(); ++i) {
        exps[i] = std::exp(final_logits(0, i) - max_logit);
        sum_exp += exps[i];
    }
    
    // Create probability pairs
    for (size_t i = 0; i < final_logits.cols(); ++i) {
        float prob = exps[i] / sum_exp;
        if (prob > 1e-4) {  // Filter out very low probability tokens
            token_probs.emplace_back(prob, i);
        }
    }
    
    // Sort by probability
    std::sort(token_probs.begin(), token_probs.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Print top predictions
    std::cout << "\nTop 5 predictions:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), token_probs.size()); ++i) {
        int token_id = token_probs[i].second;
        float prob = token_probs[i].first;
        std::string token_text = tokenizer->decode({token_id});
        std::cout << " " << token_text << " (" << (prob * 100) << "%)" << std::endl;
    }
    
    // Print statistics
    size_t nonzero_probs = std::count_if(token_probs.begin(), token_probs.end(),
                                        [](const auto& p) { return p.first > 1e-4; });
    std::cout << "--\nNonzero final: " << nonzero_probs << "/" << token_probs.size() << "\n" << std::endl;
    
    transformer.set_training(true);
}

int main(int argc, char* argv[]) {
    std::cout << "entering main" << std::endl;
    Logger& logger = Logger::getInstance();
    logger.startLogging();

    try {
        // Load configuration
        std::filesystem::path exe_path = std::filesystem::current_path().parent_path();
        std::filesystem::path config_path = exe_path / "config" / "transformer_config.json";
        TransformerConfig config = Utils::load_config(config_path.string());

        // Initialize random seed
        std::srand(static_cast<unsigned int>(std::time(nullptr)));

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
        std::cout << "Initializing tiktoken with encoding: gpt2" << std::endl;
        tokenizer = std::make_unique<Tokenizer>("gpt2");
        
        try {
            tokenizer->initialize();  // Initialize with default encoding
            std::cout << "Initialized tokenizer. Vocabulary size: " 
                      << tokenizer->vocab_size() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize tokenizer: " << e.what() << std::endl;
            return 1;
        }

        // Update vocabulary size in config based on tokenizer
        config.vocab_size = tokenizer->vocab_size();
        std::cout << "Using vocabulary size: " << config.vocab_size << std::endl;

        // Initialize model with updated config
        Transformer transformer(config);
        auto lm_head = std::make_unique<LanguageModelHead>(config.hidden_size, config.vocab_size);

        // Setup advanced components
        TensorCache<Matrix> activation_cache(1024, CacheReplacementPolicy::ARC);
        QuantizationAwareTraining qat(true);
        auto sam_optimizer = std::make_unique<SAM>(0.05f);

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

        // Training loop
        size_t global_step = 0; // Move outside epoch loop
        Matrix last_hidden_states;

        // Load validation data
        auto validation_data = Utils::load_validation_data();
        std::cout << "Loaded " << validation_data.size() << " validation examples\n";

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
        for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
            std::cout << "Epoch " << epoch + 1 << "/" << config.num_epochs << "\n";
            float epoch_loss = 0.0f;
            size_t total_batches =
                (training_pairs.size() + config.batch_size - 1) / config.batch_size;

            // Process batches
            for (size_t batch = 0; batch < total_batches; ++batch) {
                metrics.start_timer("batch_processing");

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
                    valid_target_batch, *tokenizer, config.vocab_size, max_seq_len);

                // Process the batch as a single sequence
                std::vector<int> flattened_batch;
                flattened_batch.reserve(valid_input_batch.size() * max_seq_len);

                // Flatten the batch into a single sequence
                for (const auto& sequence : valid_input_batch) {
                    flattened_batch.insert(flattened_batch.end(), sequence.begin(), sequence.end());
                }
                std::cout << "Flattened batch size: " << flattened_batch.size() << " tokens\n";

                // Forward pass with the flattened batch
                transformer.set_training(true);
                metrics.start_timer("forward_pass");
                Matrix hidden_states = transformer.forward(flattened_batch, "", *tokenizer);
                metrics.stop_timer("forward_pass");

                metrics.record_memory_usage(hidden_states.bytes());

                Matrix logits = lm_head->project_to_vocab(hidden_states);

                // Compute loss and its gradients for all tokens in sequence
                float batch_loss = Utils::compute_batch_loss(logits, target_distribution, *tokenizer);

                // Compute softmax gradients for each token in the sequence
                Matrix loss_gradients = Matrix(logits.rows(), logits.cols(), 0.0f);
                for (size_t i = 0; i < logits.rows(); i++) {
                    // Compute softmax for this token's logits
                    std::vector<float> token_logits;
                    float max_logit = -std::numeric_limits<float>::infinity();

                    for (size_t j = 0; j < config.vocab_size; j++) {
                        float logit = logits(i, j);
                        token_logits.push_back(logit);
                        max_logit = std::max(max_logit, logit);
                    }

                    float sum_exp = 0.0f;
                    std::vector<float> exp_logits(config.vocab_size);

                    for (size_t j = 0; j < config.vocab_size; j++) {
                        exp_logits[j] = std::exp(token_logits[j] - max_logit);
                        sum_exp += exp_logits[j];
                    }

                    // Compute gradients for cross-entropy loss
                    float scaling_factor = 100.0f;  // Base scaling factor
                    size_t non_pad_tokens = 0;

                    // First pass to count non-padding tokens
                    for (size_t j = 0; j < config.vocab_size; j++) {
                        if (target_distribution(i, j) > 0.0f && j != tokenizer->get_pad_token_id()) {
                            non_pad_tokens++;
                        }
                    }

                    // Adjust scaling factor based on token density
                    if (non_pad_tokens > 0) {
                        scaling_factor *= std::sqrt(float(config.vocab_size) / non_pad_tokens);
                    }

                    for (size_t j = 0; j < config.vocab_size; j++) {
                        float softmax_output = exp_logits[j] / sum_exp;
                        float gradient = 0.0f;
                        
                        // Only compute meaningful gradients for non-padding tokens
                        if (j != tokenizer->get_pad_token_id()) {
                            gradient = (softmax_output - target_distribution(i, j)) * scaling_factor;
                            // Clip gradients to reasonable range
                            gradient = std::max(std::min(gradient, 1.0f), -1.0f);
                        }
                        loss_gradients(i, j) = gradient;
                    }
                }

                // Log gradient and token statistics
                float total_grad_magnitude = 0.0f;
                float max_grad_magnitude = 0.0f;
                size_t total_tokens = loss_gradients.rows() * loss_gradients.cols();
                size_t non_padding_tokens = 0;

                for (size_t i = 0; i < loss_gradients.rows(); i++) {
                    for (size_t j = 0; j < loss_gradients.cols(); j++) {
                        float grad_magnitude = std::abs(loss_gradients(i, j));
                        if (grad_magnitude > 0.0f) {  // Count non-zero gradients
                            total_grad_magnitude += grad_magnitude;
                            max_grad_magnitude = std::max(max_grad_magnitude, grad_magnitude);
                            non_padding_tokens++;
                        }
                    }
                }

                // Calculate average only over non-padding tokens
                float avg_grad_magnitude = non_padding_tokens > 0 ? 
                    total_grad_magnitude / non_padding_tokens : 0.0f;

                std::cout << "Gradient Statistics:\n"
                          << "  Average magnitude (non-padding): " << avg_grad_magnitude << "\n"
                          << "  Maximum magnitude: " << max_grad_magnitude << "\n"
                          << "  Non-padding tokens: " << non_padding_tokens << "/" << total_tokens 
                          << " (" << (100.0f * non_padding_tokens / total_tokens) << "%)\n";

                // More dynamic learning rate adjustment
                float loss_ratio = 1.0f;  // Default to neutral ratio
                if (std::isfinite(batch_loss) && std::isfinite(prev_loss) && prev_loss > 0.0f) {
                    // Only compute ratio if both losses are valid and prev_loss is positive
                    const float min_loss = 1e-6f;  // Larger epsilon for more stability
                    loss_ratio = batch_loss / std::max(prev_loss, min_loss);
                    
                    // Clamp ratio to reasonable range
                    const float max_ratio = 10.0f;
                    loss_ratio = std::max(0.1f, std::min(loss_ratio, max_ratio));
                }

                float grad_scale = std::min(avg_grad_magnitude, 1.0f);  // Scale based on gradient magnitude
                
                // Adjust thresholds based on training progress
                float upper_threshold = 1.05f - (0.01f * std::min(global_step / 1000.0f, 0.04f));
                float lower_threshold = 0.95f + (0.01f * std::min(global_step / 1000.0f, 0.04f));
                
                if (loss_ratio > upper_threshold) {
                    // More aggressive decrease when loss increases significantly
                    float decrease_factor = 0.8f + (0.15f * grad_scale);  // Between 0.8 and 0.95
                    learning_rate *= decrease_factor;
                    std::cout << "Decreasing learning rate by factor " << decrease_factor 
                              << " (loss ratio: " << loss_ratio << ")\n";
                } else if (loss_ratio < lower_threshold) {
                    // More aggressive increase when loss decreases significantly
                    float increase_factor = 1.2f - (0.15f * grad_scale);  // Between 1.05 and 1.2
                    learning_rate *= increase_factor;
                    std::cout << "Increasing learning rate by factor " << increase_factor 
                              << " (loss ratio: " << loss_ratio << ")\n";
                }
                
                // Wider bounds for learning rate
                float min_lr = 1e-6f;
                float max_lr = 5e-2f;
                if (global_step < 100) {  // Allow larger learning rates early in training
                    max_lr = 1e-1f;
                }
                
                learning_rate = std::max(min_lr, std::min(learning_rate, max_lr));
                
                std::cout << "Learning rate adjusted to: " << learning_rate 
                          << " (loss ratio: " << loss_ratio << ")\n";

                // Backpropagate through the model
                Matrix lm_head_gradients = lm_head->backward(loss_gradients);
                std::cout << "lm_head_gradients shape: " << lm_head_gradients.rows() << "x" << lm_head_gradients.cols() << std::endl;
                transformer.backward(lm_head_gradients, flattened_batch, learning_rate);
                std::cout << "lm_head_gradients shape after backward: " << lm_head_gradients.rows() << "x" << lm_head_gradients.cols() << std::endl;
                // Update tracking variables
                prev_loss = batch_loss;
                epoch_loss += batch_loss;
                global_step++;
                
                metrics.stop_timer("batch_processing");

                // Make predictions after each batch
                generate_predictions(transformer, "I go to", tokenizer.get());
                generate_predictions(transformer, "The weather is", tokenizer.get());
                generate_predictions(transformer, "I want to", tokenizer.get());
                generate_predictions(transformer, "The cat", tokenizer.get());
                generate_predictions(transformer, "She likes to", tokenizer.get());

                // Print progress and metrics every 10 batches
                if ((batch + 1) % 10 == 0 || batch + 1 == total_batches) {
                    std::cout << "\rBatch " << batch + 1 << "/" << total_batches << " in epoch "
                              << epoch + 1 << " (Loss: " << batch_loss
                              << ", Avg Loss: " << epoch_loss / (batch + 1)
                              << ", LR: " << learning_rate << ")" << std::flush;

                    // Print performance metrics
                    metrics.print_metrics();
                }

                // In the training loop, after processing each batch
                for (const auto& tokens : input_batch) {
                    lm_head->update_token_frequencies(tokens);
                }
            }

            std::cout << "\nCompleted epoch " << epoch + 1 << "/" << config.num_epochs
                      << " (Loss: " << epoch_loss / total_batches << ")" << std::endl;

            // Save checkpoint
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
                Matrix logits = lm_head->project_to_vocab(test_hidden);
                
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
        Matrix logits = lm_head->project_to_vocab(test_hidden);
        
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
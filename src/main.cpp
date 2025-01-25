#include "../include/main.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include "../include/tokenizer.hpp"

// Add necessary forward declarations and structures
std::unique_ptr<Tokenizer> tokenizer;
PerformanceMetrics metrics; // Single definition of the global metrics variable

// Configuration constants
const float INITIAL_LEARNING_RATE = 0.001f;
float learning_rate = INITIAL_LEARNING_RATE;
float prev_loss = std::numeric_limits<float>::max();
size_t global_step = 0;

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
                float batch_loss = Utils::compute_batch_loss(logits, target_distribution);

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
                float loss_ratio = batch_loss / (prev_loss + 1e-10f);
                float grad_scale = std::min(avg_grad_magnitude, 1.0f);  // Scale based on gradient magnitude
                
                // Adjust thresholds based on training progress
                float upper_threshold = 1.05f - (0.01f * std::min(global_step / 1000.0f, 0.04f));  // Starts at 1.05, decreases to 1.01
                float lower_threshold = 0.95f + (0.01f * std::min(global_step / 1000.0f, 0.04f));  // Starts at 0.95, increases to 0.99
                
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
                transformer.backward(lm_head_gradients, flattened_batch, learning_rate);

                // Update tracking variables
                prev_loss = batch_loss;
                epoch_loss += batch_loss;
                global_step++;

                metrics.stop_timer("batch_processing");

                // Make predictions after each batch
                std::string test_input = "I go to";
                std::string processed_input = test_input;
                tokenizer->preprocess_text(processed_input);
                std::vector<int> test_tokens = tokenizer->encode(processed_input);
                
                // Get model prediction (in evaluation mode)
                transformer.set_training(false);
                Matrix test_hidden = transformer.forward(test_tokens, test_input, *tokenizer);
                Matrix pred_logits = lm_head->project_to_vocab(test_hidden);
                transformer.set_training(true);  // Set back to training mode
                
                // Show the top predictions
                std::cout << "\n=== Batch " << batch + 1 << " Predictions for '" << test_input << "' ===\n";
                Utils::print_top_predictions(pred_logits, *tokenizer, transformer, 5);
                std::cout << "================================================\n";

                // Test additional queries
                std::vector<std::string> additional_queries = {
                    "The weather is",
                    "I want to",
                    "The cat",
                    "She likes to"
                };

                for (const auto& query : additional_queries) {
                    processed_input = query;
                    tokenizer->preprocess_text(processed_input);
                    test_tokens = tokenizer->encode(processed_input);
                    
                    transformer.set_training(false);
                    test_hidden = transformer.forward(test_tokens, query, *tokenizer);
                    pred_logits = lm_head->project_to_vocab(test_hidden);
                    transformer.set_training(true);
                    
                    std::cout << "\n=== Batch " << batch + 1 << " Predictions for '" << query << "' ===\n";
                    Utils::print_top_predictions(pred_logits, *tokenizer, transformer, 5);
                    std::cout << "================================================\n";
                }

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
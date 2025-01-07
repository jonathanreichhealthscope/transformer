#include "../include/main.hpp"
#include <random>
#include <nlohmann/json.hpp>
#include <fstream>

// Add necessary forward declarations and structures
std::unique_ptr<Tokenizer> tokenizer;
PerformanceMetrics metrics;  // Single definition of the global metrics variable

// Configuration constants
const float INITIAL_LEARNING_RATE = 0.001f;
const float MIN_LEARNING_RATE = 1e-6f;
const float MAX_LEARNING_RATE = 0.1f;
const float GRADIENT_CLIP_THRESHOLD = 1.0f;
const float LOSS_SPIKE_THRESHOLD = 1.5f;
const size_t WARMUP_STEPS = 100;
float learning_rate = INITIAL_LEARNING_RATE;
float prev_loss = std::numeric_limits<float>::max();
size_t global_step = 0;

// Define the special character map (definition)
const std::unordered_map<char, std::string> SPECIAL_CHAR_MAP = {
    {'\n', "<newline>"},
    {'\t', "<tab>"},
    {'.', "<period>"},
    {'!', "<exclamation>"},
    {'?', "<question>"},
    {',', "<comma>"}
};

int main(int argc, char *argv[]) {
    std::cout << "entering main" << std::endl;
    Logger &logger = Logger::getInstance();
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
#endif
        // Initialize tokenizer first to get vocab size
        tokenizer = std::make_unique<Tokenizer>();
        tokenizer->print_vocabulary_mappings();
        tokenizer->clear_cache();
        
        // Get vocabulary size from the tokenizer
        size_t actual_vocab_size = tokenizer->vocab_size();
        std::cout << "Actual vocabulary size: " << actual_vocab_size << std::endl;

        // Only update vocab size from tokenizer, keep other settings from config file
        config.vocab_size = actual_vocab_size;
        // Only compute head_dim as it depends on other config values
        config.head_dim = config.hidden_size / config.num_heads;

        std::cout << "Using configuration from file:\n"
                  << "- Hidden size: " << config.hidden_size << "\n"
                  << "- Number of heads: " << config.num_heads << "\n"
                  << "- Number of layers: " << config.num_layers << "\n"
                  << "- Using Flash Attention: " << config.use_flash_attention << "\n"
                  << "- Using GQA: " << config.use_gqa << "\n"
                  << "- Using RoPE: " << config.use_rope << "\n"
                  << "- Using Sliding Window: " << config.use_sliding_window << "\n";

        // Initialize components
        Transformer transformer(config);
        size_t vocab_size = tokenizer->vocab_size();
        std::cout << "Actual vocabulary size from tokenizer: " << vocab_size << std::endl;

        // Initialize language model head with correct vocab size
        auto lm_head = std::make_unique<LanguageModelHead>(config.hidden_size, vocab_size);

        // Setup advanced components
        TensorCache<Matrix> activation_cache(1024, CacheReplacementPolicy::ARC);
        QuantizationAwareTraining qat(true);
        auto sam_optimizer = std::make_unique<SAM>(0.05f);

        // Print and verify vocabulary mappings
        std::cout << "\nVerifying vocabulary mappings:\n";
        tokenizer->print_vocabulary_mappings();

        if (!tokenizer->verify_mappings()) {
          std::cerr << "Error: Vocabulary mappings are inconsistent!\n";
          return 1;
        }

        // Get training data
        std::vector<std::pair<std::string, std::string>> training_data = Utils::create_training_data();
        
        // Preprocess the training data (convert to lowercase)
        training_data = TextPreprocessor::preprocess_training_data(training_data);
        
        // Analyze token mappings
        Utils::analyze_token_mappings(training_data, *tokenizer);
        
        // Print vocabulary for inspection
        std::cout << "\n=== Full Vocabulary Mapping ===\n";
        tokenizer->print_vocabulary_mappings();
        std::cout << "\n";
        // Training parameters
        const size_t checkpoint_frequency = config.paths.checkpoint_frequency; // Save checkpoint every 2 epochs

        // Initialize model saver
        ModelSaver model_saver;
        std::string save_directory = config.paths.save_directory;
        std::string model_name = config.paths.model_name;

        // After transformer initialization but before training loop
        if (config.load_from_checkpoint) {
            std::cout << "Attempting to load checkpoint from: " << config.checkpoint_to_load << std::endl;
            
            try {
                if (!std::filesystem::exists(config.checkpoint_to_load)) {
                    std::cout << "Warning: Checkpoint file does not exist: " << config.checkpoint_to_load << std::endl;
                    std::cout << "Proceeding with training from scratch..." << std::endl;
                } else {
                    // Attempt to load the checkpoint
                    if (!model_saver.loadCheckpoint(transformer, config.checkpoint_to_load)) {
                        std::cerr << "Warning: Failed to load checkpoint from: " << config.checkpoint_to_load << std::endl;
                        std::cout << "Proceeding with training from scratch..." << std::endl;
                    } else {
                        // Extract epoch number from checkpoint filename
                        std::string filename = std::filesystem::path(config.checkpoint_to_load).filename().string();
                        size_t epoch_pos = filename.find("epoch_");
                        if (epoch_pos != std::string::npos) {
                            std::string epoch_str = filename.substr(epoch_pos + 6);
                            size_t end_pos = epoch_str.find_first_not_of("0123456789");
                            epoch_str = epoch_str.substr(0, end_pos);
                            global_step = std::stoul(epoch_str) * (training_data.size() / config.batch_size);
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
        size_t global_step = 0;  // Move outside epoch loop
        Matrix last_hidden_states;

        // Load validation data
        auto validation_data = Utils::load_validation_data();
        std::cout << "Loaded " << validation_data.size() << " validation examples\n";

        for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
            std::cout << "Epoch " << epoch + 1 << "/" << config.num_epochs << "\n";
            float epoch_loss = 0.0f;
            size_t total_batches = (training_data.size() + config.batch_size - 1) / config.batch_size;
            
            // Process batches
            for (size_t batch = 0; batch < total_batches; ++batch) {
                metrics.start_timer("batch_processing");
                
                size_t start_idx = batch * config.batch_size;
                size_t end_idx = std::min(start_idx + config.batch_size, training_data.size());
                size_t current_batch_size = end_idx - start_idx;
                
                // Create batch with validation
                std::vector<std::vector<int>> input_batch;
                std::vector<std::vector<int>> target_tokens;
                
                // Find maximum sequence length in this batch
                size_t max_seq_len = 0;
                for (size_t j = start_idx; j < end_idx; ++j) {
                    const auto& [input_str, target_str] = training_data[j];
                    std::vector<int> input_tokens = tokenizer->encode(input_str);
                    max_seq_len = std::max(max_seq_len, input_tokens.size());
                }
                
                // Fill and validate batch with padding
                bool batch_valid = true;
                for (size_t j = start_idx; j < end_idx; ++j) {
                    const auto& [input_str, target_str] = training_data[j];
                    
                    // Preprocess both input and target
                    std::string processed_input = input_str;
                    std::string processed_target = target_str;
                    
                    tokenizer->preprocess_text(processed_input);
                    tokenizer->preprocess_text(processed_target);
                    
                    std::vector<int> input_tokens = tokenizer->encode(processed_input);
                    std::vector<int> curr_target_tokens = tokenizer->encode(processed_target);
                    
                    // Validate sequences
                    if (!Utils::validate_input_sequence(input_tokens, config.vocab_size) || 
                        !Utils::validate_input_sequence(curr_target_tokens, config.vocab_size)) {
                        std::cerr << "Invalid sequence at position " << j << std::endl;
                        batch_valid = false;
                        break;
                    }
                    
                    // Pad sequences to max_seq_len
                    while (input_tokens.size() < max_seq_len) {
                        input_tokens.push_back(tokenizer->get_pad_token_id());
                    }
                    
                    input_batch.push_back(input_tokens);
                    target_tokens.push_back(curr_target_tokens);
                }
                
                if (!batch_valid) continue;  // Skip invalid batches
                
                // Create target distribution for entire batch
                Matrix target_distribution = Utils::create_batch_target_distribution(target_tokens, *tokenizer, config.vocab_size);
                
                // Process the batch as a single sequence
                std::vector<int> flattened_batch;
                flattened_batch.reserve(current_batch_size * max_seq_len);
                
                // Flatten the batch into a single sequence
                for (const auto& sequence : input_batch) {
                    flattened_batch.insert(flattened_batch.end(), sequence.begin(), sequence.end());
                }
                
                // Forward pass with the flattened batch
                transformer.set_training(true);
                metrics.start_timer("forward_pass");
                Matrix hidden_states = transformer.forward(flattened_batch);
                metrics.stop_timer("forward_pass");
                
                metrics.record_memory_usage(hidden_states.bytes());
                
                Matrix logits = lm_head->project_to_vocab(hidden_states);
                
                // Take only the last token's logits for each sequence in the batch
                Matrix final_logits(current_batch_size, config.vocab_size);
                for (size_t i = 0; i < current_batch_size; i++) {
                    size_t seq_end_idx = (i + 1) * max_seq_len - 1;
                    for (size_t j = 0; j < config.vocab_size; j++) {
                        if (seq_end_idx < logits.rows()) {
                            final_logits(i, j) = logits(seq_end_idx, j);
                        }
                    }
                }
                
                // Compute loss and its gradients
                float batch_loss = Utils::compute_batch_loss(final_logits, target_distribution);
                
                // Compute softmax gradients for each sequence in the batch
                Matrix loss_gradients = Matrix(logits.rows(), logits.cols(), 0.0f);
                for (size_t i = 0; i < current_batch_size; i++) {
                    size_t seq_end_idx = (i + 1) * max_seq_len - 1;
                    if (seq_end_idx >= logits.rows()) continue;
                    
                    // Compute softmax for this sequence's logits
                    std::vector<float> sequence_logits;
                    float max_logit = -std::numeric_limits<float>::infinity();
                    
                    for (size_t j = 0; j < config.vocab_size; j++) {
                        float logit = logits(seq_end_idx, j);
                        sequence_logits.push_back(logit);
                        max_logit = std::max(max_logit, logit);
                    }
                    
                    float sum_exp = 0.0f;
                    std::vector<float> exp_logits(config.vocab_size);
                    
                    for (size_t j = 0; j < config.vocab_size; j++) {
                        exp_logits[j] = std::exp(sequence_logits[j] - max_logit);
                        sum_exp += exp_logits[j];
                    }
                    
                    // Compute gradients for cross-entropy loss
                    for (size_t j = 0; j < config.vocab_size; j++) {
                        float softmax_output = exp_logits[j] / sum_exp;
                        loss_gradients(seq_end_idx, j) = 
                            (softmax_output - target_distribution(i, j)) / current_batch_size;
                    }
                }
                
                // Update learning rate based on loss
                float loss_ratio = batch_loss / (prev_loss + 1e-10f);
                learning_rate = Utils::adjust_learning_rate(learning_rate, loss_ratio, global_step);
                
                // Backpropagate through the model
                Matrix lm_head_gradients = lm_head->backward(loss_gradients);
                transformer.backward(lm_head_gradients, flattened_batch, learning_rate);
                
                // Update tracking variables
                prev_loss = batch_loss;
                epoch_loss += batch_loss;
                global_step++;
                
                metrics.stop_timer("batch_processing");
                
                // Print progress and metrics every 10 batches
                if ((batch + 1) % 10 == 0 || batch + 1 == total_batches) {
                    std::cout << "\rBatch " << batch + 1 << "/" << total_batches 
                             << " in epoch " << epoch + 1 
                             << " (Loss: " << batch_loss 
                             << ", Avg Loss: " << epoch_loss/(batch+1)
                             << ", LR: " << learning_rate 
                             << ")" << std::flush;
                    
                    // Print performance metrics
                    metrics.print_metrics();
                }

                // In the training loop, after processing each batch
                for (const auto& tokens : input_batch) {
                    lm_head->update_token_frequencies(tokens);
                }
            }
            
            std::cout << "\nCompleted epoch " << epoch + 1 << "/" << config.num_epochs 
                      << " (Loss: " << epoch_loss/total_batches << ")" << std::endl;
            
            // Save checkpoint
            if ((epoch + 1) % checkpoint_frequency == 0) {
                if (!model_saver.saveCheckpoint(transformer, save_directory, model_name,
                                                epoch + 1, epoch_loss)) {
                    std::cout << "Failed to save checkpoint" << std::endl;
                    return 1;
                }
            }

            // Test prediction on a sample input
            if ((epoch + 1) % 2 == 0) {
                // Test multiple different contexts
                std::vector<std::string> test_inputs = {
                    "I go to",                  
                    "Surgeons operate in the",  
                    "Athletes train in the",    
                    "Musicians perform in the", // Entertainment context
                    "Students research in the", // Educational context
                    "Chefs cook in the",        // Culinary context
                    "Artists create in the",    // Creative context
                    "Engineers work in the",    // Technical context
                    "Lawyers practice in the",  // Legal context
                    "Teachers instruct in the", // Educational context
                    "Scientists experiment in", // Research context
                    "Pilots fly through the",   // Aviation context
                    "Dancers rehearse in the",  // Performance context
                    "Writers compose in the",   // Literary context
                    "Mechanics repair in the"   // Automotive context
                };

                for (const auto &test_input : test_inputs) {
                    std::cout << "\nTesting: '" << test_input << "'\n";
                    // Add preprocessing step
                    std::string processed_input = test_input;
                    tokenizer->preprocess_text(processed_input);
                    std::vector<int> test_tokens = tokenizer->encode(processed_input);
                    Matrix test_hidden = transformer.forward(test_tokens);
                    Matrix test_logits = lm_head->forward(test_hidden);
                    Utils::print_top_predictions(test_logits, *tokenizer, 5);
                }
            }

            if ((epoch + 1) % 5 == 0) {  // Clear cache every 5 epochs
                tokenizer->clear_cache();
            }

            // Run validation every 3 epochs
            if ((epoch + 1) % 3 == 0) {  // Validate every 3 epochs
                std::cout << "\nRunning validation after epoch " << (epoch + 1) << "...\n";
                float validation_loss = Utils::evaluate_validation(transformer, *tokenizer, validation_data);
                
                // Log validation results
                std::cout << "Epoch " << (epoch + 1) << " Validation Loss: " << validation_loss << std::endl;
            }
        }

        std::cout << "\nTraining completed!\n";

        // Create directories if they don't exist
        std::filesystem::create_directories(save_directory);

        // Save the trained model
        std::cout << "\nSaving final model to " << save_directory << "/"
                  << model_name << "...\n";
        bool save_success =
            model_saver.saveModel(transformer, save_directory, model_name);
        if (save_success) {
            std::cout << "Successfully saved model to " + save_directory + "/" +
                       model_name << std::endl;
            std::cout << "Model saved successfully!\n";
        } else {
            std::cout << "Failed to save model to " + save_directory + "/" + model_name << std::endl;
            return 1;
        }

    } catch (const std::exception &e) {
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
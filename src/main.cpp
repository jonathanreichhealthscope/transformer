#include "../include/main.hpp"

// Add necessary forward declarations and structures
std::unique_ptr<Tokenizer> tokenizer;

// Define training example structure
struct TrainingExample {
    std::vector<int> input_tokens;
    Matrix target;
};

// Configuration constants
const float INITIAL_LEARNING_RATE = 0.001f;
const float MIN_LEARNING_RATE = 1e-6f;
const float MAX_LEARNING_RATE = 0.1f;
const float GRADIENT_CLIP_THRESHOLD = 1.0f;
const float LOSS_SPIKE_THRESHOLD = 1.5f;
const size_t WARMUP_STEPS = 100;
float learning_rate = INITIAL_LEARNING_RATE;
float prev_loss = std::numeric_limits<float>::max();

// Define the special character map (definition)
const std::unordered_map<char, std::string> SPECIAL_CHAR_MAP = {
    {'\n', "<newline>"},
    {'\t', "<tab>"},
    {'.', "<period>"},
    {'!', "<exclamation>"},
    {'?', "<question>"},
    {',', "<comma>"}
};

// Helper function to clip gradients
void clip_gradients(std::vector<Matrix>& gradients, float threshold) {
    float total_norm = 0.0f;
    
    // Compute total gradient norm
    for (const auto& grad : gradients) {
        for (size_t i = 0; i < grad.size(); i++) {
            total_norm += grad.data()[i] * grad.data()[i];
        }
    }
    total_norm = std::sqrt(total_norm);
    
    // Clip if necessary
    if (total_norm > threshold) {
        float scaling_factor = threshold / (total_norm + 1e-6f);
        for (auto& grad : gradients) {
            for (size_t i = 0; i < grad.size(); i++) {
                grad.data()[i] *= scaling_factor;
            }
        }
    }
}

// Helper function to adjust learning rate
float adjust_learning_rate(float current_lr, float loss_ratio, size_t step) {
    const size_t WARMUP_STEPS = 50;        // Reduced from 1000
    const float PEAK_LR = 5e-4;            // Increased from 1e-4 for smaller batches
    const float MIN_LR = 1e-5;             // Increased from 1e-6
    
    // Warmup phase
    if (step < WARMUP_STEPS) {
        return MIN_LR + (PEAK_LR - MIN_LR) * (static_cast<float>(step) / WARMUP_STEPS);
    }
    
    // Cosine decay after warmup
    const size_t DECAY_STEPS = 5000;       // Reduced from 50000
    float progress = static_cast<float>(step - WARMUP_STEPS) / DECAY_STEPS;
    progress = std::min(1.0f, progress);
    
    float decay_factor = 0.5f * (1.0f + std::cos(progress * M_PI));
    float lr = MIN_LR + (PEAK_LR - MIN_LR) * decay_factor;
    
    // More aggressive learning rate reduction on loss spikes
    if (loss_ratio > LOSS_SPIKE_THRESHOLD) {
        lr *= 0.1f;  // Reduced more aggressively from 0.5f
    }
    
    return std::clamp(lr, MIN_LR, PEAK_LR);
}

// Helper function to validate input tokens
bool validate_input_sequence(const std::vector<int>& tokens, size_t vocab_size, size_t max_seq_length = 512) {
    if (tokens.empty() || tokens.size() > max_seq_length) {
        std::cerr << "Invalid sequence length: " << tokens.size() << std::endl;
        return false;
    }
    
    for (int token : tokens) {
        if (token < 0 || token >= static_cast<int>(vocab_size)) {
            std::cerr << "Invalid token id: " << token << std::endl;
            return false;
        }
    }
    
    return true;
}

// Helper function to create target distribution for next-token prediction
Matrix create_batch_target_distribution(const std::vector<std::vector<int>>& token_sequences, 
                                      const Tokenizer& tokenizer,
                                      size_t vocab_size) {
    if (token_sequences.empty()) {
        throw std::runtime_error("Cannot create target distribution from empty batch");
    }
    
    Matrix distribution(token_sequences.size(), vocab_size, 0.0f);
    
    for (size_t batch_idx = 0; batch_idx < token_sequences.size(); batch_idx++) {
        const auto& tokens = token_sequences[batch_idx];
        
        if (tokens.empty()) {
            throw std::runtime_error("Empty token sequence in batch at position " + 
                                   std::to_string(batch_idx));
        }
        
        // Skip special tokens when looking for the target
        size_t target_idx = tokens.size() - 1;
        while (target_idx > 0 && tokenizer.is_special_token(tokens[target_idx])) {
            target_idx--;
        }
        
        int target_token = tokens[target_idx];
        if (target_token < 0 || target_token >= static_cast<int>(vocab_size)) {
            throw std::runtime_error("Token " + std::to_string(target_token) + 
                " is out of vocabulary range [0, " + std::to_string(vocab_size) + 
                ") at batch position " + std::to_string(batch_idx));
        }
        
        distribution(batch_idx, target_token) = 1.0f;
    }
    
    return distribution;
}

// Helper function to compute loss for next-token prediction
float compute_batch_loss(const Matrix& logits, const Matrix& targets) {
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
            if (targets(i, j) > 0.0f) {
                loss -= targets(i, j) * std::log(probs[j] + epsilon);
            }
        }
    }
    
    return loss / logits.rows();
}

void print_matrix(const Matrix &m, const std::string &name, size_t max_rows = 5,
                  size_t max_cols = 5) {
  std::cout << "entering print_matrix" << std::endl;
  std::cout << "\n" << name << " (" << m.rows() << "x" << m.cols() << "):\n";
  for (size_t i = 0; i < std::min(max_rows, m.rows()); ++i) {
    for (size_t j = 0; j < std::min(max_cols, m.cols()); ++j) {
      std::cout << std::fixed << std::setprecision(4) << m(i, j) << " ";
    }
    std::cout << (m.cols() > max_cols ? "..." : "") << "\n";
  }
  if (m.rows() > max_rows)
    std::cout << "...\n";
  std::cout << "exiting print_matrix" << std::endl;
}

void print_top_predictions(const Matrix &logits, const Tokenizer &tokenizer,
                           size_t k = 5) {
  // Get the last row of logits
  std::vector<float> last_logits;
  for (size_t i = 0; i < logits.cols(); ++i) {
    last_logits.push_back(logits(logits.rows() - 1, i));
  }

  // Apply softmax with numerical stability
  float max_logit = *std::max_element(last_logits.begin(), last_logits.end());
  std::vector<float> probs(last_logits.size());
  float sum_exp = 0.0f;
  
  for (size_t i = 0; i < last_logits.size(); ++i) {
    probs[i] = std::exp(last_logits[i] - max_logit);
    sum_exp += probs[i];
  }
  
  // Normalize to get probabilities
  for (float& prob : probs) {
    prob /= sum_exp;
  }

  // Create scores with probabilities instead of raw logits
  std::vector<std::pair<float, int>> scores;
  for (size_t i = 0; i < probs.size(); ++i) {
    scores.push_back({probs[i], static_cast<int>(i)});
  }

  std::partial_sort(
      scores.begin(), scores.begin() + k, scores.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  std::cout << "\nTop " << k << " predictions:\n";
  for (size_t i = 0; i < k; ++i) {
    std::string token = tokenizer.decode({scores[i].second});
    std::cout << i + 1 << ". \"" << token << "\" (probability: " << std::fixed
              << std::setprecision(4) << scores[i].first << ")\n";
  }
}

std::vector<std::pair<std::string, std::string>> create_training_data() {
  std::cout << "entering create_training_data" << std::endl;
  std::vector<std::pair<std::string, std::string>> training_pairs;
  // Get the executable directory
  std::filesystem::path exe_path =
      std::filesystem::current_path().parent_path();
  std::filesystem::path data_dir = exe_path / "data";
  std::filesystem::path file_path = data_dir / "training_pairs.txt";

  // Create data directory if it doesn't exist
  if (!std::filesystem::exists(data_dir)) {
    std::filesystem::create_directories(data_dir);
  }

  std::ifstream file(file_path);

  if (!file.is_open()) {
    throw std::runtime_error("Could not open training data file: " +
                             file_path.string());
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

  if (training_pairs.empty()) {
    throw std::runtime_error("No training pairs loaded from file");
  }

  std::cout << "exiting create_training_data" << std::endl;
  return training_pairs;
}

// Update the analyze_token_mappings function
void analyze_token_mappings(const std::vector<std::pair<std::string, std::string>>& training_data, 
                          const Tokenizer& tokenizer) {
    std::cout << "\n=== Analyzing Token Mappings ===\n";
    
    size_t total_words = 0;
    size_t unknown_tokens = 0;
    std::unordered_map<std::string, int> unknown_words;
    
    for (const auto& pair : training_data) {
        // Preprocess and analyze input string
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
        
        // Preprocess and analyze target string
        std::string processed_target = pair.second;
        tokenizer.preprocess_text(processed_target);
        tokens = tokenizer.encode(processed_target);
        
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
    
    // Print statistics
    std::cout << "\nToken Mapping Statistics:\n";
    std::cout << "Total words processed: " << total_words << "\n";
    std::cout << "Unknown token occurrences: " << unknown_tokens << " (" 
              << (100.0f * unknown_tokens / total_words) << "%)\n\n";
    
    if (!unknown_words.empty()) {
        std::cout << "Words mapped to <unk> token:\n";
        for (const auto& [word, count] : unknown_words) {
            std::cout << "'" << word << "': " << count << " times\n";
        }
    }
    
    std::cout << "\n=== End Token Mapping Analysis ===\n\n";
}

int main(int argc, char *argv[]) {
  std::cout << "entering main" << std::endl;
  // Initialize logger
  Logger &logger = Logger::getInstance();
  logger.startLogging();

  try {
#ifdef CUDA_AVAILABLE
    initialize_cuda();
#endif
    // Initialize tokenizer first to get vocab size
    auto tokenizer = std::make_unique<Tokenizer>();
    tokenizer->print_vocabulary_mappings(); // Print initial mappings
    tokenizer->clear_cache();  // We need to add this method to Tokenizer class
    
    // Get vocabulary size from the tokenizer
    size_t actual_vocab_size = tokenizer->vocab_size();
    
    std::cout << "Actual vocabulary size: " << actual_vocab_size << std::endl;

    // Configure the transformer with actual vocab size
    TransformerConfig config;
    config.vocab_size = actual_vocab_size;
    config.hidden_size = 128;        // Reduced from 360 to prevent overfitting
    config.num_heads = 4;            // Reduced from 12 to match smaller hidden size
    config.num_layers = 3;           // Reduced from 6 to prevent overfitting
    config.use_cuda = false;
    config.use_flash_attention = true;
    config.use_rope = true;
    config.use_sliding_window = true;
    config.window_size = 128;        // Reduced from 256 since we have shorter sequences
    config.use_fp16 = false;
    config.head_dim = config.hidden_size / config.num_heads;
    config.batch_size = 4;           // Reduced from 8 due to small dataset
    config.num_epochs = 30;          // Increased from 10 to allow more training iterations
    config.dropout_rate = 0.2f;      // Add dropout to prevent overfitting
    config.weight_decay = 0.01f;     // Add L2 regularization

    std::cout << "Initializing transformer with configuration:\n"
              << "- Hidden size: " << config.hidden_size << "\n"
              << "- Attention heads: " << config.num_heads << "\n"
              << "- Layers: " << config.num_layers << "\n"
              << "- Batch size: " << config.batch_size << "\n"
              << "- Number of epochs: " << config.num_epochs << "\n"
              << "- Using Flash Attention: " << std::boolalpha
              << config.use_flash_attention << "\n"
              << "- Using RoPE: " << config.use_rope << "\n"
              << "- Using Sliding Window: " << config.use_sliding_window
              << "\n";

    // Initialize components
    Transformer transformer(config);
    size_t vocab_size = tokenizer->vocab_size();
    std::cout << "Actual vocabulary size from tokenizer: " << vocab_size << std::endl;

    // Update config with actual vocab size
    config.vocab_size = vocab_size;

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
    std::vector<std::pair<std::string, std::string>> training_data = create_training_data();
    
    // Preprocess the training data (convert to lowercase)
    training_data = TextPreprocessor::preprocess_training_data(training_data);
    
    // Analyze token mappings
    analyze_token_mappings(training_data, *tokenizer);
    
    // Print vocabulary for inspection
    std::cout << "\n=== Full Vocabulary Mapping ===\n";
    tokenizer->print_vocabulary_mappings();
    std::cout << "\n";

    // Training parameters
    const size_t checkpoint_frequency = 2; // Save checkpoint every 2 epochs

    // Initialize model saver
    ModelSaver model_saver;
    std::string save_directory = "models";
    std::string model_name = "transformer_model";

    // Training loop
    size_t global_step = 0;  // Move outside epoch loop
    Matrix last_hidden_states;
    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << config.num_epochs << "\n";
        float epoch_loss = 0.0f;
        size_t total_batches = (training_data.size() + config.batch_size - 1) / config.batch_size;
        
        // Process batches
        for (size_t batch = 0; batch < total_batches; ++batch) {
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
                if (!validate_input_sequence(input_tokens, config.vocab_size) || 
                    !validate_input_sequence(curr_target_tokens, config.vocab_size)) {
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
            Matrix target_distribution = create_batch_target_distribution(target_tokens, *tokenizer, config.vocab_size);
            
            // Process the batch as a single sequence
            std::vector<int> flattened_batch;
            flattened_batch.reserve(current_batch_size * max_seq_len);
            
            // Flatten the batch into a single sequence
            for (const auto& sequence : input_batch) {
                flattened_batch.insert(flattened_batch.end(), sequence.begin(), sequence.end());
            }
            
            // Forward pass with the flattened batch
            transformer.set_training(true);  // Ensure training mode is on
            Matrix hidden_states = transformer.forward(flattened_batch);
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
            float batch_loss = compute_batch_loss(final_logits, target_distribution);
            
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
            learning_rate = adjust_learning_rate(learning_rate, loss_ratio, global_step);
            
            // Backpropagate through the model
            Matrix lm_head_gradients = lm_head->backward(loss_gradients);
            transformer.backward(lm_head_gradients, flattened_batch, learning_rate);
            
            // Update tracking variables
            prev_loss = batch_loss;
            epoch_loss += batch_loss;
            global_step++;
            
            // Print progress
            if ((batch + 1) % 10 == 0 || batch + 1 == total_batches) {
                std::cout << "\rBatch " << batch + 1 << "/" << total_batches 
                         << " in epoch " << epoch + 1 
                         << " (Loss: " << batch_loss 
                         << ", Avg Loss: " << epoch_loss/(batch+1)
                         << ", LR: " << learning_rate 
                         << ")" << std::flush;
            }
        }
        
        std::cout << "\nCompleted epoch " << epoch + 1 << "/" << config.num_epochs 
                  << " (Loss: " << epoch_loss/total_batches << ")" << std::endl;
        
        // Save checkpoint
        if ((epoch + 1) % checkpoint_frequency == 0) {
            if (!model_saver.saveCheckpoint(transformer, save_directory, model_name,
                                            epoch + 1, epoch_loss)) {
                logger.log("Failed to save checkpoint", true);
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
                print_top_predictions(test_logits, *tokenizer, 5);
            }
        }

        if ((epoch + 1) % 5 == 0) {  // Clear cache every 5 epochs
            tokenizer->clear_cache();
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
        logger.log("Successfully saved model to " + save_directory + "/" +
                   model_name);
        std::cout << "Model saved successfully!\n";
    } else {
        logger.log("Failed to save model to " + save_directory + "/" + model_name,
                   true);
        return 1;
    }

    // Demonstrate quantization
    std::cout << "\nTesting quantization...\n";
    std::vector<Matrix> calibration_data{
        last_hidden_states}; // Use stored hidden states
    qat.calibrate(transformer, calibration_data);
    Matrix quantized = qat.quantize_weights(last_hidden_states, "layer_0");
    print_matrix(quantized, "Quantized hidden states");

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
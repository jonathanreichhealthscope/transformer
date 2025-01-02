#include "../include/attention.hpp"
#ifdef CUDA_AVAILABLE
#include "../include/cuda/cuda_init.cuh"
#endif
#include "../include/lm_head.hpp"
#include "../include/logger.hpp"
#include "../include/model_saver.hpp"
#include "../include/optimizer/sam.hpp"
#include "../include/quantization.hpp"
#include "../include/tokenizer.hpp"
#include "../include/transformer.hpp"
#include "../include/utils/tensor_cache.hpp"
#include "../include/vocabulary.hpp"
#include "../include/matrix.hpp"
#include "../include/gradient_tape.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <sstream>

// Add necessary forward declarations and structures
class Tokenizer;
std::unique_ptr<Tokenizer> tokenizer;

// Define training example structure
struct TrainingExample {
    std::vector<int> input_tokens;
    Matrix target;
};

// Configuration constants
const size_t BATCH_SIZE = 32;
const size_t num_epochs = 10;
const float INITIAL_LEARNING_RATE = 0.001f;
const float MIN_LEARNING_RATE = 1e-6f;
const float MAX_LEARNING_RATE = 0.1f;
const float GRADIENT_CLIP_THRESHOLD = 1.0f;
const float LOSS_SPIKE_THRESHOLD = 1.5f;
const size_t WARMUP_STEPS = 100;
float learning_rate = INITIAL_LEARNING_RATE;
float prev_loss = std::numeric_limits<float>::max();

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
    // Warmup phase
    if (step < WARMUP_STEPS) {
        return INITIAL_LEARNING_RATE * (step + 1) / WARMUP_STEPS;
    }
    
    // Adjust based on loss
    if (loss_ratio > LOSS_SPIKE_THRESHOLD) {
        current_lr *= 0.5f;  // Halve the learning rate
    } else if (loss_ratio < 0.8f) {
        current_lr *= 1.1f;  // Slightly increase learning rate
    }
    
    // Ensure learning rate stays within bounds
    return std::clamp(current_lr, MIN_LEARNING_RATE, MAX_LEARNING_RATE);
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

// Helper function to compute loss with improved numerical stability
float compute_batch_loss(const Matrix& logits, const Matrix& targets) {
    if (logits.rows() != targets.rows() || logits.cols() != targets.cols()) {
        throw std::runtime_error("Dimension mismatch in loss computation");
    }
    
    float loss = 0.0f;
    const float epsilon = 1e-10f;  // Small constant for numerical stability
    
    for (size_t i = 0; i < logits.rows(); i++) {
        for (size_t j = 0; j < logits.cols(); j++) {
            // Clip predictions to avoid log(0)
            float pred = std::clamp(logits(i, j), epsilon, 1.0f - epsilon);
            float target = targets(i, j);
            
            // Cross-entropy with improved numerical stability
            if (target > 0.0f) {
                loss -= target * std::log(pred);
            }
        }
    }
    
    return loss / logits.rows();  // Normalize by batch size
}

// Helper function to create target distribution for a batch
Matrix create_batch_target_distribution(const std::vector<std::vector<int>>& token_sequences, size_t vocab_size) {
    if (token_sequences.empty()) {
        throw std::runtime_error("Cannot create target distribution from empty batch");
    }
    
    // Create batch matrix (batch_size x vocab_size)
    Matrix distribution(token_sequences.size(), vocab_size, 0.0f);
    
    // Process each sequence in the batch
    for (size_t batch_idx = 0; batch_idx < token_sequences.size(); batch_idx++) {
        const auto& tokens = token_sequences[batch_idx];
        
        // Validate tokens
        if (tokens.empty()) {
            throw std::runtime_error("Empty token sequence in batch at position " + std::to_string(batch_idx));
        }
        
        for (int token : tokens) {
            if (token < 0 || token >= static_cast<int>(vocab_size)) {
                throw std::runtime_error("Token " + std::to_string(token) + 
                    " is out of vocabulary range [0, " + std::to_string(vocab_size) + 
                    ") at batch position " + std::to_string(batch_idx));
            }
        }
        
        // Set probabilities for this sequence
        float weight = 1.0f / tokens.size();
        for (int token : tokens) {
            distribution(batch_idx, token) = weight;
        }
    }
    
    // Validate output
    for (size_t i = 0; i < distribution.rows(); i++) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < distribution.cols(); j++) {
            float val = distribution(i, j);
            if (std::isnan(val) || std::isinf(val)) {
                throw std::runtime_error("Invalid value in target distribution at position (" + 
                    std::to_string(i) + "," + std::to_string(j) + ")");
            }
            row_sum += val;
        }
        if (std::abs(row_sum - 1.0f) > 1e-6) {
            throw std::runtime_error("Target distribution row " + std::to_string(i) + 
                " does not sum to 1.0 (sum = " + std::to_string(row_sum) + ")");
        }
    }
    
    return distribution;
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
  std::cout << "entering print_top_predictions" << std::endl;
  std::vector<std::pair<float, int>> scores;
  for (size_t i = 0; i < logits.cols(); ++i) {
    scores.push_back({logits(logits.rows() - 1, i), static_cast<int>(i)});
  }

  std::partial_sort(
      scores.begin(), scores.begin() + k, scores.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  std::cout << "\nTop " << k << " predictions:\n";
  for (size_t i = 0; i < k; ++i) {
    std::string token = tokenizer.decode({scores[i].second});
    std::cout << i + 1 << ". \"" << token << "\" (probability: " << std::fixed
              << std::setprecision(4) << std::exp(scores[i].first) << ")\n";
  }
  std::cout << "exiting print_top_predictions" << std::endl;
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

void analyze_token_mappings(const std::vector<std::pair<std::string, std::string>>& training_data, 
                          const Tokenizer& tokenizer) {
    std::cout << "\n=== Analyzing Token Mappings ===\n";
    
    // Track statistics
    size_t total_words = 0;
    size_t unknown_tokens = 0;
    std::unordered_map<std::string, int> unknown_words;
    
    for (const auto& pair : training_data) {
        // Analyze input string
        std::istringstream input_ss(pair.first);
        std::string word;
        while (input_ss >> word) {
            total_words++;
            std::vector<int> tokens = tokenizer.encode(word);
            for (int token : tokens) {
                if (tokenizer.decode({token}) == "<unk>") {
                    unknown_tokens++;
                    unknown_words[word]++;
                }
            }
        }
        
        // Analyze target string
        std::istringstream target_ss(pair.second);
        while (target_ss >> word) {
            total_words++;
            std::vector<int> tokens = tokenizer.encode(word);
            for (int token : tokens) {
                if (tokenizer.decode({token}) == "<unk>") {
                    unknown_tokens++;
                    unknown_words[word]++;
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
    
    // Get vocabulary size from the tokenizer
    size_t actual_vocab_size = tokenizer->vocab_size();
    
    std::cout << "Actual vocabulary size: " << actual_vocab_size << std::endl;

    // Configure the transformer with actual vocab size
    TransformerConfig config;
    config.vocab_size = actual_vocab_size;
    config.hidden_size = 360;
    config.num_heads = 6;
    config.num_layers = 6;
    config.use_cuda = false;
    config.use_flash_attention = false;
    config.use_rope = true;
    config.use_sliding_window = true;
    config.window_size = 256;
    config.use_fp16 = false;
    config.head_dim = config.hidden_size / config.num_heads;  // Add explicit head_dim calculation

    std::cout << "Initializing transformer with configuration:\n"
              << "- Hidden size: " << config.hidden_size << "\n"
              << "- Attention heads: " << config.num_heads << "\n"
              << "- Layers: " << config.num_layers << "\n"
              << "- Using Flash Attention: " << std::boolalpha
              << config.use_flash_attention << "\n"
              << "- Using RoPE: " << config.use_rope << "\n"
              << "- Using Sliding Window: " << config.use_sliding_window
              << "\n";

    // Initialize components
    Transformer transformer(config);
    auto lm_head = std::make_unique<LanguageModelHead>(config.vocab_size,
                                                       config.hidden_size);

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
    size_t global_step = 0;
    Matrix last_hidden_states;
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << "\n";
        float epoch_loss = 0.0f;
        size_t total_batches = (training_data.size() + BATCH_SIZE - 1) / BATCH_SIZE;
        
        for (size_t batch = 0; batch < total_batches; ++batch) {
            size_t start_idx = batch * BATCH_SIZE;
            size_t end_idx = std::min(start_idx + BATCH_SIZE, training_data.size());
            
            // Create batch with validation
            std::vector<std::vector<int>> input_batch;
            std::vector<std::vector<int>> target_tokens;
            
            // Fill and validate batch
            bool batch_valid = true;
            for (size_t j = start_idx; j < end_idx; ++j) {
                const auto &[input_str, target_str] = training_data[j];
                std::vector<int> input_tokens = tokenizer->encode(input_str);
                std::vector<int> curr_target_tokens = tokenizer->encode(target_str);
                
                if (!validate_input_sequence(input_tokens, config.vocab_size) || 
                    !validate_input_sequence(curr_target_tokens, config.vocab_size)) {
                    std::cerr << "Invalid sequence at position " << j << std::endl;
                    batch_valid = false;
                    break;
                }
                
                input_batch.push_back(input_tokens);
                target_tokens.push_back(curr_target_tokens);
            }
            
            if (!batch_valid) continue;  // Skip invalid batches
            
            // Create target distribution for entire batch at once
            Matrix target_distribution = create_batch_target_distribution(target_tokens, config.vocab_size);
            
            // Forward pass with gradient accumulation
            Matrix accumulated_gradients(config.hidden_size, config.vocab_size, 0.0f);
            float batch_loss = 0.0f;
            
            for (size_t i = 0; i < input_batch.size(); ++i) {
                // Forward pass
                Matrix hidden_states = transformer.forward(input_batch[i]);
                Matrix logits = lm_head->project_to_vocab(hidden_states);
                
                // Extract corresponding row from target distribution
                Matrix target_slice(1, target_distribution.cols());
                for (size_t j = 0; j < target_distribution.cols(); j++) {
                    target_slice(0, j) = target_distribution(i, j);
                }
                
                // Compute loss and gradients
                float sample_loss = compute_batch_loss(logits, target_slice);
                batch_loss += sample_loss;
                
                // Compute gradients using SAM
                std::vector<Matrix> param_grads;
                param_grads.reserve(transformer.getLayers().size());
                sam_optimizer->compute_parameter_gradients(hidden_states, target_slice, param_grads);
                
                // Clip gradients
                clip_gradients(param_grads, GRADIENT_CLIP_THRESHOLD);
                
                // Accumulate gradients
                for (const auto& grad : param_grads) {
                    for (size_t j = 0; j < grad.size(); j++) {
                        accumulated_gradients.data()[j] += grad.data()[j];
                    }
                }
            }
            
            // Average gradients
            float scale = 1.0f / input_batch.size();
            for (size_t i = 0; i < accumulated_gradients.size(); i++) {
                accumulated_gradients.data()[i] *= scale;
            }
            
            // Update learning rate
            float loss_ratio = batch_loss / (prev_loss + 1e-10f);
            learning_rate = adjust_learning_rate(learning_rate, loss_ratio, global_step++);
            
            // Apply gradients
            transformer.backward(accumulated_gradients, input_batch[0]);  // Use first sequence for shape
            
            // Update loss tracking
            prev_loss = batch_loss;
            epoch_loss += batch_loss;
            
            // Print progress with learning rate
            std::cout << "\rBatch " << batch + 1 << "/" << total_batches 
                      << " in epoch " << epoch + 1 
                      << " (Loss: " << batch_loss 
                      << ", LR: " << learning_rate << ")" << std::flush;
        }
        
        std::cout << "\nCompleted epoch " << epoch + 1 << "/" << num_epochs 
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
                "I go to",                  // Basic location
                "Surgeons operate in the",  // Medical context
                "Athletes train in the",    // Sports context
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
                std::vector<int> test_tokens = tokenizer->encode(test_input);
                Matrix test_hidden = transformer.forward(test_tokens);
                Matrix test_logits = lm_head->forward(test_hidden);
                print_top_predictions(test_logits, *tokenizer, 3);
            }
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
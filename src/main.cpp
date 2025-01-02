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

// Add necessary forward declarations and structures
class Tokenizer;
std::unique_ptr<Tokenizer> tokenizer;
size_t vocab_size;

// Define training example structure
struct TrainingExample {
    std::vector<int> input_tokens;
    Matrix target;
};

// Configuration constants
const size_t BATCH_SIZE = 32;
const size_t num_epochs = 10;
float learning_rate = 0.001f;
float prev_loss = std::numeric_limits<float>::max();

// Helper function to create target distribution
Matrix create_target_distribution(const std::vector<int>& tokens, size_t vocab_size) {
    std::cout << "entering create_target_distribution" << std::endl;
    Matrix distribution(1, vocab_size, 0.0f);
    for (int token : tokens) {
        if (token < static_cast<int>(vocab_size)) {
            distribution(0, token) = 1.0f;
        }
    }
    std::cout << "exiting create_target_distribution" << std::endl;
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

int main(int argc, char *argv[]) {
  std::cout << "entering main" << std::endl;
  // Initialize logger
  Logger &logger = Logger::getInstance();
  logger.startLogging();

  try {
#ifdef CUDA_AVAILABLE
    initialize_cuda(); // Initialize CUDA at program start
#endif

    // Configure the transformer
    TransformerConfig config;
    config.vocab_size = 50000;
    config.hidden_size = 768;
    config.num_heads = 12;
    config.num_layers = 6;
    config.use_cuda = false;
    config.use_flash_attention = false;
/*#ifdef CUDA_AVAILABLE
    config.use_flash_attention = true;
    config.use_cuda = true;
#else
    config.use_flash_attention = false;
    config.use_cuda = false;
#endif*/
    config.use_rope = true;
    config.use_sliding_window = true;
    config.window_size = 256;
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
    auto tokenizer = std::make_unique<Tokenizer>();
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
    auto training_data = create_training_data();

    // Training parameters
    const size_t checkpoint_frequency = 2; // Save checkpoint every 2 epochs

    // Initialize model saver
    ModelSaver model_saver;
    std::string save_directory = "models";
    std::string model_name = "transformer_model";

    // Training loop
    Matrix last_hidden_states; // Add this to store the last hidden states
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
      std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << "\n";
      float epoch_loss = 0.0f;

      size_t total_batches = (training_data.size() + BATCH_SIZE - 1) / BATCH_SIZE;
      
      // Process each batch sequentially to avoid thread issues
      for (size_t batch = 0; batch < total_batches; ++batch) {
        size_t start_idx = batch * BATCH_SIZE;
        size_t end_idx = std::min(start_idx + BATCH_SIZE, training_data.size());
        
        // Create batch
        std::vector<std::vector<int>> input_batch;
        std::vector<Matrix> target_batch;  // Store Matrix targets

        // Fill batch
        for (size_t j = start_idx; j < end_idx; ++j) {
            const auto &[input_str, target_str] = training_data[j];
            std::vector<int> input_tokens = tokenizer->encode(input_str);
            Matrix target = create_target_distribution(tokenizer->encode(target_str), config.vocab_size);
            
            input_batch.push_back(input_tokens);
            target_batch.push_back(target);
            
            std::cout << "Processing batch item " << j - start_idx + 1 
                      << "/" << end_idx - start_idx << "\n";
            std::cout << "Target: '" << target_str << "'\n";
        }

        // Get input and target from training pairs
        const auto &[input_str, target_str] = training_data[start_idx];
        std::vector<int> current_input_tokens = tokenizer->encode(input_str);
        Matrix current_target = create_target_distribution(tokenizer->encode(target_str), config.vocab_size);
        
        std::cout << "Processing pair " << start_idx << ": '" << input_str << "' -> '"
                  << target_str << "'\n";

        // Tokenize input and target
        std::vector<int> display_tokens;
        for (size_t i = 0; i < current_target.cols(); i++) {
            if (current_target(0, i) > 0.5f) {
                display_tokens.push_back(i);
            }
        }

        std::cout << "Input tokens: " << current_input_tokens.size() << "\n";
        std::cout << "Target tokens: " << display_tokens.size() << "\n";
        std::cout << "Forward pass for input tokens '" << tokenizer->decode(display_tokens) << "'\n";
        // Forward pass
        Matrix hidden_states = transformer.forward(current_input_tokens);
        last_hidden_states = hidden_states;
        std::cout << "Forward pass for hidden states '" << tokenizer->decode(display_tokens) << "'\n";
        // Project to vocabulary space
        Matrix logits = lm_head->project_to_vocab(hidden_states);
        std::cout << "Hidden states shape: " << hidden_states.rows() << "x"
                  << hidden_states.cols() << "\n";
        std::cout << "Logits shape: " << logits.rows() << "x" << logits.cols()
                  << "\n";

        // Compute gradients using SAM
        std::vector<Matrix> param_grads;
        param_grads.reserve(transformer.getLayers().size());
        
        // Create target distribution matrix
        Matrix target_distribution = create_target_distribution(tokenizer->encode(target_str), 
            transformer.getConfig().vocab_size);
        
        // Compute parameter gradients
        sam_optimizer->compute_parameter_gradients(hidden_states, target_distribution, param_grads);
        
        // Compute loss from gradients
        float batch_loss = 0.0f;
        for (const auto& grad : param_grads) {
            for (size_t i = 0; i < grad.size(); i++) {
                batch_loss += grad.data()[i] * grad.data()[i];
            }
        }
        batch_loss = std::sqrt(batch_loss) / (param_grads.size() * hidden_states.size());
        
        // Adjust learning rate if needed
        if (batch_loss > prev_loss * 1.5f) {
            learning_rate *= 0.5f;
        }
        prev_loss = batch_loss;
        epoch_loss += batch_loss;
        
        // Backward pass
        Matrix grad = sam_optimizer->compute_gradients(hidden_states,
            transformer.get_hidden_states(),
            transformer.get_lm_head());
        transformer.backward(grad, current_input_tokens);
        
        // Print progress
        std::cout << "\rBatch " << batch + 1 << "/" << total_batches 
                  << " in epoch " << epoch + 1 << std::flush;
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
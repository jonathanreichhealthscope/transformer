#include "../include/attention/advanced_attention.hpp"
#include "../include/lm_head.hpp"
#include "../include/optimizer/sam.hpp"
#include "../include/quantization.hpp"
#include "../include/tokenizer.hpp"
#include "../include/transformer.hpp"
#include "../include/utils/tensor_cache.hpp"
#include "../include/vocabulary.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

void print_matrix(const Matrix &m, const std::string &name, size_t max_rows = 5,
                  size_t max_cols = 5) {
  std::cout << "\n" << name << " (" << m.rows() << "x" << m.cols() << "):\n";
  for (size_t i = 0; i < std::min(max_rows, m.rows()); ++i) {
    for (size_t j = 0; j < std::min(max_cols, m.cols()); ++j) {
      std::cout << std::fixed << std::setprecision(4) << m(i, j) << " ";
    }
    std::cout << (m.cols() > max_cols ? "..." : "") << "\n";
  }
  if (m.rows() > max_rows)
    std::cout << "...\n";
}

void print_top_predictions(const Matrix &logits, const Tokenizer &tokenizer,
                           size_t k = 5) {
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
}

int main() {
  try {
    // Configure the transformer
    TransformerConfig config;
    config.vocab_size = 50000;
    config.hidden_size = 768;
    config.num_heads = 12;
    config.num_layers = 6;
    config.use_flash_attention = true;
    config.use_rope = true;
    config.use_sliding_window = true;
    config.window_size = 256;
    config.use_cuda = true;

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

    // Example input
    std::string input_text = "I go to";
    std::cout << "\nProcessing input: '" << input_text << "'\n";

    // Tokenize input
    std::vector<int> tokens = tokenizer->encode(input_text);
    std::cout << "Tokenized to " << tokens.size() << " tokens: ";
    for (int token : tokens) {
      std::cout << token << " ";
    }
    std::cout << "\nDecoded tokens: " << tokenizer->decode(tokens) << "\n";

    // Process input with performance measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Forward pass through transformer
    Matrix hidden_states = transformer.forward(tokens);
    print_matrix(hidden_states, "Transformer hidden states");

    // Apply language model head
    Matrix logits = lm_head->forward(hidden_states);
    print_matrix(logits, "Language model logits");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print predictions
    print_top_predictions(logits, *tokenizer);

    std::cout << "\nProcessing time: " << duration.count() << "ms\n";

    // Demonstrate multi-GPU capability
    std::cout << "\nTesting multi-GPU processing...\n";
    MultiGPUManager gpu_manager;
    std::vector<Matrix> batch_inputs(8, hidden_states); // Create a batch of 8
    auto batch_results = gpu_manager.parallel_forward(batch_inputs);
    std::cout << "Successfully processed batch across " << batch_results.size()
              << " GPUs\n";

    // Demonstrate quantization
    std::cout << "\nTesting quantization...\n";
    std::vector<Matrix> calibration_data{
        hidden_states}; // Use current hidden states as calibration
    qat.calibrate(transformer, calibration_data);
    Matrix quantized = qat.quantize_weights(hidden_states, "layer_0");
    print_matrix(quantized, "Quantized hidden states");

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
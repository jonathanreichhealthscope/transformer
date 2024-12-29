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

// Add this helper function to create a simple dataset
std::vector<std::pair<std::string, std::string>> create_training_data() {
    return {
        // Basic locations
        {"I go to", "school"}, {"I walk to", "work"}, {"We drive to the", "store"},
        {"They went to the", "park"}, {"She runs to the", "gym"},
        {"He walks to the", "office"}, {"Students go to the", "library"},
        {"We went to the", "beach"}, {"I drive to the", "airport"},
        {"They walk to the", "station"},

        // Animal patterns
        {"The cat sits on the", "mat"}, {"The dog sleeps on the", "bed"},
        {"Birds fly in the", "sky"}, {"Fish swim in the", "ocean"},
        {"The lion rests in the", "shade"}, {"Ducks swim in the", "pond"},
        {"The bear lives in the", "forest"}, {"Wolves hunt in the", "night"},
        {"Eagles soar in the", "air"}, {"Rabbits hop in the", "grass"},

        // Work and study
        {"She works at the", "hospital"}, {"He teaches at the", "university"},
        {"They study in the", "classroom"}, {"He works in the", "factory"},
        {"She teaches at the", "school"}, {"They work in the", "office"},
        {"He studies at the", "college"}, {"She practices in the", "studio"},
        {"They train at the", "center"}, {"He performs in the", "theater"},

        // Children activities
        {"Children play in the", "park"}, {"Kids swim in the", "pool"},
        {"Students learn in the", "classroom"}, {"Children read in the", "library"},
        {"Kids practice in the", "gym"}, {"Students eat in the", "cafeteria"},
        {"Children draw in the", "room"}, {"Kids play on the", "playground"},
        {"Students work in the", "laboratory"}, {"Children sing in the", "hall"},

        // Nature scenes
        {"Trees grow in the", "forest"}, {"Flowers bloom in the", "garden"},
        {"Rivers flow through the", "valley"}, {"Stars shine in the", "sky"},
        {"Waves crash on the", "shore"}, {"Snow falls on the", "ground"},
        {"Wind blows through the", "trees"}, {"Rain falls on the", "earth"},
        {"Clouds float in the", "sky"}, {"Grass grows in the", "field"},

        // Urban scenes
        {"Cars drive on the", "road"}, {"People walk on the", "sidewalk"},
        {"Trains stop at the", "station"}, {"Planes land at the", "airport"},
        {"Ships dock at the", "port"}, {"Buses stop at the", "terminal"},
        {"Cyclists ride on the", "path"}, {"Shoppers browse in the", "mall"},
        {"Workers build the", "building"}, {"Artists paint in the", "studio"},

        // Home activities
        {"Mother cooks in the", "kitchen"}, {"Father reads in the", "study"},
        {"Sister plays in the", "room"}, {"Brother sleeps in the", "bedroom"},
        {"Family eats in the", "dining room"}, {"Grandmother sits in the", "garden"},
        {"Baby crawls on the", "floor"}, {"Parents relax in the", "living room"},
        {"Dog sleeps by the", "fireplace"}, {"Cat watches from the", "window"},

        // Time of day
        {"Sun rises in the", "morning"}, {"Moon shines in the", "night"},
        {"People wake at", "dawn"}, {"Stars appear at", "dusk"},
        {"Workers leave at", "sunset"}, {"Children wake in the", "morning"},
        {"Owls hunt in the", "night"}, {"Farmers work at", "sunrise"},
        {"People sleep at", "midnight"}, {"Students arrive at", "noon"},

        // Weather patterns
        {"Snow falls from the", "sky"}, {"Rain pours on the", "ground"},
        {"Wind blows through the", "trees"}, {"Lightning flashes in the", "sky"},
        {"Thunder rolls across the", "sky"}, {"Fog covers the", "ground"},
        {"Sun shines in the", "sky"}, {"Clouds gather in the", "sky"},
        {"Rainbow appears in the", "sky"}, {"Frost covers the", "ground"},

        // Professional settings
        {"Doctor works in the", "hospital"}, {"Lawyer speaks in the", "court"},
        {"Chef cooks in the", "kitchen"}, {"Teacher instructs in the", "classroom"},
        {"Artist paints in the", "studio"}, {"Scientist works in the", "laboratory"},
        {"Musician plays in the", "hall"}, {"Dancer performs on the", "stage"},
        {"Writer works in the", "office"}, {"Engineer builds in the", "workshop"},

        // Sports and recreation
        {"Athletes train in the", "gym"}, {"Players compete in the", "stadium"},
        {"Swimmers practice in the", "pool"}, {"Runners race on the", "track"},
        {"Teams play in the", "field"}, {"Climbers scale the", "mountain"},
        {"Skaters glide on the", "ice"}, {"Surfers ride the", "waves"},
        {"Hikers walk on the", "trail"}, {"Cyclists ride on the", "path"},

        // Entertainment venues
        {"Audience sits in the", "theater"}, {"Bands play at the", "concert"},
        {"People dance in the", "club"}, {"Visitors explore the", "museum"},
        {"Fans cheer in the", "stadium"}, {"Readers browse in the", "bookstore"},
        {"Gamers play in the", "arcade"}, {"Diners eat in the", "restaurant"},
        {"Viewers watch in the", "cinema"}, {"Guests relax at the", "resort"},

        // Shopping and commerce
        {"People shop in the", "mall"}, {"Vendors sell at the", "market"},
        {"Customers wait in the", "store"}, {"Cashiers work at the", "register"},
        {"Shoppers browse in the", "boutique"}, {"Merchants trade in the", "bazaar"},
        {"Buyers gather at the", "auction"}, {"Sellers display in the", "shop"},
        {"People bargain at the", "market"}, {"Customers line up at the", "checkout"},

        // Transportation hubs
        {"Passengers wait at the", "station"}, {"Travelers rush through the", "airport"},
        {"People board at the", "terminal"}, {"Commuters gather at the", "platform"},
        {"Drivers stop at the", "garage"}, {"Pilots land at the", "runway"},
        {"Sailors dock at the", "port"}, {"Tourists arrive at the", "terminal"},
        {"Crews work at the", "hangar"}, {"Controllers work in the", "tower"},

        // Religious and cultural
        {"People pray in the", "temple"}, {"Worshippers gather in the", "church"},
        {"Muslims pray in the", "mosque"}, {"Monks meditate in the", "monastery"},
        {"Believers meet in the", "sanctuary"}, {"Priests teach in the", "seminary"},
        {"Devotees worship at the", "shrine"}, {"Pilgrims visit the", "temple"},
        {"Congregations meet in the", "chapel"}, {"Students study in the", "seminary"},

        // Educational settings
        {"Professors teach in the", "university"}, {"Students research in the", "library"},
        {"Children learn in the", "classroom"}, {"Teachers work in the", "school"},
        {"Scholars study in the", "academy"}, {"Pupils gather in the", "auditorium"},
        {"Researchers work in the", "laboratory"}, {"Tutors teach in the", "center"},
        {"Instructors train in the", "facility"}, {"Learners practice in the", "workshop"},

        // Emergency services
        {"Firefighters work at the", "station"}, {"Police patrol on the", "street"},
        {"Paramedics rush to the", "hospital"}, {"Guards work at the", "facility"},
        {"Rangers patrol in the", "park"}, {"Officers work at the", "precinct"},
        {"Medics train at the", "center"}, {"Rescuers gather at the", "base"},
        {"Crews respond from the", "station"}, {"Teams deploy from the", "headquarters"},

        // Technology spaces
        {"Programmers work in the", "office"}, {"Engineers build in the", "laboratory"},
        {"Developers code in the", "workspace"}, {"Technicians repair in the", "shop"},
        {"Designers create in the", "studio"}, {"Analysts work in the", "center"},
        {"Researchers test in the", "facility"}, {"Inventors build in the", "workshop"},
        {"Teams collaborate in the", "space"}, {"Experts work in the", "lab"}
    };
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

    // Create training data
    auto training_pairs = create_training_data();
    std::cout << "\nTraining on " << training_pairs.size() << " examples\n";

    // Training parameters
    const size_t num_epochs = 10;
    const float learning_rate = 0.001f;

    // Training loop
    Matrix last_hidden_states;  // Add this to store the last hidden states
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << "\n";
        float epoch_loss = 0.0f;
        
        for (const auto& [input_text, target_text] : training_pairs) {
            std::cout << "Processing input: '" << input_text << "'\n";
            // Tokenize input and target
            std::vector<int> input_tokens = tokenizer->encode(input_text);
            std::cout << "Input tokens: " << input_tokens.size() << "\n"; 
            std::vector<int> target_tokens = tokenizer->encode(target_text);
            std::cout << "Target tokens: " << target_tokens.size() << "\n";
            std::cout << "Forward pass for input tokens '" << target_text << "'\n";
            // Forward pass
            Matrix hidden_states = transformer.forward(input_tokens);
            last_hidden_states = hidden_states;
            std::cout << "Forward pass for hidden states '" << target_text << "'\n";
            Matrix logits = lm_head->forward(hidden_states);
            std::cout << "Logits shape: " << logits.rows() << "x" << logits.cols() << "\n";

            // Compute loss and gradients
            Matrix target_matrix(logits.rows(), logits.cols(), 0.0f);
            std::cout << "Target matrix shape: " << target_matrix.rows() << "x" << target_matrix.cols() << "\n";
            // We only care about the last token's prediction
            size_t last_position = logits.rows() - 1;  // Last position in sequence
            std::cout << "Last position: " << last_position << "\n";
            for (int token : target_tokens) {
                target_matrix(last_position, token) = 1.0f;  // One-hot encode only the last position
            }
            std::cout << "Target matrix after one-hot encoding: " << target_matrix.rows() << "x" << target_matrix.cols() << "\n";
            // Cross entropy loss (only for last position)
            float loss = 0.0f;
            for (size_t j = 0; j < logits.cols(); ++j) {
                if (target_matrix(last_position, j) > 0.0f) {
                    loss -= std::log(logits(last_position, j) + 1e-10);
                }
            }
            epoch_loss += loss;
            std::cout << "Loss: " << loss << "\n";
            // Backward pass - compute gradients
            Matrix grad_output = logits;
            std::cout << "Created gradient output matrix\n";
            
            // Apply softmax derivative: grad * (softmax - target)
            // First, apply softmax to the last position
            float max_val = -std::numeric_limits<float>::infinity();
            std::cout << "Initialized max value for softmax\n";
            
            for (size_t j = 0; j < logits.cols(); ++j) {
                max_val = std::max(max_val, logits(last_position, j));
            }
            std::cout << "Found max value: " << max_val << "\n";
            
            float sum = 0.0f;
            std::cout << "Initialized sum for softmax normalization\n";
            
            for (size_t j = 0; j < logits.cols(); ++j) {
                grad_output(last_position, j) = std::exp(logits(last_position, j) - max_val);
                sum += grad_output(last_position, j);
            }
            std::cout << "Computed exponentials and sum: " << sum << "\n";
            
            // Normalize and compute gradient
            for (size_t j = 0; j < logits.cols(); ++j) {
                grad_output(last_position, j) = grad_output(last_position, j) / sum;  // Softmax
                grad_output(last_position, j) -= target_matrix(last_position, j);     // Subtract target
                grad_output(last_position, j) /= logits.rows();  // Scale by batch size
            }
            std::cout << "Normalized gradients and subtracted targets\n";
            
            // Zero out gradients for positions other than last_position
            for (size_t i = 0; i < logits.rows(); ++i) {
                if (i != last_position) {
                    for (size_t j = 0; j < logits.cols(); ++j) {
                        grad_output(i, j) = 0.0f;
                    }
                }
            }
            std::cout << "Zeroed out gradients for non-target positions\n";

            // Update weights using SAM optimizer
            std::vector<Matrix*> params;
            std::vector<Matrix> grads;
            std::cout << "Updating weights using SAM optimizer\n";
            // Add transformer parameters
            auto transformer_weights = transformer.get_layer_weights();
            for (const auto& layer_weights : transformer_weights) {
                for (auto& weight : layer_weights) {
                    params.push_back(&weight.get());
                }
            }
            std::cout << "Transformer parameters added\n";
            // Add language model head parameters
            auto lm_params = lm_head->get_parameters();
            for (auto& param : lm_params) {
                params.push_back(&param.get());
            }
            std::cout << "Language model parameters added\n";
            // Initialize gradients
            grads.push_back(grad_output);  // For hidden states
            for (size_t i = 1; i < params.size(); ++i) {
                grads.push_back(Matrix(params[i]->rows(), params[i]->cols()));  // Initialize with zeros
            }
            std::cout << "Gradients initialized\n";
            // Update parameters
            sam_optimizer->first_step(params, grads);
            std::cout << "First step completed\n";
            sam_optimizer->second_step(params, grads);
            std::cout << "Second step completed\n";

            // Handle bias separately if needed
            // Note: You might want to implement a separate update rule for the bias
        }

        // Print epoch statistics
        epoch_loss /= training_pairs.size();
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs 
                 << ", Loss: " << epoch_loss << "\n";

        // Test prediction on a sample input
        if ((epoch + 1) % 2 == 0) {
            std::string test_input = "I go to";
            std::cout << "\nTesting: '" << test_input << "'\n";
            std::vector<int> test_tokens = tokenizer->encode(test_input);
            Matrix test_hidden = transformer.forward(test_tokens);
            Matrix test_logits = lm_head->forward(test_hidden);
            print_top_predictions(test_logits, *tokenizer, 3);
        }
    }

    std::cout << "\nTraining completed!\n";

    // Demonstrate multi-GPU capability
    std::cout << "\nTesting multi-GPU processing...\n";
    MultiGPUManager gpu_manager;
    std::vector<Matrix> batch_inputs(8, last_hidden_states); // Create a batch of 8
    auto batch_results = gpu_manager.parallel_forward(batch_inputs);
    std::cout << "Successfully processed batch across " << batch_results.size()
              << " GPUs\n";

    // Demonstrate quantization
    std::cout << "\nTesting quantization...\n";
    std::vector<Matrix> calibration_data{last_hidden_states}; // Use stored hidden states
    qat.calibrate(transformer, calibration_data);
    Matrix quantized = qat.quantize_weights(last_hidden_states, "layer_0");
    print_matrix(quantized, "Quantized hidden states");

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
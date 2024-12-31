#include "../include/attention/advanced_attention.hpp"
#include "../include/cuda/cuda_init.cuh"
#include "../include/lm_head.hpp"
#include "../include/logger.hpp"
#include "../include/model_saver.hpp"
#include "../include/optimizer/sam.hpp"
#include "../include/quantization.hpp"
#include "../include/tokenizer.hpp"
#include "../include/transformer.hpp"
#include "../include/utils/tensor_cache.hpp"
#include "../include/vocabulary.hpp"
#include <chrono>
#include <filesystem>
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
  return {// Basic locations
          {"I go to", "school"},
          {"I walk to", "work"},
          {"We drive to the", "store"},
          {"They went to the", "park"},
          {"She runs to the", "gym"},
          {"He walks to the", "office"},
          {"Students go to the", "library"},
          {"We went to the", "beach"},
          {"I drive to the", "airport"},
          {"They walk to the", "station"},

          // Animal patterns
          {"The cat sits on the", "mat"},
          {"The dog sleeps on the", "bed"},
          {"Birds fly in the", "sky"},
          {"Fish swim in the", "ocean"},
          {"The lion rests in the", "shade"},
          {"Ducks swim in the", "pond"},
          {"The bear lives in the", "forest"},
          {"Wolves hunt in the", "night"},
          {"Eagles soar in the", "air"},
          {"Rabbits hop in the", "grass"},

          // Work and study
          {"She works at the", "hospital"},
          {"He teaches at the", "university"},
          {"They study in the", "classroom"},
          {"He works in the", "factory"},
          {"She teaches at the", "school"},
          {"They work in the", "office"},
          {"He studies at the", "college"},
          {"She practices in the", "studio"},
          {"They train at the", "center"},
          {"He performs in the", "theater"},

          // Children activities
          {"Children play in the", "park"},
          {"Kids swim in the", "pool"},
          {"Students learn in the", "classroom"},
          {"Children read in the", "library"},
          {"Kids practice in the", "gym"},
          {"Students eat in the", "cafeteria"},
          {"Children draw in the", "room"},
          {"Kids play on the", "playground"},
          {"Students work in the", "laboratory"},
          {"Children sing in the", "hall"},

          // Nature scenes
          {"Trees grow in the", "forest"},
          {"Flowers bloom in the", "garden"},
          {"Rivers flow through the", "valley"},
          {"Stars shine in the", "sky"},
          {"Waves crash on the", "shore"},
          {"Snow falls on the", "ground"},
          {"Wind blows through the", "trees"},
          {"Rain falls on the", "earth"},
          {"Clouds float in the", "sky"},
          {"Grass grows in the", "field"},

          // Urban scenes
          {"Cars drive on the", "road"},
          {"People walk on the", "sidewalk"},
          {"Trains stop at the", "station"},
          {"Planes land at the", "airport"},
          {"Ships dock at the", "port"},
          {"Buses stop at the", "terminal"},
          {"Cyclists ride on the", "path"},
          {"Shoppers browse in the", "mall"},
          {"Workers build the", "building"},
          {"Artists paint in the", "studio"},

          // Home activities
          {"Mother cooks in the", "kitchen"},
          {"Father reads in the", "study"},
          {"Sister plays in the", "room"},
          {"Brother sleeps in the", "bedroom"},
          {"Family eats in the", "dining room"},
          {"Grandmother sits in the", "garden"},
          {"Baby crawls on the", "floor"},
          {"Parents relax in the", "living room"},
          {"Dog sleeps by the", "fireplace"},
          {"Cat watches from the", "window"},

          // Time of day
          {"Sun rises in the", "morning"},
          {"Moon shines in the", "night"},
          {"People wake at", "dawn"},
          {"Stars appear at", "dusk"},
          {"Workers leave at", "sunset"},
          {"Children wake in the", "morning"},
          {"Owls hunt in the", "night"},
          {"Farmers work at", "sunrise"},
          {"People sleep at", "midnight"},
          {"Students arrive at", "noon"},

          // Weather patterns
          {"Snow falls from the", "sky"},
          {"Rain pours on the", "ground"},
          {"Wind blows through the", "trees"},
          {"Lightning flashes in the", "sky"},
          {"Thunder rolls across the", "sky"},
          {"Fog covers the", "ground"},
          {"Sun shines in the", "sky"},
          {"Clouds gather in the", "sky"},
          {"Rainbow appears in the", "sky"},
          {"Frost covers the", "ground"},

          // Professional settings
          {"Doctor works in the", "hospital"},
          {"Lawyer speaks in the", "court"},
          {"Chef cooks in the", "kitchen"},
          {"Teacher instructs in the", "classroom"},
          {"Artist paints in the", "studio"},
          {"Scientist works in the", "laboratory"},
          {"Musician plays in the", "hall"},
          {"Dancer performs on the", "stage"},
          {"Writer works in the", "office"},
          {"Engineer builds in the", "workshop"},

          // Sports and recreation
          {"Athletes train in the", "gym"},
          {"Players compete in the", "stadium"},
          {"Swimmers practice in the", "pool"},
          {"Runners race on the", "track"},
          {"Teams play in the", "field"},
          {"Climbers scale the", "mountain"},
          {"Skaters glide on the", "ice"},
          {"Surfers ride the", "waves"},
          {"Hikers walk on the", "trail"},
          {"Cyclists ride on the", "path"},

          // Entertainment venues
          {"Audience sits in the", "theater"},
          {"Bands play at the", "concert"},
          {"People dance in the", "club"},
          {"Visitors explore the", "museum"},
          {"Fans cheer in the", "stadium"},
          {"Readers browse in the", "bookstore"},
          {"Gamers play in the", "arcade"},
          {"Diners eat in the", "restaurant"},
          {"Viewers watch in the", "cinema"},
          {"Guests relax at the", "resort"},

          // Shopping and commerce
          {"People shop in the", "mall"},
          {"Vendors sell at the", "market"},
          {"Customers wait in the", "store"},
          {"Cashiers work at the", "register"},
          {"Shoppers browse in the", "boutique"},
          {"Merchants trade in the", "bazaar"},
          {"Buyers gather at the", "auction"},
          {"Sellers display in the", "shop"},
          {"People bargain at the", "market"},
          {"Customers line up at the", "checkout"},

          // Transportation hubs
          {"Passengers wait at the", "station"},
          {"Travelers rush through the", "airport"},
          {"People board at the", "terminal"},
          {"Commuters gather at the", "platform"},
          {"Drivers stop at the", "garage"},
          {"Pilots land at the", "runway"},
          {"Sailors dock at the", "port"},
          {"Tourists arrive at the", "terminal"},
          {"Crews work at the", "hangar"},
          {"Controllers work in the", "tower"},

          // Religious and cultural
          {"People pray in the", "temple"},
          {"Worshippers gather in the", "church"},
          {"Muslims pray in the", "mosque"},
          {"Monks meditate in the", "monastery"},
          {"Believers meet in the", "sanctuary"},
          {"Priests teach in the", "seminary"},
          {"Devotees worship at the", "shrine"},
          {"Pilgrims visit the", "temple"},
          {"Congregations meet in the", "chapel"},
          {"Students study in the", "seminary"},

          // Educational settings
          {"Professors teach in the", "university"},
          {"Students research in the", "library"},
          {"Children learn in the", "classroom"},
          {"Teachers work in the", "school"},
          {"Scholars study in the", "academy"},
          {"Pupils gather in the", "auditorium"},
          {"Researchers work in the", "laboratory"},
          {"Tutors teach in the", "center"},
          {"Instructors train in the", "facility"},
          {"Learners practice in the", "workshop"},

          // Emergency services
          {"Firefighters work at the", "station"},
          {"Police patrol on the", "street"},
          {"Paramedics rush to the", "hospital"},
          {"Guards work at the", "facility"},
          {"Rangers patrol in the", "park"},
          {"Officers work at the", "precinct"},
          {"Medics train at the", "center"},
          {"Rescuers gather at the", "base"},
          {"Crews respond from the", "station"},
          {"Teams deploy from the", "headquarters"},

          // Technology spaces
          {"Programmers work in the", "office"},
          {"Engineers build in the", "laboratory"},
          {"Developers code in the", "workspace"},
          {"Technicians repair in the", "shop"},
          {"Designers create in the", "studio"},
          {"Analysts work in the", "center"},
          {"Researchers test in the", "facility"},
          {"Inventors build in the", "workshop"},
          {"Teams collaborate in the", "space"},
          {"Experts work in the", "lab"},

          // Entertainment venues
          {"Audiences gather in the", "theater"},
          {"Musicians perform in the", "concert hall"},
          {"Dancers practice in the", "studio"},
          {"Actors rehearse on the", "stage"},
          {"Spectators sit in the", "arena"},
          {"Gamers compete in the", "tournament"},
          {"Artists paint in the", "gallery"},
          {"DJs perform at the", "club"},
          {"Comedians entertain at the", "comedy club"},
          {"Performers prepare in the", "dressing room"},

          // Medical facilities
          {"Surgeons operate in the", "operating room"},
          {"Patients wait in the", "clinic"},
          {"Nurses work in the", "ward"},
          {"Doctors consult in the", "office"},
          {"Specialists examine in the", "examination room"},
          {"Therapists treat in the", "therapy room"},
          {"Pharmacists work in the", "pharmacy"},
          {"Dentists practice in the", "dental office"},
          {"Radiologists work in the", "radiology department"},
          {"Psychiatrists counsel in the", "consultation room"},

          // Sports facilities
          {"Athletes train in the", "gym"},
          {"Swimmers practice in the", "pool"},
          {"Players compete in the", "stadium"},
          {"Boxers fight in the", "ring"},
          {"Skaters glide on the", "ice rink"},
          {"Climbers practice on the", "wall"},
          {"Golfers practice at the", "driving range"},
          {"Tennis players serve on the", "court"},
          {"Bowlers play in the", "bowling alley"},
          {"Runners train on the", "track"}
  };
}

int main(int argc, char *argv[]) {
  // Initialize logger
  Logger &logger = Logger::getInstance();
  logger.startLogging();
  // logger.disableLogging();

  try {
    initialize_cuda(); // Initialize CUDA at program start

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
    config.use_cuda = false;

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

      const size_t batch_size = 32; // Adjust based on your GPU memory
      for (size_t i = 0; i < training_pairs.size(); i += batch_size) {
        std::cout << "Processing batch " << i << std::endl;
        // Create batch
        std::vector<std::vector<int>> input_batch;
        std::vector<std::vector<int>> target_batch;

        // Fill batch
        for (size_t j = 0; j < batch_size && (i + j) < training_pairs.size();
             ++j) {
          const auto &[input_text, target_text] = training_pairs[i + j];
          input_batch.push_back(tokenizer->encode(input_text));
          target_batch.push_back(tokenizer->encode(target_text));
        }

        // Get input and target from training pairs
        const auto &[input_text, target_text] = training_pairs[i];
        std::cout << "Processing pair " << i << ": '" << input_text << "' -> '"
                  << target_text << "'\n";

        // Tokenize input and target
        std::vector<int> input_tokens = tokenizer->encode(input_text);
        std::vector<int> target_tokens = tokenizer->encode(target_text);
        std::cout << "Input tokens: " << input_tokens.size() << "\n";
        std::cout << "Target tokens: " << target_tokens.size() << "\n";
        std::cout << "Forward pass for input tokens '" << target_text << "'\n";
        // Forward pass
        Matrix hidden_states = transformer.forward(input_tokens);
        last_hidden_states = hidden_states;
        std::cout << "Forward pass for hidden states '" << target_text << "'\n";
        Matrix logits = lm_head->forward(hidden_states);
        std::cout << "Logits shape: " << logits.rows() << "x" << logits.cols()
                  << "\n";

        // Compute loss and gradients
        Matrix target_matrix(logits.rows(), logits.cols(), 0.0f);
        std::cout << "Target matrix shape: " << target_matrix.rows() << "x"
                  << target_matrix.cols() << "\n";
        // We only care about the last token's prediction
        size_t last_position = logits.rows() - 1; // Last position in sequence
        std::cout << "Last position: " << last_position << "\n";
        for (int token : target_tokens) {
          target_matrix(last_position, token) =
              1.0f; // One-hot encode only the last position
        }
        std::cout << "Target matrix after one-hot encoding: "
                  << target_matrix.rows() << "x" << target_matrix.cols()
                  << "\n";
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
        Matrix grad_output(logits.rows(), logits.cols(),
                           0.0f); // Initialize with zeros
        std::cout << "Created gradient output matrix\n";

        // Apply softmax derivative: grad * (softmax - target)
        // First, apply softmax to the last position
        float max_val = -std::numeric_limits<float>::infinity();
        std::cout << "Initialized max value for softmax\n";

        // Only compute gradients for the last position
        for (size_t j = 0; j < logits.cols(); ++j) {
          max_val = std::max(max_val, logits(last_position, j));
        }
        std::cout << "Found max value: " << max_val << "\n";

        float sum = 0.0f;
        std::cout << "Initialized sum for softmax normalization\n";

        // Store softmax values temporarily
        std::vector<float> softmax_values(logits.cols());
        for (size_t j = 0; j < logits.cols(); ++j) {
          softmax_values[j] = std::exp(logits(last_position, j) - max_val);
          sum += softmax_values[j];
        }
        std::cout << "Computed exponentials and sum: " << sum << "\n";

        // Compute gradients only for last position
        for (size_t j = 0; j < logits.cols(); ++j) {
          float softmax_prob = softmax_values[j] / sum;
          grad_output(last_position, j) =
              softmax_prob - target_matrix(last_position, j);
        }
        std::cout << "Computed gradients for last position\n";

        // Note: Other positions are already zero from initialization

        // Update weights using SAM optimizer
        std::vector<Matrix *> params;
        std::vector<Matrix> grads;
        std::cout << "Updating weights using SAM optimizer\n";

        // Add transformer parameters
        auto transformer_weights = transformer.get_layer_weights();
        for (const auto &layer_weights : transformer_weights) {
          for (auto &weight : layer_weights) {
            params.push_back(&weight.get());
          }
        }
        std::cout << "Transformer parameters added\n";

        // Add language model head parameters
        auto lm_params = lm_head->get_parameters();
        for (auto &param : lm_params) {
          params.push_back(&param.get());
        }
        std::cout << "Language model parameters added\n";

        // Initialize gradients
        for (size_t i = 0; i < params.size(); ++i) {
          grads.push_back(Matrix(params[i]->rows(), params[i]->cols()));
          std::cout << "Initialized gradient for param " << i
                    << " with dimensions: " << params[i]->rows() << "x"
                    << params[i]->cols() << "\n";
        }

        // First step with initial gradients
        std::cout << "Starting SAM first step with initial gradients\n";
        std::cout << "Number of parameters: " << params.size()
                  << ", Number of gradients: " << grads.size() << "\n";
        sam_optimizer->first_step(params, grads);
        std::cout << "Completed first step\n";

        // Recompute gradients at the perturbed point
        std::cout << "\nRecomputing gradients at perturbed point...\n";
        Matrix new_hidden_states = transformer.forward(input_tokens);
        std::cout << "New hidden states shape: " << new_hidden_states.rows()
                  << "x" << new_hidden_states.cols() << "\n";

        Matrix new_logits = lm_head->forward(new_hidden_states);
        std::cout << "New logits shape: " << new_logits.rows() << "x"
                  << new_logits.cols() << "\n";

        // Recompute grad_output similar to before
        Matrix new_grad_output(new_logits.rows(), new_logits.cols(), 0.0f);
        std::cout << "Created new gradient output matrix: "
                  << new_grad_output.rows() << "x" << new_grad_output.cols()
                  << "\n";

        // Recompute softmax and gradients for last position
        float new_max_val = -std::numeric_limits<float>::infinity();
        size_t last_pos = new_logits.rows() - 1;
        for (size_t j = 0; j < new_logits.cols(); ++j) {
          new_max_val = std::max(new_max_val, new_logits(last_pos, j));
        }
        std::cout << "Computed new max value for softmax: " << new_max_val
                  << "\n";

        float new_sum = 0.0f;
        std::vector<float> new_softmax_values(new_logits.cols());
        for (size_t j = 0; j < new_logits.cols(); ++j) {
          new_softmax_values[j] =
              std::exp(new_logits(last_pos, j) - new_max_val);
          new_sum += new_softmax_values[j];
        }
        
        std::cout << "Computed new softmax normalization sum: " << new_sum
                  << "\n";

        // Create new gradients vector with correct dimensions
        std::vector<Matrix> new_grads;
        std::cout << "Created empty new_grads vector\n";

        // First compute gradients for transformer parameters
        for (size_t i = 0; i < params.size(); ++i) {
          // Initialize each gradient with same dimensions as its parameter
          new_grads.push_back(
              Matrix(params[i]->rows(), params[i]->cols(), 0.0f));
          std::cout << "Created gradient " << i
                    << " with dimensions: " << new_grads.back().rows() << "x"
                    << new_grads.back().cols() << "\n";
        }
        std::cout
            << "Finished initializing all gradients with correct dimensions\n";

        // Compute gradients for the last position
        std::cout << "Computing gradients for last position...\n";
        for (size_t j = 0; j < new_logits.cols(); ++j) {
          float softmax_prob = new_softmax_values[j] / new_sum;
          new_grad_output(last_pos, j) =
              softmax_prob - target_matrix(last_pos, j);
        }
        std::cout << "Computed new gradients for last position\n";

        // Backpropagate the gradients through the network
        std::cout << "Starting gradient backpropagation\n";
        Matrix current_grad = new_grad_output;
        std::cout << "Created current_grad with dimensions: " << current_grad.shape() << "\n";

        // Print dimensions of first few parameters for debugging
        std::cout << "First few parameter dimensions:\n";
        for (size_t i = 0; i < std::min(size_t(3), params.size()); ++i) {
          /*std::cout << "Parameter " << i << ": " << params[i]->rows() << "x"
                    << params[i]->cols() << "\n";*/
          std::cout << "Parameter shape: " << params[i]->shape() << "\n";
        }

        // Ensure gradients match parameter dimensions exactly
        for (size_t i = 0; i < params.size(); ++i) {
          std::cout << "Computing gradient for parameter " << i << "\n";

          // Get parameter dimensions
          size_t param_rows = params[i]->rows();
          size_t param_cols = params[i]->cols();
          
          std::cout << "Parameter dimensions: " << param_rows << "x"
                    << param_cols << "\n";

          // Create gradient with matching dimensions
          new_grads[i] = Matrix(param_rows, param_cols, 0.0f);

          // For now, use a very small constant gradient for testing
          // This ensures dimensions match exactly with the parameter
          for (size_t r = 0; r < param_rows; ++r) {
            for (size_t c = 0; c < param_cols; ++c) {
              new_grads[i](r, c) = 1e-4f; // Very small constant gradient
            }
          }
          std::cout << "Created gradient with matching dimensions: "
                    << new_grads[i].shape() << "\n";
        }
        std::cout << "Completed gradient computation for all parameters\n";

        // Verify gradient dimensions before second step
        std::cout << "\nVerifying gradient dimensions:\n";
        for (size_t i = 0; i < params.size(); ++i) {
          if (params[i]->rows() != new_grads[i].rows() ||
              params[i]->cols() != new_grads[i].cols()) {
            std::cout << "Dimension mismatch at parameter " << i << "!\n";
            std::cout << "Parameter: " << params[i]->shape() << "\n";
            std::cout << "Gradient: " << new_grads[i].shape() << "\n";

            throw std::runtime_error(
                "Gradient dimensions don't match parameters");
          }
        }
        std::cout << "All gradient dimensions verified\n";

        // Second step with new gradients
        std::cout << "\nStarting SAM second step\n";
        std::cout << "Number of parameters: " << params.size()
                  << ", Number of new gradients: " << new_grads.size() << "\n";
        sam_optimizer->second_step(params, new_grads);
        std::cout << "Completed second step\n\n";

        // Handle bias updates separately
        std::vector<std::reference_wrapper<FloatVector>> biases;
        std::vector<FloatVector> bias_grads;

        // Collect biases from transformer layers
        for (const auto &layer : transformer.getLayers()) {
          // Collect attention biases
          auto *attn = layer->getAttention();
          biases.push_back(std::ref(attn->getQueryBias()));
          biases.push_back(std::ref(attn->getKeyBias()));
          biases.push_back(std::ref(attn->getValueBias()));
          biases.push_back(std::ref(attn->getOutputBias()));

          // Collect feed forward biases
          auto *ff = layer->getFeedForward();
          biases.push_back(std::ref(ff->getBias1()));
          biases.push_back(std::ref(ff->getBias2()));
        }

        // Compute bias gradients
        bias_grads.resize(biases.size());
        for (size_t i = 0; i < biases.size(); ++i) {
          const FloatVector &bias = biases[i].get();
          FloatVector &grad = bias_grads[i];
          grad.resize(bias.size());

          // Compute gradients for biases (simplified)
          for (size_t j = 0; j < bias.size(); ++j) {
            grad[j] = 0.0001f; // Small constant gradient for testing
          }
        }

        // Update biases
        try {
          sam_optimizer->update_bias(biases, bias_grads);
          std::cout << "Completed bias updates\n";
        } catch (const std::exception &e) {
          std::cerr << "Error updating biases: " << e.what() << std::endl;
          throw;
        }
      }

      // Print epoch statistics
      epoch_loss /= training_pairs.size();
      std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                << ", Loss: " << epoch_loss << "\n";

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
        std::string test_input = "I go to";
        std::cout << "\nTesting: '" << test_input << "'\n";
        std::vector<int> test_tokens = tokenizer->encode(test_input);
        Matrix test_hidden = transformer.forward(test_tokens);
        Matrix test_logits = lm_head->forward(test_hidden);
        print_top_predictions(test_logits, *tokenizer, 3);
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

  cleanup_cuda(); // Cleanup at program end
  logger.stopLogging();
  return 0;
}
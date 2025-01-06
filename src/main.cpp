#include "../include/transformer.hpp"
#include "../include/tokenizer.hpp"
#include "../include/utils.hpp"
#include "../include/logger.hpp"
#include <iostream>
#include <memory>
#include <string>

int main() {
    try {
        // Initialize the logger
        Logger::getInstance().startLogging("transformer.log");

        // Log the start of the program
        Logger::getInstance().log("Starting transformer program");

        // Load config
        TransformerConfig config = Utils::load_config("config/transformer_config.json");
        Logger::getInstance().log("Loaded transformer configuration");

        // Log configuration parameters
        std::cout << "\n=== Configuration Parameters ===\n";
        std::cout << "Vocab Size: " << config.vocab_size << "\n";
        std::cout << "Max Sequence Length: " << config.max_seq_length << "\n";
        std::cout << "Hidden Size: " << config.hidden_size << "\n";
        std::cout << "Number of Layers: " << config.num_layers << "\n";
        std::cout << "Number of Heads: " << config.num_heads << "\n";
        std::cout << "Head Dimension: " << config.head_dim << "\n";
        std::cout << "Intermediate Size: " << config.intermediate_size << "\n";
        std::cout << "Batch Size: " << config.batch_size << "\n";
        std::cout << "Number of Epochs: " << config.num_epochs << "\n";
        std::cout << "Dropout Rate: " << config.dropout_rate << "\n";
        std::cout << "Weight Decay: " << config.weight_decay << "\n";
        std::cout << "Using Flash Attention: " << (config.use_flash_attention ? "Yes" : "No") << "\n";
        std::cout << "Using RoPE: " << (config.use_rope ? "Yes" : "No") << "\n";
        std::cout << "Using GQA: " << (config.use_gqa ? "Yes" : "No") << "\n";
        if (config.use_gqa) {
            std::cout << "Number of KV Heads: " << config.num_kv_heads << "\n";
        }
        std::cout << "================================\n\n";

        // Create SAM optimizer with custom settings
        auto sam_optimizer = std::make_unique<SAM>(
            0.05f,  // rho (sharpness-aware weight)
            std::make_unique<Optimizer>(
                0.001f,  // learning rate
                0.9f,    // beta1
                0.999f   // beta2
            )
        );
        Logger::getInstance().log("Initialized SAM optimizer");

        // Create transformer
        Transformer transformer(config, std::move(sam_optimizer));
        Logger::getInstance().log("Created transformer model");

        // Create tokenizer
        auto tokenizer = std::make_unique<Tokenizer>();
        Logger::getInstance().log("Initialized tokenizer");

        // Load training data
        auto training_data = Utils::create_training_data();
        Logger::getInstance().log("Loaded training data");

    // Load validation data
    auto validation_data = Utils::load_validation_data();
        Logger::getInstance().log("Loaded validation data");

        // Train model with checkpoint callback
        const size_t checkpoint_frequency = 1000;
        size_t global_step = 0;
        float total_loss = 0.0f;
        size_t batch_count = 0;
        const float learning_rate = 0.001f;  // Define learning rate

        // Print training configuration
        std::cout << "\n=== Training Configuration ===\n";
        std::cout << "Number of epochs: " << config.num_epochs << "\n";
        std::cout << "Learning rate: " << learning_rate << "\n";
        std::cout << "Batch size: " << config.batch_size << "\n";
        std::cout << "Training samples: " << training_data.size() << "\n";
        std::cout << "Validation samples: " << validation_data.size() << "\n";
        std::cout << "Checkpoint frequency: " << checkpoint_frequency << "\n";
        std::cout << "=============================\n\n";

        auto checkpoint_callback = [&](size_t step) {
            global_step = step;
            if (step % checkpoint_frequency == 0) {
                std::string checkpoint_path = "checkpoints/model_" + std::to_string(step) + ".pt";
                std::cout << "\n=== Saving Checkpoint ===\n";
                std::cout << "Step: " << step << "\n";
                std::cout << "Path: " << checkpoint_path << "\n";
                Logger::getInstance().log("Saving checkpoint to " + checkpoint_path);
                transformer.save_model(checkpoint_path);
                GradientCheckpoint::save_activation(transformer.get_hidden_states(), step);
                Logger::getInstance().log("Checkpoint saved successfully");
                std::cout << "======================\n\n";
            }
        };

        std::cout << "\n=== Starting Training ===\n";
        std::cout << "Time: " << std::time(nullptr) << "\n";
        transformer.train(training_data, validation_data, config.num_epochs, learning_rate, checkpoint_callback);
        std::cout << "\n=== Training Complete ===\n";
        std::cout << "Total steps: " << global_step << "\n";
        std::cout << "Final loss: " << (total_loss / batch_count) << "\n";
        std::cout << "Time: " << std::time(nullptr) << "\n";
        std::cout << "========================\n\n";

        // Save final model
        std::string final_path = "checkpoints/model_final.pt";
        std::cout << "\n=== Saving Final Model ===\n";
        std::cout << "Path: " << final_path << "\n";
        Logger::getInstance().log("Saving final model to " + final_path);
        transformer.save_model(final_path);
        GradientCheckpoint::save_activation(transformer.get_hidden_states(), global_step + 1);
        Logger::getInstance().log("Final model saved successfully");
        std::cout << "=======================\n\n";

        // Stop logging before exit
        Logger::getInstance().stopLogging();

        return 0;
    } catch (const std::exception& e) {
        Logger::getInstance().log("Error: " + std::string(e.what()), true);
        Logger::getInstance().stopLogging();
      return 1;
    }
}
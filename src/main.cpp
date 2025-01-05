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
        Logger::getInstance().startLogging("build/transformer.log");

        // Log the start of the program
        Logger::getInstance().log("Starting transformer program");

        // Load config
        TransformerConfig config = Utils::load_config("config/transformer_config.json");
        Logger::getInstance().log("Loaded transformer configuration");

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

        // Train model
        transformer.train(training_data, validation_data, config.num_epochs, 0.001f);
        Logger::getInstance().log("Training completed");

        // Stop logging before exit
        Logger::getInstance().stopLogging();

        return 0;
    } catch (const std::exception& e) {
        Logger::getInstance().log("Error: " + std::string(e.what()), true);
        Logger::getInstance().stopLogging();
        return 1;
    }
}
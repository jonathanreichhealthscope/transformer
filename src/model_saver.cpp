#include "../include/model_saver.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

ModelSaver::ModelSaver() : logger(Logger::getInstance()) {}

bool ModelSaver::saveModel(const Transformer& transformer, const std::string& directory,
                           const std::string& model_name) {
    try {
        std::string dir_path = createDirectory(directory);
        std::string model_path = dir_path + "/" + model_name + ".ckpt";

        logger.log("Saving model to: " + model_path);

        // Save model configuration
        if (!writeMetadata(directory, model_name, transformer.getConfig())) {
            logger.log("Failed to save model metadata", true);
            return false;
        }

        // Save model weights
        std::ofstream model_file(model_path, std::ios::binary);
        if (!model_file) {
            logger.log("Failed to open model file for writing", true);
            return false;
        }

        // Save each layer's weights
        const auto& layers = transformer.getLayers();
        for (const auto& layer : layers) {
            layer->save(model_file);
        }

        logger.log("Model saved successfully");
        return true;
    } catch (const std::exception& e) {
        logger.log("Error saving model: " + std::string(e.what()), true);
        return false;
    }
}

bool ModelSaver::loadModel(Transformer& transformer, const std::string& directory,
                           const std::string& model_name) {
    try {
        std::string model_path = directory + "/" + model_name;
        logger.log("Loading model from: " + model_path);

        // Load and verify configuration
        TransformerConfig config;
        if (!readMetadata(directory, model_name, config)) {
            logger.log("Failed to read model metadata", true);
            return false;
        }

        // Verify configuration matches
        if (config != transformer.getConfig()) {
            logger.log("Model configuration mismatch", true);
            return false;
        }

        // Load model weights
        std::ifstream model_file(model_path + ".bin", std::ios::binary);
        if (!model_file) {
            logger.log("Failed to open model file for reading", true);
            return false;
        }

        // Load each layer's weights
        auto& layers = transformer.getLayers();
        for (auto& layer : layers) {
            layer->load(model_file);
        }

        logger.log("Model loaded successfully");
        return true;
    } catch (const std::exception& e) {
        logger.log("Error loading model: " + std::string(e.what()), true);
        return false;
    }
}

bool ModelSaver::saveCheckpoint(const Transformer& transformer, const std::string& directory,
                                const std::string& model_name, int epoch, float loss) {
    try {
        // Create directory first and check permissions
        fs::path dir_path(directory);
        if (!fs::exists(dir_path)) {
            if (!fs::create_directories(dir_path)) {
                logger.log("Failed to create directory: " + directory +
                               " (Check permissions and path)",
                           true);
                return false;
            }
        } else if (!fs::is_directory(dir_path)) {
            logger.log("Path exists but is not a directory: " + directory, true);
            return false;
        }

        // Check directory permissions
        std::error_code ec;
        auto perms = fs::status(dir_path, ec).permissions();
        if (ec) {
            logger.log("Failed to check directory permissions: " + ec.message(), true);
            return false;
        }

        if ((perms & fs::perms::owner_write) == fs::perms::none) {
            logger.log("Directory is not writable: " + directory, true);
            return false;
        }

        std::string checkpoint_file = getCheckpointFilename(directory, model_name, epoch);
        logger.log("Saving checkpoint to: " + checkpoint_file);

        // Test file writability before proceeding
        {
            std::ofstream test_file(checkpoint_file);
            if (!test_file) {
                logger.log("Cannot write to checkpoint file: " + checkpoint_file +
                               " (Check permissions)",
                           true);
                return false;
            }
        }

        // Open checkpoint file for actual writing
        std::ofstream ckpt_file(checkpoint_file, std::ios::binary);
        if (!ckpt_file) {
            logger.log("Failed to open checkpoint file for writing: " + checkpoint_file, true);
            return false;
        }

        // Write metadata as JSON to start of file
        json checkpoint_meta;
        const auto& config = transformer.getConfig();

        checkpoint_meta["epoch"] = epoch;
        checkpoint_meta["loss"] = loss;
        checkpoint_meta["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
        checkpoint_meta["model_config"] = {{"vocab_size", config.vocab_size},
                                           {"hidden_size", config.hidden_size},
                                           {"num_heads", config.num_heads},
                                           {"num_layers", config.num_layers}};
        checkpoint_meta["batch_size"] = config.batch_size;

        std::string meta_str = checkpoint_meta.dump();
        size_t meta_size = meta_str.size();

        // Write metadata size and content
        if (!ckpt_file.write(reinterpret_cast<const char*>(&meta_size), sizeof(meta_size)) ||
            !ckpt_file.write(meta_str.c_str(), meta_size)) {
            logger.log("Failed to write metadata to checkpoint file", true);
            return false;
        }

        // Save model state
        const auto& layers = transformer.getLayers();
        for (const auto& layer : layers) {
            layer->save(ckpt_file);
        }

        // Ensure everything is written
        ckpt_file.flush();
        if (!ckpt_file) {
            logger.log("Error occurred while writing checkpoint file", true);
            return false;
        }

        logger.log("Checkpoint saved successfully");
        return true;
    } catch (const std::exception& e) {
        logger.log("Error saving checkpoint: " + std::string(e.what()), true);
        return false;
    }
}

bool ModelSaver::loadCheckpoint(Transformer& transformer, const std::string& checkpoint_path) {
    std::ifstream ckpt_file(checkpoint_path, std::ios::binary);
    if (!ckpt_file) {
        logger.log("Failed to open checkpoint file for reading", true);
        return false;
    }

    try {
        // Read metadata size
        size_t meta_size;
        ckpt_file.read(reinterpret_cast<char*>(&meta_size), sizeof(meta_size));

        // Read metadata JSON
        std::string meta_str(meta_size, '\0');
        ckpt_file.read(&meta_str[0], meta_size);

        json checkpoint_meta = json::parse(meta_str);

        // Verify model configuration
        const auto& config = transformer.getConfig();
        const auto& saved_config = checkpoint_meta["model_config"];

        if (saved_config["vocab_size"] != config.vocab_size ||
            saved_config["hidden_size"] != config.hidden_size ||
            saved_config["num_heads"] != config.num_heads ||
            saved_config["num_layers"] != config.num_layers) {
            logger.log("Model configuration mismatch in checkpoint", true);
            return false;
        }

        // Load model state
        transformer.load(ckpt_file);

        logger.log("Successfully loaded checkpoint from epoch " +
                   std::to_string(checkpoint_meta["epoch"].get<int>()));
        return true;
    } catch (const std::exception& e) {
        logger.log("Error loading checkpoint: " + std::string(e.what()), true);
        return false;
    }
}

bool ModelSaver::loadLatestCheckpoint(Transformer& transformer, const std::string& directory,
                                      const std::string& model_name, int& epoch, float& loss) {
    try {
        // Find latest checkpoint
        int latest_epoch = -1;
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.path().extension() == ".ckpt") {
                std::string filename = entry.path().stem().string();
                if (filename.find(model_name + "_checkpoint_") == 0) {
                    int checkpoint_epoch =
                        std::stoi(filename.substr(filename.find_last_of("_") + 1));
                    latest_epoch = std::max(latest_epoch, checkpoint_epoch);
                }
            }
        }

        if (latest_epoch == -1) {
            logger.log("No checkpoints found", true);
            return false;
        }

        // Load the latest checkpoint
        std::string checkpoint_file = getCheckpointFilename(directory, model_name, latest_epoch);
        if (!loadCheckpoint(transformer, checkpoint_file)) {
            return false;
        }

        // Update epoch and loss from the loaded checkpoint
        std::ifstream ckpt_file(checkpoint_file, std::ios::binary);
        size_t meta_size;
        ckpt_file.read(reinterpret_cast<char*>(&meta_size), sizeof(meta_size));

        std::string meta_str(meta_size, '\0');
        ckpt_file.read(&meta_str[0], meta_size);

        json checkpoint_meta = json::parse(meta_str);
        epoch = checkpoint_meta["epoch"];
        loss = checkpoint_meta["loss"];

        return true;
    } catch (const std::exception& e) {
        logger.log("Error loading latest checkpoint: " + std::string(e.what()), true);
        return false;
    }
}

std::string ModelSaver::createDirectory(const std::string& base_dir) const {
    fs::path dir_path(base_dir);
    fs::create_directories(dir_path);
    return dir_path.string();
}

std::string ModelSaver::getCheckpointFilename(const std::string& directory,
                                              const std::string& model_name, int epoch) const {
    return directory + "/" + model_name + "_checkpoint_" + std::to_string(epoch) + ".ckpt";
}

bool ModelSaver::writeMetadata(const std::string& directory, const std::string& model_name,
                               const TransformerConfig& config) const {
    json meta;
    meta["model_name"] = model_name;
    meta["vocab_size"] = config.vocab_size;
    meta["hidden_size"] = config.hidden_size;
    meta["num_heads"] = config.num_heads;
    meta["num_layers"] = config.num_layers;
    meta["use_flash_attention"] = config.use_flash_attention;
    meta["use_rope"] = config.use_rope;
    meta["use_sliding_window"] = config.use_sliding_window;
    meta["window_size"] = config.window_size;

    std::ofstream meta_file(directory + "/" + model_name + ".meta.json");
    meta_file << std::setw(4) << meta << std::endl;
    return true;
}

bool ModelSaver::readMetadata(const std::string& directory, const std::string& model_name,
                              TransformerConfig& config) const {
    std::ifstream meta_file(directory + "/" + model_name + ".meta.json");
    if (!meta_file) {
        return false;
    }

    json meta;
    meta_file >> meta;

    config.vocab_size = meta["vocab_size"];
    config.hidden_size = meta["hidden_size"];
    config.num_heads = meta["num_heads"];
    config.num_layers = meta["num_layers"];
    config.use_flash_attention = meta["use_flash_attention"];
    config.use_rope = meta["use_rope"];
    config.use_sliding_window = meta["use_sliding_window"];
    config.window_size = meta["window_size"];

    return true;
}

void ModelSaver::save_vocabulary(const std::string& path, const Vocabulary& vocab) {
    try {
        // Save special token mappings in consistent order
        special_tokens_json = {
            {"<pad>", 0},
            {"<unk>", 1},
            {"<bos>", 2},
            {"<eos>", 3},
            {"<mask>", 4}
        };

        // Write to file
        std::ofstream file(path);
        if (!file) {
            logger.log("Failed to open vocabulary file for writing: " + path, true);
            return;
        }
        file << std::setw(4) << special_tokens_json << std::endl;
    } catch (const std::exception& e) {
        logger.log("Error saving vocabulary: " + std::string(e.what()), true);
    }
}
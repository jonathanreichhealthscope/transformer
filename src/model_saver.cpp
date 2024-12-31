#include "../include/model_saver.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

ModelSaver::ModelSaver() : logger(Logger::getInstance()) {}

bool ModelSaver::saveModel(const Transformer &transformer,
                           const std::string &directory,
                           const std::string &model_name) {
  try {
    std::string dir_path = createDirectory(directory);
    std::string model_path = dir_path + "/" + model_name;

    logger.log("Saving model to: " + model_path);

    // Save model configuration
    if (!writeMetadata(directory, model_name, transformer.getConfig())) {
      logger.log("Failed to save model metadata", true);
      return false;
    }

    // Save model weights
    std::ofstream model_file(model_path + ".bin", std::ios::binary);
    if (!model_file) {
      logger.log("Failed to open model file for writing", true);
      return false;
    }

    // Save each layer's weights
    const auto &layers = transformer.getLayers();
    for (const auto &layer : layers) {
      layer->save(model_file);
    }

    logger.log("Model saved successfully");
    return true;
  } catch (const std::exception &e) {
    logger.log("Error saving model: " + std::string(e.what()), true);
    return false;
  }
}

bool ModelSaver::loadModel(Transformer &transformer,
                           const std::string &directory,
                           const std::string &model_name) {
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
    auto &layers = transformer.getLayers();
    for (auto &layer : layers) {
      layer->load(model_file);
    }

    logger.log("Model loaded successfully");
    return true;
  } catch (const std::exception &e) {
    logger.log("Error loading model: " + std::string(e.what()), true);
    return false;
  }
}

bool ModelSaver::saveCheckpoint(const Transformer &transformer,
                                const std::string &directory,
                                const std::string &model_name, int epoch,
                                float loss) {
  try {
    std::string checkpoint_file =
        getCheckpointFilename(directory, model_name, epoch);
    logger.log("Saving checkpoint to: " + checkpoint_file);

    // Save model state
    if (!saveModel(transformer, directory,
                   model_name + "_checkpoint_" + std::to_string(epoch))) {
      return false;
    }

    // Save checkpoint metadata
    json checkpoint_meta;
    checkpoint_meta["epoch"] = epoch;
    checkpoint_meta["loss"] = loss;
    checkpoint_meta["timestamp"] =
        std::chrono::system_clock::now().time_since_epoch().count();

    std::ofstream meta_file(checkpoint_file + ".meta.json");
    meta_file << std::setw(4) << checkpoint_meta << std::endl;

    logger.log("Checkpoint saved successfully");
    return true;
  } catch (const std::exception &e) {
    logger.log("Error saving checkpoint: " + std::string(e.what()), true);
    return false;
  }
}

bool ModelSaver::loadLatestCheckpoint(Transformer &transformer,
                                      const std::string &directory,
                                      const std::string &model_name, int &epoch,
                                      float &loss) {
  try {
    // Find latest checkpoint
    int latest_epoch = -1;
    for (const auto &entry : fs::directory_iterator(directory)) {
      if (entry.path().extension() == ".meta.json") {
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

    // Load checkpoint metadata
    std::string checkpoint_file =
        getCheckpointFilename(directory, model_name, latest_epoch);
    std::ifstream meta_file(checkpoint_file + ".meta.json");
    json checkpoint_meta;
    meta_file >> checkpoint_meta;

    epoch = checkpoint_meta["epoch"];
    loss = checkpoint_meta["loss"];

    // Load model state
    return loadModel(transformer, directory,
                     model_name + "_checkpoint_" +
                         std::to_string(latest_epoch));
  } catch (const std::exception &e) {
    logger.log("Error loading checkpoint: " + std::string(e.what()), true);
    return false;
  }
}

std::string ModelSaver::createDirectory(const std::string &base_dir) const {
  fs::path dir_path(base_dir);
  fs::create_directories(dir_path);
  return dir_path.string();
}

std::string ModelSaver::getCheckpointFilename(const std::string &directory,
                                              const std::string &model_name,
                                              int epoch) const {
  return directory + "/" + model_name + "_checkpoint_" + std::to_string(epoch);
}

bool ModelSaver::writeMetadata(const std::string &directory,
                               const std::string &model_name,
                               const TransformerConfig &config) const {
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

bool ModelSaver::readMetadata(const std::string &directory,
                              const std::string &model_name,
                              TransformerConfig &config) const {
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
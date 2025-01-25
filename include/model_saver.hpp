#pragma once

#include "logger.hpp"
#include "transformer.hpp"
#include "vocabulary.hpp"
#include <memory>
#include <string>
#include <nlohmann/json.hpp>

/**
 * @brief Manages model checkpointing and persistence.
 * 
 * The ModelSaver class provides functionality for saving and loading
 * transformer models and training checkpoints. Features include:
 * - Complete model state persistence
 * - Training checkpoint management
 * - Automatic directory creation
 * - Metadata handling
 * - Error logging and recovery
 */
class ModelSaver {
  public:
    /**
     * @brief Constructs a model saver instance.
     * 
     * Initializes logging and creates necessary directories
     * for model and checkpoint storage.
     */
    ModelSaver();

    /**
     * @brief Saves a complete model to disk.
     * 
     * Persists the entire model state including:
     * - Model architecture and configuration
     * - Trained parameters
     * - Tokenizer vocabulary
     * - Model metadata
     * 
     * @param transformer Model to save
     * @param directory Base directory for model storage
     * @param model_name Name of the model
     * @return true if save successful, false otherwise
     */
    bool saveModel(const Transformer& transformer, const std::string& directory,
                   const std::string& model_name);

    /**
     * @brief Loads a complete model from disk.
     * 
     * Restores the entire model state including:
     * - Model architecture and configuration
     * - Trained parameters
     * - Tokenizer vocabulary
     * - Model metadata
     * 
     * @param transformer Model to load into
     * @param directory Directory containing the model
     * @param model_name Name of the model to load
     * @return true if load successful, false otherwise
     */
    bool loadModel(Transformer& transformer, const std::string& directory,
                   const std::string& model_name);

    /**
     * @brief Saves a training checkpoint.
     * 
     * Creates a checkpoint containing:
     * - Current model state
     * - Training epoch
     * - Current loss value
     * - Optimizer state
     * 
     * @param transformer Current model state
     * @param directory Checkpoint directory
     * @param model_name Model identifier
     * @param epoch Current training epoch
     * @param loss Current loss value
     * @return true if checkpoint saved successfully
     */
    bool saveCheckpoint(const Transformer& transformer, const std::string& directory,
                        const std::string& model_name, int epoch, float loss);

    /**
     * @brief Loads the most recent checkpoint.
     * 
     * Finds and loads the latest checkpoint by epoch number.
     * 
     * @param transformer Model to restore
     * @param directory Checkpoint directory
     * @param model_name Model identifier
     * @param[out] epoch Restored epoch number
     * @param[out] loss Restored loss value
     * @return true if checkpoint loaded successfully
     */
    bool loadLatestCheckpoint(Transformer& transformer, const std::string& directory,
                              const std::string& model_name, int& epoch, float& loss);

    /**
     * @brief Loads a specific checkpoint.
     * 
     * @param transformer Model to restore
     * @param checkpoint_path Full path to checkpoint file
     * @return true if checkpoint loaded successfully
     */
    bool loadCheckpoint(Transformer& transformer, const std::string& checkpoint_path);

    /**
     * @brief Saves vocabulary to a file
     * @param path Path to save the vocabulary
     * @param vocab The vocabulary to save
     */
    void save_vocabulary(const std::string& path, const Vocabulary& vocab);

  private:
    Logger& logger;  ///< Logger for error and status messages
    nlohmann::json special_tokens_json;  // Add this member

    /**
     * @brief Creates directory structure for model storage.
     * @param base_dir Base directory path
     * @return Created directory path
     * @throws std::runtime_error if directory creation fails
     */
    std::string createDirectory(const std::string& base_dir) const;

    /**
     * @brief Generates checkpoint filename.
     * @param directory Checkpoint directory
     * @param model_name Model identifier
     * @param epoch Training epoch
     * @return Formatted checkpoint filename
     */
    std::string getCheckpointFilename(const std::string& directory, const std::string& model_name,
                                      int epoch) const;

    /**
     * @brief Writes model metadata to disk.
     * @param directory Target directory
     * @param model_name Model identifier
     * @param config Model configuration
     * @return true if write successful
     */
    bool writeMetadata(const std::string& directory, const std::string& model_name,
                       const TransformerConfig& config) const;

    /**
     * @brief Reads model metadata from disk.
     * @param directory Source directory
     * @param model_name Model identifier
     * @param[out] config Configuration to populate
     * @return true if read successful
     */
    bool readMetadata(const std::string& directory, const std::string& model_name,
                      TransformerConfig& config) const;
};
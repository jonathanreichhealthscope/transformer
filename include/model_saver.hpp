#ifndef MODEL_SAVER_HPP
#define MODEL_SAVER_HPP

#include "logger.hpp"
#include "transformer.hpp"
#include <memory>
#include <string>

class ModelSaver {
public:
  ModelSaver();

  // Save model to specified directory
  bool saveModel(const Transformer &transformer, const std::string &directory,
                 const std::string &model_name);

  // Load model from specified directory
  bool loadModel(Transformer &transformer, const std::string &directory,
                 const std::string &model_name);

  // Save checkpoint during training
  bool saveCheckpoint(const Transformer &transformer,
                      const std::string &directory,
                      const std::string &model_name, int epoch, float loss);

  // Load latest checkpoint
  bool loadLatestCheckpoint(Transformer &transformer,
                            const std::string &directory,
                            const std::string &model_name, int &epoch,
                            float &loss);

private:
  Logger &logger;

  // Helper functions
  std::string createDirectory(const std::string &base_dir) const;
  std::string getCheckpointFilename(const std::string &directory,
                                    const std::string &model_name,
                                    int epoch) const;
  bool writeMetadata(const std::string &directory,
                     const std::string &model_name,
                     const TransformerConfig &config) const;
  bool readMetadata(const std::string &directory, const std::string &model_name,
                    TransformerConfig &config) const;
};

#endif // MODEL_SAVER_HPP
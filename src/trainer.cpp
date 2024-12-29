#include "../include/trainer.hpp"
#include <fstream>

TransformerTrainer::TransformerTrainer(Transformer &model_, float learning_rate)
    : model(model_) {
  optimizer = std::make_unique<Optimizer>(learning_rate);

  // Add model parameters to optimizer
  for (auto &param : model.parameters()) {
    optimizer->add_parameter(param);
  }
}

void TransformerTrainer::backward_pass(const std::vector<Matrix> &activations,
                                       const Matrix &loss_grad) {
  std::vector<Matrix> layer_gradients;
  Matrix current_grad = loss_grad;

  // Backward through layers
  for (int i = static_cast<int>(model.layers.size()) - 1; i >= 0; --i) {
    current_grad = model.backward(current_grad, activations[i], i);
    layer_gradients.push_back(current_grad);
  }

  // Update parameters
  optimizer->update(model.parameters(), layer_gradients);
}

void TransformerTrainer::save_checkpoint(const std::string &path) const {
  std::ofstream os(path, std::ios::binary);
  if (!os) {
    throw std::runtime_error("Failed to open file for saving checkpoint");
  }

  // Save model and optimizer state
  model.save(os);
  optimizer->save(os);
}

void TransformerTrainer::load_checkpoint(const std::string &path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    throw std::runtime_error("Failed to open file for loading checkpoint");
  }

  // Load model and optimizer state
  model.load(is);
  optimizer->load(is);
}
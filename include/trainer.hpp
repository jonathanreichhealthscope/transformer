#pragma once
#include "optimizer.hpp"
#include "transformer.hpp"
#include <memory>

/**
 * @brief Training orchestrator for transformer models.
 * 
 * The TransformerTrainer class manages the training process for transformer
 * models, handling:
 * - Backpropagation through time
 * - Parameter updates via optimizer
 * - Checkpoint management
 * - Training state persistence
 * 
 * This implementation supports both full model training and fine-tuning
 * with configurable learning rates and optimization strategies.
 */
class TransformerTrainer {
  private:
    Transformer& model;                    ///< Reference to the model being trained
    std::unique_ptr<Optimizer> optimizer;  ///< Optimizer for parameter updates

  public:
    /**
     * @brief Constructs a trainer for a transformer model.
     * 
     * Initializes the training environment with the specified model
     * and learning rate, setting up the optimizer and any necessary
     * training state.
     * 
     * @param model_ Reference to the transformer model to train
     * @param learning_rate Initial learning rate for optimization
     */
    TransformerTrainer(Transformer& model_, float learning_rate);

    /**
     * @brief Performs the backward pass for gradient computation.
     * 
     * Computes gradients through the entire model using the provided
     * activations and loss gradient. The computed gradients are
     * accumulated in the optimizer for the next update step.
     * 
     * @param activations Vector of intermediate activations from forward pass
     * @param loss_grad Gradient of the loss with respect to model output
     */
    void backward_pass(const std::vector<Matrix>& activations, const Matrix& loss_grad);

    /**
     * @brief Saves the current training state.
     * 
     * Creates a checkpoint containing:
     * - Model parameters
     * - Optimizer state
     * - Training statistics
     * 
     * @param path Path where checkpoint should be saved
     */
    void save_checkpoint(const std::string& path) const;

    /**
     * @brief Restores a previous training state.
     * 
     * Loads a checkpoint to resume training from a previous state,
     * restoring:
     * - Model parameters
     * - Optimizer state
     * - Training statistics
     * 
     * @param path Path to the checkpoint file
     * @throws std::runtime_error if checkpoint cannot be loaded
     */
    void load_checkpoint(const std::string& path);
};
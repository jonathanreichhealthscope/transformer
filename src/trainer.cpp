#include "../include/trainer.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cuda_runtime.h>

TransformerTrainer::TransformerTrainer(Transformer& model_, float lr, size_t batch_size)
    : model(model_), 
      learning_rate(lr), 
      batch_size_(batch_size),
      use_cuda(true)  // Default to CUDA if available
{
    // Create Adam optimizer
    optimizer = std::make_unique<AdamOptimizer>(learning_rate);
    
    // Check CUDA availability
    int device_count;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        use_cuda = false;
    }
}

Matrix TransformerTrainer::compute_loss_gradients(
    const Matrix& logits, 
    const std::vector<int>& targets
) {
    const size_t vocab_size = logits.cols();
    const size_t seq_length = targets.size();
    Matrix gradients(seq_length, vocab_size);
    
    // Compute cross entropy loss gradients
    for (size_t i = 0; i < seq_length; ++i) {
        // Copy logits for this position
        std::vector<float> probs(vocab_size);
        float max_logit = -std::numeric_limits<float>::infinity();
        
        // Find max logit for numerical stability
        for (size_t j = 0; j < vocab_size; ++j) {
            max_logit = std::max(max_logit, logits(i, j));
        }
        
        // Compute softmax
        float sum_exp = 0.0f;
        for (size_t j = 0; j < vocab_size; ++j) {
            probs[j] = std::exp(logits(i, j) - max_logit);
            sum_exp += probs[j];
        }
        
        // Normalize and compute gradients
        for (size_t j = 0; j < vocab_size; ++j) {
            probs[j] /= sum_exp;
            // Gradient is (probability - 1) for correct class, probability for others
            gradients(i, j) = probs[j];
        }
        gradients(i, targets[i]) -= 1.0f;
    }
    
    return gradients;
}

void TransformerTrainer::backward_pass(
    const std::vector<Matrix>& activations,
    const Matrix& loss_grad
) {
    // Store gradients for each layer
    std::vector<Matrix> layer_gradients;
    layer_gradients.reserve(activations.size());
    
    // Backward pass through the model
    Matrix current_grad = loss_grad;
    
    for (int i = activations.size() - 1; i >= 0; --i) {
        if (use_cuda) {
            current_grad = model.backward_cuda(current_grad, activations[i], i);
        } else {
            current_grad = model.backward(current_grad, activations[i], i);
        }
        layer_gradients.push_back(current_grad);
    }
    
    // Update model parameters using the optimizer
    optimizer->update(model.parameters(), layer_gradients);
}

void TransformerTrainer::train_step(
    const std::vector<std::vector<int>>& input_batch,
    const std::vector<std::vector<int>>& target_batch
) {
    // Forward pass
    std::vector<Matrix> activations;
    Matrix logits;
    
    if (use_cuda) {
        logits = model.forward_cuda(input_batch[0], &activations);
    } else {
        logits = model.forward(input_batch[0], &activations);
    }
    
    const size_t batch_size = input_batch.size();
    Matrix batch_loss_grads;
    
    // Process each sequence in the batch
    for (size_t i = 0; i < batch_size; ++i) {
        // Compute loss gradients
        Matrix loss_grads = compute_loss_gradients(logits, target_batch[i]);
        
        // Store activations and gradients for backward pass
        activations.insert(activations.end(), 
                           activations.begin(), activations.end());
        
        if (i == 0) {
            batch_loss_grads = loss_grads;
        } else {
            // Accumulate gradients
            batch_loss_grads += loss_grads;
        }
    }
    
    // Average gradients over batch
    batch_loss_grads *= (1.0f / batch_size);
    
    // Backward pass with accumulated gradients
    backward_pass(activations, batch_loss_grads);
}

float TransformerTrainer::evaluate(
    const std::vector<std::vector<int>>& val_inputs,
    const std::vector<std::vector<int>>& val_targets
) {
    float total_loss = 0.0f;
    size_t total_tokens = 0;
    
    for (size_t i = 0; i < val_inputs.size(); ++i) {
        // Forward pass
        Matrix logits = use_cuda ? 
            model.forward_cuda(val_inputs[i]) : 
            model.forward(val_inputs[i]);
        
        // Compute loss
        const size_t seq_length = val_targets[i].size();
        for (size_t j = 0; j < seq_length; ++j) {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t k = 0; k < logits.cols(); ++k) {
                max_logit = std::max(max_logit, logits(j, k));
            }
            
            float sum_exp = 0.0f;
            for (size_t k = 0; k < logits.cols(); ++k) {
                sum_exp += std::exp(logits(j, k) - max_logit);
            }
            
            total_loss -= (logits(j, val_targets[i][j]) - max_logit - std::log(sum_exp));
        }
        
        total_tokens += seq_length;
    }
    
    return total_loss / total_tokens;
}

void TransformerTrainer::save_checkpoint(const std::string& path) const {
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("Failed to open file for saving checkpoint");
    }
    
    // Save model state
    model.save(os);
    
    // Save optimizer state
    optimizer->save(os);
    
    // Save trainer parameters
    os.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    os.write(reinterpret_cast<const char*>(&batch_size_), sizeof(batch_size_));
    os.write(reinterpret_cast<const char*>(&use_cuda), sizeof(use_cuda));
}

void TransformerTrainer::load_checkpoint(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("Failed to open file for loading checkpoint");
    }
    
    // Load model state
    model.load(is);
    
    // Load optimizer state
    optimizer->load(is);
    
    // Load trainer parameters
    is.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
    is.read(reinterpret_cast<char*>(&batch_size_), sizeof(batch_size_));
    is.read(reinterpret_cast<char*>(&use_cuda), sizeof(use_cuda));
} 
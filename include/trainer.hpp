#pragma once
#include <memory>
#include <vector>
#include <random>
#include "optimizer.hpp"

// Forward declaration
class Transformer;

class TransformerTrainer {
private:
    Transformer& model;
    std::unique_ptr<Optimizer> optimizer;
    float learning_rate;
    size_t batch_size_;
    bool use_cuda;
    
    // Training helpers
    Matrix compute_loss_gradients(const Matrix& logits, const std::vector<int>& targets);
    void backward_pass(const std::vector<Matrix>& activations, const Matrix& loss_grad);
    void update_parameters();
    
public:
    TransformerTrainer(Transformer& model, float learning_rate, size_t batch_size);
    size_t batch_size() const { return batch_size_; }
    
    void train_step(const std::vector<std::vector<int>>& input_batch,
                   const std::vector<std::vector<int>>& target_batch);
                   
    float evaluate(const std::vector<std::vector<int>>& val_inputs,
                  const std::vector<std::vector<int>>& val_targets);
                  
    void save_checkpoint(const std::string& path) const;
    void load_checkpoint(const std::string& path);
}; 
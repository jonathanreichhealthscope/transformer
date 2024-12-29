#pragma once
#include "transformer.hpp"
#include "optimizer.hpp"
#include <memory>

class TransformerTrainer {
private:
    Transformer& model;
    std::unique_ptr<Optimizer> optimizer;

public:
    TransformerTrainer(Transformer& model_, float learning_rate);
    
    void backward_pass(const std::vector<Matrix>& activations,
                      const Matrix& loss_grad);
    void save_checkpoint(const std::string& path) const;
    void load_checkpoint(const std::string& path);
}; 
#pragma once
#include "../matrix.hpp"
#include <vector>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    virtual void step(std::vector<Matrix*>& params, 
                     const std::vector<Matrix>& grads) = 0;
    
    virtual void zero_grad() = 0;
};

class SGD : public Optimizer {
private:
    float learning_rate;
    float momentum;
    std::vector<Matrix> velocity;

public:
    SGD(float lr = 0.001f, float momentum = 0.9f) 
        : learning_rate(lr), momentum(momentum) {}

    void step(std::vector<Matrix*>& params,
             const std::vector<Matrix>& grads) override;
             
    void zero_grad() override { velocity.clear(); }
};
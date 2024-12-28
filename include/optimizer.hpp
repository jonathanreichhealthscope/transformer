#pragma once
#include "components.hpp"
#include <vector>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(const std::vector<Matrix>& params, 
                       const std::vector<Matrix>& grads) = 0;
    virtual void save(std::ostream& os) const = 0;
    virtual void load(std::istream& is) = 0;
};

class AdamOptimizer : public Optimizer {
private:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    std::vector<Matrix> m;  // First moment
    std::vector<Matrix> v;  // Second moment
    size_t t;              // Time step

public:
    AdamOptimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f);
    void update(const std::vector<Matrix>& params, 
                const std::vector<Matrix>& grads) override;
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;
}; 
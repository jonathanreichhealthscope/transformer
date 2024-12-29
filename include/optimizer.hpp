#pragma once
#include "components.hpp"
#include <vector>

class Optimizer {
private:
  std::vector<Matrix *> parameters;
  std::vector<Matrix> gradients;
  float learning_rate;
  float beta1;
  float beta2;
  float epsilon;
  size_t t; // timestep

public:
  Optimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f,
            float eps = 1e-8f);
  void add_parameter(Matrix &param);
  void update(const std::vector<Matrix> &params,
              const std::vector<Matrix> &grads);
  void step();
  void zero_grad();
  void save(std::ostream &os) const;
  void load(std::istream &is);
};
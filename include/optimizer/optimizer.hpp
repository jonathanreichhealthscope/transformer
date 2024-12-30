#pragma once
#include "../components.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

class Optimizer {
public:
  virtual ~Optimizer() = default;
  virtual void step(std::vector<Matrix *> &params,
                    const std::vector<Matrix> &grads) = 0;
  virtual void zero_grad() = 0;
};

// Basic SGD Optimizer
class SGD : public Optimizer {
private:
  float learning_rate;
  float momentum;
  std::vector<Matrix> velocity;

public:
  explicit SGD(float lr = 0.001f, float momentum = 0.9f)
      : learning_rate(lr), momentum(momentum) {}

  void step(std::vector<Matrix *> &params,
            const std::vector<Matrix> &grads) override {
    if (velocity.empty()) {
      // Initialize velocity vectors
      for (const auto &grad : grads) {
        velocity.push_back(Matrix(grad.rows(), grad.cols(), 0.0f));
      }
    }

    for (size_t i = 0; i < params.size(); ++i) {
      // Update velocity
      for (size_t r = 0; r < velocity[i].rows(); ++r) {
        for (size_t c = 0; c < velocity[i].cols(); ++c) {
          velocity[i](r, c) =
              momentum * velocity[i](r, c) + learning_rate * grads[i](r, c);
        }
      }

      // Update parameters
      for (size_t r = 0; r < params[i]->rows(); ++r) {
        for (size_t c = 0; c < params[i]->cols(); ++c) {
          (*params[i])(r, c) -= velocity[i](r, c);
        }
      }
    }
  }

  void zero_grad() override { velocity.clear(); }
};
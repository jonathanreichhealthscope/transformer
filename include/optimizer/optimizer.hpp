#pragma once
#include "../components.hpp"
#include <unordered_map>
#include <vector>

class Optimizer {
public:
  virtual ~Optimizer() = default;
  virtual void step(const std::vector<Matrix *> &params,
                    const std::vector<Matrix> &grads) = 0;
  virtual void zero_grad() = 0;
};

class AdaFactor : public Optimizer {
private:
  struct AdaFactorState {
    Matrix v_row;
    Matrix v_col;
    float beta1;
    float eps1;
    float eps2;
    size_t step;
  };

  std::unordered_map<const Matrix *, AdaFactorState> states;
  float learning_rate;
  float clip_threshold;

  void update_factored_moments(const Matrix &param, const Matrix &grad,
                               AdaFactorState &state) {
    // Update row-wise second moment
    Matrix row_squared = Matrix(grad.rows(), 1);
    for (size_t i = 0; i < grad.rows(); ++i) {
      float sum = 0.0f;
      for (size_t j = 0; j < grad.cols(); ++j) {
        sum += grad(i, j) * grad(i, j);
      }
      row_squared(i, 0) = sum / grad.cols();
    }
    state.v_row =
        state.v_row * state.beta1 + row_squared * (1.0f - state.beta1);

    // Update column-wise second moment
    Matrix col_squared = Matrix(1, grad.cols());
    for (size_t j = 0; j < grad.cols(); ++j) {
      float sum = 0.0f;
      for (size_t i = 0; i < grad.rows(); ++i) {
        sum += grad(i, j) * grad(i, j);
      }
      col_squared(0, j) = sum / grad.rows();
    }
    state.v_col =
        state.v_col * state.beta1 + col_squared * (1.0f - state.beta1);
  }

  float compute_adaptive_lr(const AdaFactorState &state) {
    float step_size =
        learning_rate * std::sqrt(1.0f - std::pow(state.beta1, state.step));
    return step_size;
  }

  void apply_update(Matrix &param, const Matrix &grad,
                    const AdaFactorState &state, float lr) {
    for (size_t i = 0; i < param.rows(); ++i) {
      for (size_t j = 0; j < param.cols(); ++j) {
        float update =
            grad(i, j) * lr /
            (std::sqrt(state.v_row(i, 0) * state.v_col(0, j)) + state.eps1);
        if (std::abs(update) > clip_threshold) {
          update = std::copysign(clip_threshold, update);
        }
        param(i, j) -= update;
      }
    }
  }

  void update(Matrix &param, const Matrix &grad) {
    auto &state = states[&param];
    state.step++;

    update_factored_moments(param, grad, state);
    float lr = compute_adaptive_lr(state);
    apply_update(param, grad, state, lr);
  }

public:
  AdaFactor(float lr = 0.01f, float clip = 1.0f)
      : learning_rate(lr), clip_threshold(clip) {}

  void step(const std::vector<Matrix *> &params,
            const std::vector<Matrix> &grads) override {
    for (size_t i = 0; i < params.size(); ++i) {
      update(*params[i], grads[i]);
    }
  }

  void zero_grad() override {}
};
#pragma once
#include "matrix.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>
#include <cmath>

// Token category structure
struct TokenCategories {
    std::unordered_set<std::string> verb_tokens;
    std::unordered_set<std::string> adjective_tokens;
    std::unordered_set<std::string> noun_tokens;
};

class Utils {
  public:
    static float adjust_learning_rate(float current_lr, float loss_ratio, size_t step);
    static bool validate_input_sequence(const std::vector<int>& tokens, size_t vocab_size,
                                        size_t max_seq_length = 512);
    static void print_matrix(const Matrix& m, const std::string& name, size_t max_rows = 5,
                             size_t max_cols = 5);
    static void print_top_predictions(const Matrix& logits, const Tokenizer& tokenizer, 
                                    Transformer& transformer, int k);
    static std::vector<std::pair<std::string, std::string>> create_training_data();
    static void
    analyze_token_mappings(const std::vector<std::pair<std::string, std::string>>& training_data,
                           const Tokenizer& tokenizer);
    static std::vector<std::pair<std::string, std::string>> load_validation_data();
    static float
    evaluate_validation(Transformer& transformer, const Tokenizer& tokenizer,
                        const std::vector<std::pair<std::string, std::string>>& validation_data);
    static TransformerConfig load_config(const std::string& config_path);
    static Matrix
    create_batch_target_distribution(const std::vector<std::vector<int>>& target_tokens,
                                     const Tokenizer& tokenizer, size_t vocab_size,
                                     size_t input_max_seq_len);
    static float compute_batch_loss(const Matrix& logits, const Matrix& target_distribution, const Tokenizer& tokenizer);
    static void apply_sampling_parameters(std::vector<float>& logits, float temperature,
                                          float top_p);
    static std::vector<std::string>& get_vocabulary(const Tokenizer& tokenizer);
    static std::vector<std::pair<std::string, float>> get_multi_token_predictions(
        const Matrix& logits, const Tokenizer& tokenizer, int beam_width);
    
    // Token category analysis functions
    static TokenCategories analyze_token_categories(const std::vector<std::pair<std::string, std::string>>& training_data);
    static std::string get_token_category(const std::string& token, const TokenCategories& categories);
    static void trim(std::string& s);

    // Add inline utility functions for gradient computation
    static inline float compute_grad_norm(const Matrix& grad) {
        float norm = 0.0f;
        #pragma omp parallel for reduction(+:norm)
        for (size_t i = 0; i < grad.rows(); ++i) {
            for (size_t j = 0; j < grad.cols(); ++j) {
                norm += grad(i, j) * grad(i, j);
            }
        }
        return std::sqrt(norm);
    }

    static inline size_t count_params(const Matrix& param) {
        return param.rows() * param.cols();
    }

    // Add loss computation functions
    static inline float compute_loss(const Matrix& output, const Matrix& target_distribution) {
        if (output.size() != target_distribution.size()) {
            throw std::runtime_error("Output and target distribution must have the same size");
        }

        const size_t batch_size = output.rows();
        const size_t vocab_size = output.cols();
        float total_loss = 0.0f;

        #pragma omp parallel for reduction(+:total_loss)
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < vocab_size; ++j) {
                if (target_distribution(i, j) > 0.0f) {
                    const float epsilon = 1e-10f;
                    float pred = std::clamp(output(i, j), epsilon, 1.0f - epsilon);
                    total_loss -= target_distribution(i, j) * std::log(pred);
                }
            }
        }

        return total_loss / static_cast<float>(batch_size);
    }

    static inline Matrix compute_loss_gradient(const Matrix& output, const Matrix& target_distribution) {
        if (output.size() != target_distribution.size()) {
            throw std::runtime_error("Output and target distribution must have the same size");
        }

        const size_t batch_size = output.rows();
        const size_t vocab_size = output.cols();
        Matrix gradient(batch_size, vocab_size);

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < vocab_size; ++j) {
                if (target_distribution(i, j) > 0.0f) {
                    const float epsilon = 1e-10f;
                    float pred = std::clamp(output(i, j), epsilon, 1.0f - epsilon);
                    gradient(i, j) = (pred - target_distribution(i, j)) / (pred * (1.0f - pred));
                } else {
                    gradient(i, j) = 0.0f;
                }
            }
        }

        return gradient;
    }
};
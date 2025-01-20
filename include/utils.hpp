#pragma once
#include <string>
#include <vector>
#include <utility>
#include "matrix.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"

class Utils {
public:
    static float adjust_learning_rate(float current_lr, float loss_ratio, size_t step);
    static bool validate_input_sequence(const std::vector<int>& tokens, size_t vocab_size, size_t max_seq_length = 512);
    static void print_matrix(const Matrix& m, const std::string& name, size_t max_rows = 5, size_t max_cols = 5);
    static void print_top_predictions(const Matrix& logits, const Tokenizer& tokenizer, size_t k = 5);
    static std::vector<std::pair<std::string, std::string>> create_training_data();
    static void analyze_token_mappings(const std::vector<std::pair<std::string, std::string>>& training_data, 
                                     const Tokenizer& tokenizer);
    static std::vector<std::pair<std::string, std::string>> load_validation_data();
    static float evaluate_validation(Transformer& transformer, const Tokenizer& tokenizer,
                                   const std::vector<std::pair<std::string, std::string>>& validation_data);
    static TransformerConfig load_config(const std::string& config_path);
    static Matrix create_batch_target_distribution(const std::vector<std::vector<int>>& target_tokens,
                                                 const Tokenizer& tokenizer, size_t vocab_size);
    static float compute_batch_loss(const Matrix& logits, const Matrix& target_distribution);
    static void apply_sampling_parameters(std::vector<float>& logits, 
                                        float temperature, 
                                        float top_p);
}; 
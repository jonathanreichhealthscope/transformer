#pragma once
#include "matrix.hpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <random>
#include <atomic>
#include <chrono>

// Token category structure
struct TokenCategories {
    std::unordered_set<std::string> verb_tokens;
    std::unordered_set<std::string> adjective_tokens;
    std::unordered_set<std::string> noun_tokens;
};

class Utils {
private:
    static std::random_device rd;  // Hardware random number source
    static std::mt19937 random_generator;
    static std::atomic<uint64_t> prediction_counter;  // Counter for unique seeds

    // Randomization helpers
    static float apply_temperature_scaling(
        std::vector<float>& logits,
        float temperature,
        std::mt19937& gen
    );

    static void add_random_variation(
        std::vector<float>& probabilities,
        std::mt19937& gen,
        float min_var = 0.8f,
        float max_var = 1.2f
    );

    static std::vector<std::pair<float, int>> apply_nucleus_sampling(
        const std::vector<std::pair<float, int>>& token_probs,
        float p,
        std::mt19937& gen
    );

public:
    static float adjust_learning_rate(float current_lr, float loss_ratio, size_t step);
    static bool validate_input_sequence(const std::vector<int>& tokens, size_t vocab_size,
                                        size_t max_seq_length = 512);
    static void print_matrix(const Matrix& m, const std::string& name, size_t max_rows = 5,
                             size_t max_cols = 5);
    static void print_top_predictions(
        const Matrix& logits,
        const Tokenizer& tokenizer,
        Transformer& transformer,
        int k,
        std::mt19937* gen = nullptr
    );
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

    // Random number generation utilities
    static void set_random_generator(const std::mt19937& gen) {
        random_generator = gen;
    }
    
    static std::mt19937& get_random_generator() {
        return random_generator;
    }
    
    static float random_float(float min = 0.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(random_generator);
    }
    
    static int random_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(random_generator);
    }
    
    // Get a new random generator with unique seed
    static std::mt19937 get_new_generator() {
        // Combine multiple entropy sources
        auto time_seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        auto counter = prediction_counter.fetch_add(1, std::memory_order_relaxed);
        auto hw_rand = static_cast<uint64_t>(rd());
        
        // Create seed sequence from multiple sources
        std::seed_seq seq{
            static_cast<uint32_t>(time_seed),
            static_cast<uint32_t>(time_seed >> 32),
            static_cast<uint32_t>(counter),
            static_cast<uint32_t>(hw_rand),
            static_cast<uint32_t>(hw_rand >> 32),
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&counter))  // Use address as additional entropy
        };
        
        return std::mt19937(seq);
    }

    // Initialize random generator with time-based seed
    static void initialize_random() {
        auto time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::seed_seq seq{static_cast<uint32_t>(time_seed & 0xFFFFFFFF)};
        random_generator = std::mt19937(seq);
        prediction_counter = 0;
    }
};
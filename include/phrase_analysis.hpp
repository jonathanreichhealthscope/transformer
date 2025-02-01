#pragma once
#include "matrix.hpp"
#include "tokenizer.hpp"
#include "phrase_types.hpp"

// Penalty computation functions
float compute_verb_penalty(const Matrix& logits, const std::vector<int>& final_tokens,
                         const Tokenizer& tokenizer);

float compute_adjective_penalty(const Matrix& logits, const std::vector<int>& final_tokens,
                              const Tokenizer& tokenizer);

// Gradient factor computation functions
float verb_gradient_factor(size_t position, const std::vector<int>& tokens,
                         const Tokenizer& tokenizer);

float adjective_gradient_factor(size_t position, const std::vector<int>& tokens,
                              const Tokenizer& tokenizer); 
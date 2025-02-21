#include "../include/phrase_analysis.hpp"

float compute_verb_penalty(const Matrix& logits, const std::vector<int>& final_tokens,
                         const Tokenizer& tokenizer) {
    // Get the final token's logits
    Vector final_token_logits = logits.row(logits.rows() - 1);
    float penalty = 0.0f;
    
    // Apply penalty if the predicted token is not verb-like
    // This is a simple implementation - you might want to enhance it based on your needs
    std::string token_text = tokenizer.decode(final_tokens);
    if (!token_text.empty()) {
        // Common verb endings - this is a basic heuristic
        const std::vector<std::string> verb_endings = {"ing", "ed", "ate", "ize", "ify"};
        bool is_verb = false;
        for (const auto& ending : verb_endings) {
            if (token_text.length() > ending.length() && 
                token_text.substr(token_text.length() - ending.length()) == ending) {
                is_verb = true;
                break;
            }
        }
        if (!is_verb) {
            penalty = 0.5f; // Penalty weight for non-verb predictions
        }
    }
    
    return penalty;
}

float compute_adjective_penalty(const Matrix& logits, const std::vector<int>& final_tokens,
                              const Tokenizer& tokenizer) {
    // Get the final token's logits
    Vector final_token_logits = logits.row(logits.rows() - 1);
    float penalty = 0.0f;
    
    // Apply penalty if the predicted token is not adjective-like
    // This is a simple implementation - you might want to enhance it based on your needs
    std::string token_text = tokenizer.decode(final_tokens);
    if (!token_text.empty()) {
        // Common adjective endings - this is a basic heuristic
        const std::vector<std::string> adj_endings = {"ful", "ous", "ible", "able", "al", "ive"};
        bool is_adjective = false;
        for (const auto& ending : adj_endings) {
            if (token_text.length() > ending.length() && 
                token_text.substr(token_text.length() - ending.length()) == ending) {
                is_adjective = true;
                break;
            }
        }
        if (!is_adjective) {
            penalty = 0.5f; // Penalty weight for non-adjective predictions
        }
    }
    
    return penalty;
}

float verb_gradient_factor(size_t position, const std::vector<int>& tokens,
                         const Tokenizer& tokenizer) {
    // Increase gradient impact for verb-like tokens
    // This is a simple implementation - you might want to enhance it based on your needs
    if (position < tokens.size()) {
        std::string token_text = tokenizer.decode({tokens[position]});
        const std::vector<std::string> verb_endings = {"ing", "ed", "ate", "ize", "ify"};
        for (const auto& ending : verb_endings) {
            if (token_text.length() > ending.length() && 
                token_text.substr(token_text.length() - ending.length()) == ending) {
                return 1.5f; // Boost gradient for verb-like tokens
            }
        }
    }
    return 1.0f; // Default gradient factor
}

float adjective_gradient_factor(size_t position, const std::vector<int>& tokens,
                              const Tokenizer& tokenizer) {
    // Increase gradient impact for adjective-like tokens
    // This is a simple implementation - you might want to enhance it based on your needs
    if (position < tokens.size()) {
        std::string token_text = tokenizer.decode({tokens[position]});
        const std::vector<std::string> adj_endings = {"ful", "ous", "ible", "able", "al", "ive"};
        for (const auto& ending : adj_endings) {
            if (token_text.length() > ending.length() && 
                token_text.substr(token_text.length() - ending.length()) == ending) {
                return 1.5f; // Boost gradient for adjective-like tokens
            }
        }
    }
    return 1.0f; // Default gradient factor
} 
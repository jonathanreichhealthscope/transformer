#include "../include/components.hpp"
#include "../include/transformer.hpp"
#include <iostream>

int main() {
    // Create a small transformer model
    int vocab_size = 50000;
    int max_seq_length = 1024;
    int embed_dim = 768;
    int num_layers = 12;
    int num_heads = 12;
    
    Transformer model(vocab_size, max_seq_length, embed_dim, num_layers, num_heads);
    
    // Example input sequence
    std::vector<int> input_tokens = {1, 2, 3, 4, 5}; // Example token IDs
    
    // Get next token probabilities
    Vector probs = model.get_next_token_probabilities(input_tokens);
    
    // Print top 5 most likely next tokens
    std::vector<std::pair<float, int>> top_tokens;
    for (int i = 0; i < probs.size(); i++) {
        top_tokens.push_back({probs[i], i});
    }
    
    std::sort(top_tokens.begin(), top_tokens.end(), std::greater<>());
    
    std::cout << "Top 5 most likely next tokens:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "Token " << top_tokens[i].second 
                  << ": " << top_tokens[i].first << "\n";
    }
    
    return 0;
} 
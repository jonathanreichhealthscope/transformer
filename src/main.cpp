#include "../include/transformer.hpp"
#include <iostream>
#include <vector>

int main() {
    // Create transformer configuration
    TransformerConfig config(
        50000,    // vocab_size
        2048,     // max_seq_length
        768,      // hidden_size
        12,       // num_layers
        12        // num_heads
    );
    
    // Create transformer model
    Transformer model(config);
    
    // Example input tokens
    std::vector<int> input_tokens = {1, 2, 3, 4};  // Replace with actual token IDs
    
    // Forward pass
    Matrix output = model.forward(input_tokens);
    
    // Get probabilities for next token
    // Create a single-row matrix from the last row of output
    Matrix logits(1, output.cols());
    Vector last_row = output.row(output.rows() - 1);
    for (size_t i = 0; i < output.cols(); ++i) {
        logits(0, i) = last_row[i];
    }
    
    logits.apply_softmax();  // Convert to probabilities
    
    // Print top-k probabilities
    const int k = 5;
    std::vector<std::pair<float, int>> probs;
    for (size_t i = 0; i < logits.cols(); ++i) {
        probs.push_back({logits(0, i), static_cast<int>(i)});
    }
    
    std::partial_sort(probs.begin(), probs.begin() + k, probs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "Top " << k << " next token probabilities:\n";
    for (int i = 0; i < k; ++i) {
        std::cout << "Token " << probs[i].second << ": " << probs[i].first << "\n";
    }
    
    return 0;
} 
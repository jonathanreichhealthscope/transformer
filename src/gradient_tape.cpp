#include "../include/gradient_tape.hpp"
#include <cmath>
#include <iostream>

void GradientTape::record_operation(const Matrix& output,
                                  const std::vector<Matrix*>& inputs,
                                  const std::string& op_type) {
    operations.push_back({output, inputs, op_type});
}

Matrix GradientTape::compute_gradients(const Matrix& final_output,
                                     const Matrix& target_distribution) {
    if (operations.empty()) {
        throw std::runtime_error("No operations recorded in gradient tape");
    }
    
    // Initialize gradients
    Matrix gradients(final_output.rows(), final_output.cols());
    
    // Compute initial gradient from loss function
    for(size_t i = 0; i < final_output.size(); i++) {
        if (target_distribution.data()[i] > 0.0f) {
            gradients.data()[i] = final_output.data()[i] - target_distribution.data()[i];
        }
    }
    
    // Backpropagate through recorded operations in reverse order
    for(auto it = operations.rbegin(); it != operations.rend(); ++it) {
        const Operation& op = *it;
        
        if(op.op_type == "matmul") {
            // Handle matrix multiplication gradients
            // dC/dA = dC/dY * B^T
            // dC/dB = A^T * dC/dY
            // ... implement matrix multiplication gradients
        }
        else if(op.op_type == "add") {
            // Handle addition gradients
            // Gradient flows unchanged through addition
        }
        else if(op.op_type == "activation") {
            // Handle activation function gradients
            // Depends on the specific activation function used
        }
    }
    
    return gradients;
}

void GradientTape::clear() {
    operations.clear();
} 
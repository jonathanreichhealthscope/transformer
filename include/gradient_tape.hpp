#ifndef GRADIENT_TAPE_HPP
#define GRADIENT_TAPE_HPP

#include "matrix.hpp"
#include <vector>
#include <string>

class GradientTape {
public:
    struct Operation {
        Matrix output;
        std::vector<Matrix*> inputs;
        std::string op_type;
    };
    
    std::vector<Operation> operations;
    
    void record_operation(const Matrix& output, 
                         const std::vector<Matrix*>& inputs,
                         const std::string& op_type);
                         
    // Add additional methods for gradient computation
    Matrix compute_gradients(const Matrix& final_output, 
                           const Matrix& target);
                           
    void clear();
    
    // Destructor to clean up any resources
    ~GradientTape() = default;
};

#endif // GRADIENT_TAPE_HPP 
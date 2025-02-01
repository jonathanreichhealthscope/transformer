#include "count_vocabulary.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace transformer {

void VocabularyCounter::processFile(const std::string& filename, std::set<std::string>& unique_tokens) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        // Split line by whitespace and add each token to the set
        while (iss >> token) {
            unique_tokens.insert(token);
        }
    }
}

size_t VocabularyCounter::countUniqueTokens(const std::string& training_file, const std::string& validation_file) {
    std::set<std::string> unique_tokens;
    
    // Process both files
    processFile(training_file, unique_tokens);
    processFile(validation_file, unique_tokens);
    
    return unique_tokens.size();
}
} 
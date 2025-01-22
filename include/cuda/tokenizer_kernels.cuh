#pragma once
#include "../vocabulary.hpp"
#include <vector>
#include <string>

namespace cuda {
    void parallel_tokenize(const std::string& text, const Vocabulary& vocab, std::vector<int>& tokens);
} 
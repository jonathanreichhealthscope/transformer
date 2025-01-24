#pragma once
#include <string>

namespace tokens {
    // Special token IDs - matching transformer_config.json
    constexpr int PAD_ID = 0;  // Must be 0 per config
    constexpr int UNK_ID = 1;  // Must be 1 per config
    constexpr int BOS_ID = 2;  // Beginning of sequence
    constexpr int EOS_ID = 3;  // End of sequence
    constexpr int MASK_ID = 4; // Mask token for MLM

    // Special token strings
    const std::string PAD_TOKEN = "<pad>";
    const std::string UNK_TOKEN = "<unk>";
    const std::string BOS_TOKEN = "<s>";
    const std::string EOS_TOKEN = "</s>";
    const std::string MASK_TOKEN = "<mask>";

    constexpr int NUM_SPECIAL_TOKENS = 5;
} 
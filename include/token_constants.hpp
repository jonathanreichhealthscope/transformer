#pragma once
#include <string>

namespace tokens {
    // Special token IDs - order matters and must match SentencePiece's expectations
    constexpr int UNK_ID = 0;  // Must be 0 for SentencePiece
    constexpr int PAD_ID = 1;  // Padding token
    constexpr int BOS_ID = 2;  // Beginning of sequence
    constexpr int EOS_ID = 3;  // End of sequence
    constexpr int MASK_ID = 4; // Mask token for MLM

    // Special token strings - must match SentencePiece's expectations
    const std::string UNK_TOKEN = "<unk>";  // SentencePiece default UNK token
    const std::string PAD_TOKEN = "<pad>";
    const std::string BOS_TOKEN = "<s>";    // SentencePiece default BOS token
    const std::string EOS_TOKEN = "</s>";   // SentencePiece default EOS token
    const std::string MASK_TOKEN = "<mask>";

    constexpr int NUM_SPECIAL_TOKENS = 5;
} 
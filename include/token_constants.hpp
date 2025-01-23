#pragma once

namespace tokens {
    // Special token IDs
    static constexpr int PAD_ID = 0;
    static constexpr int UNK_ID = 1;
    static constexpr int BOS_ID = 2;
    static constexpr int EOS_ID = 3;
    static constexpr int MASK_ID = 4;
    static constexpr int NUM_SPECIAL_TOKENS = 5;

    // Special token strings
    static const char* PAD_TOKEN = "<pad>";
    static const char* UNK_TOKEN = "<unk>";
    static const char* BOS_TOKEN = "<bos>";
    static const char* EOS_TOKEN = "<eos>";
    static const char* MASK_TOKEN = "<mask>";
} 
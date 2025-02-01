#pragma once
#include <string>
#include <unordered_map>

enum class PhraseType {
    GENERAL,    // General phrases with | delimiter
    VERB,       // Verb endings with # delimiter
    ADJECTIVE,  // Adjective endings with * delimiter
};

class PhraseTypeHandler {
public:
    static constexpr const char* GENERAL_DELIMITER = "|";
    static constexpr const char* VERB_DELIMITER = "#";
    static constexpr const char* ADJECTIVE_DELIMITER = "*";
    
    static PhraseType detect_phrase_type(const std::string& text) {
        if (text.find(VERB_DELIMITER) != std::string::npos) {
            return PhraseType::VERB;
        } else if (text.find(ADJECTIVE_DELIMITER) != std::string::npos) {
            return PhraseType::ADJECTIVE;
        }
        return PhraseType::GENERAL;
    }
    
    static std::string get_delimiter(PhraseType type) {
        switch (type) {
            case PhraseType::VERB:
                return VERB_DELIMITER;
            case PhraseType::ADJECTIVE:
                return ADJECTIVE_DELIMITER;
            default:
                return GENERAL_DELIMITER;
        }
    }
    
    static std::string extract_final_phrase(const std::string& text) {
        size_t pos;
        if ((pos = text.find(VERB_DELIMITER)) != std::string::npos ||
            (pos = text.find(ADJECTIVE_DELIMITER)) != std::string::npos ||
            (pos = text.find(GENERAL_DELIMITER)) != std::string::npos) {
            return text.substr(pos + 1);
        }
        return "";
    }
}; 
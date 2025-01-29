#include "../include/text_analysis.hpp"
#include <sstream>

// Initialize static members
const std::vector<std::string> AdjectivePhraseAnalyzer::adjective_suffixes = {
    "able", "ible", "al", "ful", "ic", "ive", "less", "ous", "y", "ent", "ant"
};

const std::unordered_set<std::string> AdjectivePhraseAnalyzer::common_adjectives = {
    // Colors
    "red", "blue", "green", "yellow", "black", "white", "purple", "orange", "brown", "grey", "gray",
    // Sizes
    "big", "small", "large", "tiny", "huge", "massive", "little", "giant", "enormous", "miniature",
    // Qualities
    "good", "bad", "great", "poor", "excellent", "terrible", "wonderful", "awful", "perfect", "horrible",
    // States
    "hot", "cold", "warm", "cool", "wet", "dry", "clean", "dirty", "fresh", "stale",
    // Emotions
    "happy", "sad", "angry", "calm", "excited", "bored", "tired", "energetic", "peaceful", "anxious",
    // Appearances
    "beautiful", "ugly", "pretty", "handsome", "gorgeous", "plain", "elegant", "shabby", "attractive", "unattractive",
    // Textures
    "smooth", "rough", "soft", "hard", "silky", "coarse", "bumpy", "fluffy", "crisp", "fuzzy",
    // Shapes
    "round", "square", "oval", "rectangular", "circular", "triangular", "flat", "curved", "straight", "crooked",
    // Ages
    "new", "old", "young", "ancient", "modern", "fresh", "aged", "recent", "vintage", "contemporary",
    // Values
    "expensive", "cheap", "valuable", "worthless", "precious", "priceless", "costly", "affordable", "reasonable", "overpriced",
    // Intelligence
    "smart", "clever", "intelligent", "wise", "brilliant", "dumb", "stupid", "slow", "sharp", "bright",
    // Speed
    "fast", "slow", "quick", "rapid", "swift", "sluggish", "speedy", "leisurely", "hasty", "prompt",
    // Common states
    "open", "closed", "empty", "full", "light", "heavy", "deep", "shallow", "thick", "thin",
    // Difficulty
    "easy", "hard", "difficult", "simple", "complex", "complicated", "straightforward", "challenging", "demanding", "effortless",
    // Importance
    "important", "unimportant", "crucial", "vital", "essential", "trivial", "significant", "minor", "major", "critical",
    // Frequency
    "common", "rare", "frequent", "occasional", "regular", "unusual", "unique", "typical", "normal", "strange",
    // Certainty
    "sure", "uncertain", "definite", "clear", "obvious", "vague", "apparent", "evident", "doubtful", "questionable",
    // Necessity
    "necessary", "unnecessary", "needed", "optional", "required", "essential", "vital", "dispensable", "mandatory", "voluntary",
    // Completeness
    "complete", "incomplete", "whole", "partial", "finished", "unfinished", "total", "entire", "full", "partial"
};

bool AdjectivePhraseAnalyzer::isAdjective(const std::string& word) {
    // Convert word to lowercase for comparison
    std::string lower_word = word;
    std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
    
    // Check if it's in our common adjectives list
    if (common_adjectives.find(lower_word) != common_adjectives.end()) {
        return true;
    }
    
    // Check if it ends with common adjective suffixes
    for (const auto& suffix : adjective_suffixes) {
        if (lower_word.length() > suffix.length() && 
            lower_word.substr(lower_word.length() - suffix.length()) == suffix) {
            return true;
        }
    }
    
    return false;
}

bool AdjectivePhraseAnalyzer::endsWithAdjective(const std::string& phrase) {
    std::istringstream iss(phrase);
    std::vector<std::string> words;
    std::string word;
    
    // Split phrase into words
    while (iss >> word) {
        words.push_back(word);
    }
    
    // Check if the last word is an adjective
    if (!words.empty()) {
        return isAdjective(words.back());
    }
    
    return false;
}

std::vector<std::string> AdjectivePhraseAnalyzer::extractAdjectivePhrases(const std::string& filename) {
    std::vector<std::string> phrases;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        size_t separator_pos = line.find("|");
        if (separator_pos != std::string::npos) {
            // Get the part after the separator and trim whitespace
            std::string phrase = line.substr(separator_pos + 1);
            phrase = trim(phrase);
            
            // Only add phrases that end with an adjective
            if (endsWithAdjective(phrase)) {
                phrases.push_back(phrase);
            }
        }
    }
    
    return phrases;
}

void AdjectivePhraseAnalyzer::analyzeAndLogPhrases(const std::vector<std::string>& phrases, std::ofstream& log_file) {
    // Count unique phrases
    std::set<std::string> unique_phrases(phrases.begin(), phrases.end());
    
    // Count frequencies
    std::map<std::string, int> phrase_frequencies;
    for (const auto& phrase : phrases) {
        phrase_frequencies[phrase]++;
    }
    
    // Sort by frequency
    std::vector<std::pair<std::string, int>> sorted_phrases(
        phrase_frequencies.begin(), phrase_frequencies.end()
    );
    std::sort(sorted_phrases.begin(), sorted_phrases.end(),
             [](const auto& a, const auto& b) {
                 return a.second > b.second;
             });
    
    // Log unique phrases count
    log_file << "Found " << unique_phrases.size() << " unique adjective phrases\n\n";
    
    // Log top 10 most common phrases
    log_file << "Top 10 most common adjective phrases:\n";
    for (size_t i = 0; i < std::min(size_t(10), sorted_phrases.size()); ++i) {
        log_file << sorted_phrases[i].first << ": " 
                << sorted_phrases[i].second << " occurrences\n";
    }
    log_file << "\n";
}

std::string AdjectivePhraseAnalyzer::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
} 
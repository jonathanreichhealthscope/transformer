#include "../include/vocabulary.hpp"
#include "../include/token_constants.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_set>

Vocabulary::Vocabulary() {
    // Initialize special tokens in consistent order using constants
    add_special_token(tokens::PAD_TOKEN, tokens::PAD_ID);
    add_special_token(tokens::UNK_TOKEN, tokens::UNK_ID);
    add_special_token(tokens::BOS_TOKEN, tokens::BOS_ID);
    add_special_token(tokens::EOS_TOKEN, tokens::EOS_ID);
    add_special_token(tokens::MASK_TOKEN, tokens::MASK_ID);

    // Update member variables
    pad_token_id = tokens::PAD_ID;
    unk_token_id = tokens::UNK_ID;
    bos_token_id = tokens::BOS_ID;
    eos_token_id = tokens::EOS_ID;
    mask_token_id = tokens::MASK_ID;

    initialize_basic_vocabulary();
}

void Vocabulary::add_word(const std::string& word) {
    if (token_to_id.find(word) == token_to_id.end()) {
        int id = id_to_token.size();
        token_to_id[word] = id;
        id_to_token[id] = word;
    }
}

void Vocabulary::add_special_token(const std::string& token, int id) {
    token_to_id[token] = id;
    id_to_token[id] = token;
}

void Vocabulary::initialize_basic_vocabulary() {
    // Add articles and determiners first (before pronouns)
    std::vector<std::string> articles = {
        // Articles
        "a", "an", "the",

        // Demonstrative determiners
        "this", "that", "these", "those",

        // Possessive determiners
        "my", "your", "his", "her", "its", "our", "their",

        // Quantifiers and other determiners
        "all", "any", "both", "each", "every", "few", "many", "much", "several", "some", "such",
        "no", "none", "neither", "either",

        // Numbers as determiners
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "first",
        "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",

        // Other common determiners
        "another", "other", "what", "whatever", "which", "whichever", "whose", "enough", "various",
        "certain", "plenty", "lots", "most", "least", "last", "next", "previous", "same", "certain",

        // Distributive determiners
        "each", "every", "either", "neither"};

    // Basic pronouns and their contractions
    std::vector<std::string> pronouns = {
        "i",        "me",         "my",         "mine",      "myself",    "you",      "your",
        "yours",    "yourself",   "he",         "him",       "his",       "himself",  "she",
        "her",      "hers",       "herself",    "it",        "its",       "itself",   "we",
        "us",       "our",        "ours",       "ourselves", "they",      "them",     "their",
        "theirs",   "themselves", "this",       "that",      "these",     "those",    "who",
        "whom",     "whose",      "which",      "what",      "whatever",  "whoever",  "whomever",
        "anyone",   "everyone",   "someone",    "nobody",    "everybody", "somebody", "anyone",
        "everyone", "no one",     "each",       "either",    "neither",   "many",     "few",
        "several",  "all",        "both",       "any",       "some",      "oneself",  "y'all",
        "youse",    "thee",       "thou",       "thy",       "thine",     "ye",       "yon",
        "yonder",   "whichever",  "whatsoever", "whosoever", "whomsoever"};

    // Common contractions and their variations
    std::vector<std::string> contractions = {
        "i'm",     "i've",     "i'll",     "i'd",      "you're",  "you've",    "you'll",
        "you'd",   "he's",     "he'll",    "he'd",     "she's",   "she'll",    "she'd",
        "it's",    "it'll",    "it'd",     "we're",    "we've",   "we'll",     "we'd",
        "they're", "they've",  "they'll",  "they'd",   "isn't",   "aren't",    "wasn't",
        "weren't", "haven't",  "hasn't",   "hadn't",   "doesn't", "don't",     "didn't",
        "won't",   "wouldn't", "can't",    "couldn't", "mustn't", "shouldn't", "mightn't",
        "shan't",  "let's",    "that's",   "who's",    "what's",  "here's",    "there's",
        "where's", "when's",   "why's",    "how's",    "daren't", "needn't",   "oughtn't",
        "ain't",   "y'all're", "y'all've", "y'all'll", "ma'am",   "o'clock",   "'tis",
        "'twas",   "g'day",    "y'know",   "d'you",    "c'mon",   "dunno",     "gonna",
        "gotta",   "wanna",    "gimme",    "lemme",    "kinda",   "sorta",     "hafta",
        "oughta",  "supposta", "useta",    "coulda",   "woulda",  "shoulda",   "musta"};

    // Common verbs with all their forms
    std::vector<std::string> verbs = {
        // Basic verbs
        "be", "am", "is", "are", "was", "were", "being", "been", "have", "has", "had", "having",
        "do", "does", "did", "doing", "done",
        // Common action verbs
        "go", "goes", "went", "going", "gone", "say", "says", "said", "saying", "get", "gets",
        "got", "getting", "gotten", "make", "makes", "made", "making", "know", "knows", "knew",
        "knowing", "known", "think", "thinks", "thought", "thinking", "take", "takes", "took",
        "taking", "taken", "see", "sees", "saw", "seeing", "seen", "come", "comes", "came",
        "coming", "want", "wants", "wanted", "wanting", "look", "looks", "looked", "looking", "use",
        "uses", "used", "using", "find", "finds", "found", "finding", "give", "gives", "gave",
        "giving", "given", "tell", "tells", "told", "telling", "work", "works", "worked", "working",
        "call", "calls", "called", "calling", "try", "tries", "tried", "trying", "ask", "asks",
        "asked", "asking", "need", "needs", "needed", "needing", "feel", "feels", "felt", "feeling",
        "become", "becomes", "became", "becoming", "leave", "leaves", "left", "leaving", "put",
        "puts", "putting", "mean", "means", "meant", "meaning", "keep", "keeps", "kept", "keeping",
        "let", "lets", "letting", "begin", "begins", "began", "beginning", "begun", "seem", "seems",
        "seemed", "seeming", "help", "helps", "helped", "helping", "talk", "talks", "talked",
        "talking", "turn", "turns", "turned", "turning", "show", "shows", "showed", "showing",
        "shown",
        // Additional verbs
        "write", "writes", "wrote", "writing", "written", "read", "reads", "reading", "sing",
        "sings", "sang", "singing", "sung", "dance", "dances", "danced", "dancing", "play", "plays",
        "played", "playing", "run", "runs", "ran", "running", "jump", "jumps", "jumped", "jumping",
        "swim", "swims", "swam", "swimming", "swum", "eat", "eats", "ate", "eating", "eaten",
        "drink", "drinks", "drank", "drinking", "drunk", "sleep", "sleeps", "slept", "sleeping",
        "walk", "walks", "walked", "walking", "fly", "flies", "flew", "flying", "flown", "draw",
        "draws", "drew", "drawing", "drawn",
        // Adding frequently occurring verbs from logs
        "prepare", "wait", "compete", "meet", "collaborate", "repair", "cook", "rush", "entertain",
        "hop", "code", "respond", "train", "examine", "soar", "maintain", "hunt", "patrol",
        "meditate", "consult", "study", "practice", "deploy", "serve", "rehearse", "build",
        "analyze", "learn", "drive", "create", "gather", "sit", "teach", "worship", "visit", "test",
        "clean", "operate", "mix", "treat", "research", "counsel", "fight", "glide", "preside",
        "rest", "settle", "pray", "organize", "file", "type", "experiment", "observe", "perform",
        "collect", "plan"};

    // Common prepositions and conjunctions
    std::vector<std::string> connectors = {
        // Prepositions
        "in", "on", "at", "to", "for", "with", "by", "from", "up", "about", "into", "over", "after",
        "beneath", "under", "above", "below", "behind", "between", "beyond", "during", "except",
        "through", "toward", "within", "without", "across", "along", "around", "before", "beside",
        "besides", "down", "inside", "near", "off", "since", "upon", "within", "throughout",
        // Additional prepositions
        "amid", "amidst", "among", "amongst", "atop", "barring", "concerning", "considering",
        "despite", "excluding", "following", "including", "minus", "notwithstanding", "opposite",
        "outside", "past", "per", "plus", "regarding", "round", "save", "unlike", "versus", "via",
        "worth",
        // Conjunctions
        "and", "but", "or", "nor", "for", "yet", "so", "because", "although", "unless", "since",
        "while", "where", "if", "then", "else", "therefore", "however", "moreover", "furthermore",
        "nevertheless", "meanwhile", "afterwards", "consequently", "otherwise", "instead",
        "whereas",
        // Additional conjunctions
        "accordingly", "additionally", "albeit", "besides", "hence", "likewise", "namely",
        "notwithstanding", "provided", "similarly", "thus", "wherefore", "wherever", "whenever",
        "whence", "whereby", "wherein", "whereupon"};

    // Common adjectives
    std::vector<std::string> adjectives = {
        "good", "new", "first", "last", "long", "great", "little", "own", "other", "old", "right",
        "big", "high", "different", "small", "large", "next", "early", "young", "important", "few",
        "public", "bad", "same", "able", "best", "better", "low", "late", "general", "specific",
        "certain", "free", "full", "special", "easy", "clear", "recent", "final", "main", "sure",
        "real", "available", "local", "particular", "hard", "major", "current", "nice", "happy",
        "serious", "ready", "simple", "possible", "whole", "short", "private", "past", "beautiful",
        "strong", "quick",
        // Additional adjectives
        "amazing", "awesome", "brilliant", "calm", "clever", "colorful", "creative", "curious",
        "delicate", "eager", "elegant", "energetic", "enormous", "excellent", "excited", "famous",
        "fantastic", "fierce", "friendly", "gentle", "gorgeous", "graceful", "handsome", "healthy",
        "helpful", "honest", "humble", "hungry", "innocent", "intelligent", "kind", "lively",
        "lovely", "lucky", "magical", "mysterious", "natural", "patient", "peaceful", "perfect",
        "pleasant", "polite", "powerful", "proud", "quiet", "rare", "reliable", "rich", "scared",
        "shy", "silly", "smart", "smooth", "soft", "sweet", "talented", "tiny", "tough", "unique",
        "warm", "wise", "wonderful", "worried", "young"};

    // Common nouns
    std::vector<std::string> nouns_vec = {
        // People and roles
        "person", "people", "family", "friend", "parent", "mother", "father", "child", "baby",
        "teacher", "student", "doctor", "worker", "artist",
        // Additional people and roles
        "accountant", "actor", "architect", "athlete", "author", "baker", "banker", "barber",
        "carpenter", "chef", "clerk", "coach", "dancer", "dentist", "designer", "director",
        "driver", "engineer", "farmer", "firefighter", "judge", "lawyer", "mechanic", "musician",
        "nurse", "painter", "pilot", "plumber", "poet", "police", "professor", "programmer",
        "reporter", "sailor", "scientist", "secretary", "singer", "soldier", "surgeon", "tailor",
        "therapist", "trainer", "translator", "veterinarian", "waiter", "writer",

        // Places
        "home", "house", "school", "office", "store", "hospital", "city", "country", "world",
        "room", "building", "street", "park", "garden",
        // Additional places
        "airport", "apartment", "arena", "bank", "beach", "bridge", "cafe", "castle", "cathedral",
        "church", "cinema", "clinic", "college", "court", "factory", "farm", "gallery", "gym",
        "harbor", "hotel", "island", "laboratory", "library", "mall", "market", "museum", "palace",
        "prison", "restaurant", "shop", "stadium", "station", "studio", "theater", "tower",
        "university", "village", "warehouse", "zoo",

        // Time
        "time", "day", "night", "morning", "evening", "week", "month", "year", "today", "tomorrow",
        "minute", "hour", "moment", "future", "past",
        // Additional time-related
        "afternoon", "age", "century", "dawn", "decade", "dusk", "era", "eternity", "history",
        "lifetime", "midnight", "millennium", "noon", "period", "present", "season", "spring",
        "summer", "autumn", "winter", "twilight", "weekend", "yesterday",

        // Nature
        "water", "air", "earth", "fire", "sun", "moon", "star", "sky", "tree", "flower", "grass",
        "river", "ocean", "mountain", "forest",
        // Additional nature
        "aurora", "avalanche", "beach", "breeze", "brook", "canyon", "cave", "cliff", "cloud",
        "coast", "coral", "crater", "desert", "dew", "dust", "earthquake", "eclipse", "fog",
        "frost", "galaxy", "geyser", "glacier", "hill", "hurricane", "iceberg", "island", "lake",
        "landscape", "meteor", "mist", "oasis", "planet", "rain", "rainbow", "reef", "sand", "sea",
        "snow", "storm", "stream", "sunrise", "sunset", "thunder", "tornado", "valley", "volcano",
        "wave", "wind",

        // Objects
        "book", "phone", "computer", "car", "door", "window", "table", "chair", "bed", "food",
        "money", "paper", "key", "screen", "picture",
        // Additional objects
        "alarm", "album", "anchor", "arrow", "badge", "bag", "ball", "basket", "battery", "bell",
        "blanket", "bottle", "bowl", "box", "bracelet", "brush", "bucket", "button", "camera",
        "candle", "card", "carpet", "clock", "coin", "compass", "crown", "cup", "curtain", "desk",
        "diary", "dictionary", "dish", "doll", "envelope", "fan", "flag", "flask", "fork", "frame",
        "glass", "glove", "hammer", "hat", "helmet", "knife", "lamp", "lock", "magazine", "map",
        "medal", "mirror", "needle", "newspaper", "notebook", "package", "paint", "pen", "pencil",
        "pillow", "plate", "radio", "ribbon", "ring", "rope", "ruler", "scissors", "shelf", "shoe",
        "soap", "spoon", "stamp", "stapler", "sword", "telescope", "ticket", "tool", "torch", "toy",
        "umbrella", "vase", "wallet", "watch", "wheel", "wire",

        // Abstract concepts
        "life", "death", "love", "hate", "peace", "war", "truth", "lie", "idea", "thought", "dream",
        "hope", "fear", "mind", "soul",
        // Additional abstract concepts
        "ability", "achievement", "action", "adventure", "advice", "age", "anger", "anxiety", "art",
        "balance", "beauty", "belief", "blame", "chance", "change", "chaos", "choice", "comfort",
        "communication", "confidence", "conflict", "confusion", "connection", "consciousness",
        "control", "courage", "creativity", "crisis", "culture", "curiosity", "democracy",
        "destiny", "difference", "difficulty", "dignity", "discipline", "discovery", "diversity",
        "doubt", "duty", "education", "emotion", "energy", "equality", "evil", "excellence",
        "existence", "experience", "failure", "faith", "fame", "fate", "freedom", "friendship",
        "fun", "future", "glory", "goal", "goodness", "grace", "gratitude", "grief", "growth",
        "guilt", "happiness", "harmony", "health", "heaven", "hell", "history", "honor", "humanity",
        "humor", "identity", "imagination", "independence", "infinity", "influence", "information",
        "innocence", "inspiration", "intelligence", "interest", "intuition", "irony", "joy",
        "justice", "kindness", "knowledge", "language", "laughter", "law", "liberty", "logic",
        "loneliness", "loss", "luck", "luxury", "magic", "meaning", "memory", "mercy", "miracle",
        "mystery", "nature", "necessity", "need", "opportunity", "pain", "passion", "patience",
        "perception", "perfection", "philosophy", "pleasure", "politics", "possibility", "poverty",
        "power", "pride", "principle", "progress", "promise", "prosperity", "purpose", "quality",
        "quantity", "question", "reality", "reason", "recognition", "religion", "respect",
        "responsibility", "revenge", "risk", "romance", "sacrifice", "safety", "satisfaction",
        "science", "security", "self", "sense", "serenity", "shame", "silence", "simplicity", "sin",
        "skill", "society", "solitude", "sorrow", "spirit", "strength", "stress", "structure",
        "success", "suffering", "surprise", "talent", "taste", "technology", "theory", "thinking",
        "time", "tolerance", "tradition", "trust", "understanding", "unity", "universe", "value",
        "victory", "violence", "virtue", "vision", "wealth", "wisdom", "wonder", "work", "world",
        "worth", "youth",

        // Adding frequently occurring nouns from logs
        "service", "simulator", "tunnel", "briefing", "hangar", "club", "hall", "space", "field",
        "players", "headquarters", "chamber", "workspace", "facility", "meat", "comedy", "bakery",
        "center", "tarmac", "auditorium", "bay", "ward", "pond", "wall", "ice", "lab", "sanctuary",
        "temple", "mosque", "observatory", "academy", "range", "shrine", "wine", "workshop",
        "chapel", "classroom", "base", "records", "pharmacy", "department", "pool", "galley",
        "rink", "track", "kitchen", "cellar", "precinct", "bar", "courtroom", "conference",
        "meeting", "district", "mat", "cargo", "anteroom", "monastery", "stage", "cockpit",

        // Teams and groups
        "crew", "teams", "assistants", "handlers", "technicians",

        // Facilities and rooms
        "examination", "consultation", "therapy", "radiology", "filing",

        // Adding missing professional and role-related nouns
        "aircraft", "baggage", "instructor", "instructors", "aviation", "mechanic", "mechanics",
        "controller", "controllers", "attendant", "attendants", "analyst", "analysts", "researcher",
        "researchers", "geologist", "geologists", "developer", "developers", "rescuer", "rescuers",
        "medic", "medics", "chemist", "chemists", "designer", "designers", "therapist",
        "therapists", "barista", "baristas", "paramedic", "paramedics", "climber", "climbers",
        "runner", "runners", "sous", "teacher", "teachers", "biologist", "biologists", "skater",
        "skaters", "pilgrim", "pilgrims", "attorney", "attorneys", "scientist", "scientists",
        "ranger", "rangers", "prosecutor", "prosecutors", "professor", "professors", "bartender",
        "bartenders", "worshipper", "worshippers", "priest", "priests", "pilot", "pilots",
        "scholar", "scholars", "pupil", "pupils", "firefighter", "firefighters", "inventor",
        "inventors", "officer", "officers", "musician", "musicians", "astronomer", "astronomers",
        "programmer", "programmers", "dancer", "dancers", "actor", "actors", "doctor", "doctors",
        "spectator", "spectators", "bowler", "bowlers", "clerk", "clerks", "tutor", "tutors",
        "baker", "bakers", "gamer", "gamers", "artist", "artists", "monk", "monks", "performer",
        "performers", "dj", "djs", "expert", "experts", "comedian", "comedians", "surgeon",
        "surgeons", "judge", "judges", "radiologist", "radiologists", "patient", "patients",
        "paralegal", "paralegals", "nurse", "nurses", "specialist", "specialists", "pharmacist",
        "pharmacists", "technician", "technicians", "boxer", "boxers", "police", "dentist",
        "dentists", "navigator", "navigators", "psychiatrist", "psychiatrists", "athlete",
        "athletes", "swimmer", "swimmers", "player", "players", "golfer", "golfers", "chef",
        "chefs", "sommelier", "sommeliers", "butcher", "butchers", "waiter", "waiters", "guard",
        "guards", "believer", "believers", "mediator", "mediators", "lawyer", "lawyers", "witness",
        "witnesses", "reporter", "reporters", "physicist", "physicists",

        // Adding missing animal-related nouns
        "duck", "ducks", "bird", "birds", "wolf", "wolves", "bear", "fish", "dog", "rabbit",
        "rabbits", "eagle", "eagles", "lion", "cat",

        // Adding missing place and facility-related nouns
        "seminary", "concert", "tournament", "alley", "patisserie",

        // Adding missing abstract/activity nouns
        "traffic", "shade", "lives", "prep", "testing", "driving", "bowling", "coffee",

        // Adding missing organizational/group nouns
        "congregation", "congregations", "audience", "audiences",

        // Adding missing specialized terms
        "air", "flight", "data", "defense", "ground", "legal", "operating", "dressing", "chambers"};

    // Store nouns in the set for quick lookup
    for (const auto& noun : nouns_vec) {
        nouns.insert(noun);
    }

    // Add all words to vocabulary
    std::vector<std::string> all_words;
    all_words.insert(all_words.end(), articles.begin(), articles.end());
    all_words.insert(all_words.end(), pronouns.begin(), pronouns.end());
    all_words.insert(all_words.end(), contractions.begin(), contractions.end());
    all_words.insert(all_words.end(), verbs.begin(), verbs.end());
    all_words.insert(all_words.end(), connectors.begin(), connectors.end());
    all_words.insert(all_words.end(), adjectives.begin(), adjectives.end());
    all_words.insert(all_words.end(), nouns_vec.begin(), nouns_vec.end());

    // Sort and remove duplicates
    std::sort(all_words.begin(), all_words.end());
    all_words.erase(std::unique(all_words.begin(), all_words.end()), all_words.end());

    // Add each word
    for (const auto& word : all_words) {
        add_word(word);
    }
}

int Vocabulary::get_id(const std::string& token) const {
    auto it = token_to_id.find(token);
    return it != token_to_id.end() ? it->second : unk_token_id;
}

std::string Vocabulary::get_token(int id) const {
    auto it = id_to_token.find(id);
    if (it == id_to_token.end()) {
        // Return the unknown token string, not its ID
        auto unk_it = id_to_token.find(unk_token_id);
        if (unk_it == id_to_token.end()) {
            throw std::runtime_error("Unknown token not found in vocabulary");
        }
        return unk_it->second;
    }
    return it->second;
}

size_t Vocabulary::size() const {
    return id_to_token.size();
}

void Vocabulary::print_vocabulary_mappings() const {
    std::cout << "\n=== Special Tokens ===\n";
    
    // Use .at() for const access to map
    std::cout << "PAD token: " << pad_token_id << " <-> " << id_to_token.at(pad_token_id) << "\n";
    std::cout << "UNK token: " << unk_token_id << " <-> " << id_to_token.at(unk_token_id) << "\n";
    std::cout << "BOS token: " << bos_token_id << " <-> " << id_to_token.at(bos_token_id) << "\n";
    std::cout << "EOS token: " << eos_token_id << " <-> " << id_to_token.at(eos_token_id) << "\n";
    std::cout << "MASK token: " << mask_token_id << " <-> " << id_to_token.at(mask_token_id) << "\n";

    std::cout << "\n=== Full Vocabulary ===\n";
    std::cout << "Total size: " << size() << " tokens\n\n";
    
    // Print first few tokens as sample
    size_t sample_size = std::min(size_t(10), size());
    for (size_t i = 0; i < sample_size; i++) {
        std::cout << "ID " << i << " -> '" << id_to_token.at(i) << "'\n";
    }
    if (size() > sample_size) {
        std::cout << "... and " << (size() - sample_size) << " more tokens\n";
    }
}

bool Vocabulary::verify_mappings() const {
    try {
        // Validate special tokens directly
        const std::vector<std::pair<std::string, int>> special_tokens = {
            {tokens::PAD_TOKEN, tokens::PAD_ID},
            {tokens::UNK_TOKEN, tokens::UNK_ID},
            {tokens::BOS_TOKEN, tokens::BOS_ID},
            {tokens::EOS_TOKEN, tokens::EOS_ID},
            {tokens::MASK_TOKEN, tokens::MASK_ID}
        };

        // Check special token mappings
        for (const auto& [token, id] : special_tokens) {
            auto token_id_it = token_to_id.find(token);
            auto id_token_it = id_to_token.find(id);
            
            if (token_id_it == token_to_id.end() || id_token_it == id_to_token.end()) {
                throw std::runtime_error("Missing special token mapping for: " + token);
            }
            if (token_id_it->second != id || id_token_it->second != token) {
                throw std::runtime_error("Inconsistent special token mapping for: " + token);
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Vocabulary validation failed: " << e.what() << std::endl;
        return false;
    }
}

bool Vocabulary::is_noun(const std::string& token) const {
    return nouns.find(token) != nouns.end();
}

void Vocabulary::load_nouns(const std::string& noun_file_path) {
    std::ifstream file(noun_file_path);
    std::string line;
    while (std::getline(file, line)) {
        nouns.insert(line);
    }
}
#include "../include/sentencepiece_tokenizer.hpp"
#include <stdexcept>
#include <fstream>
#include <sentencepiece_trainer.h>
#include <filesystem>
#include <iostream>
#include <sstream>  // for std::ostringstream

SentencePieceTokenizer::SentencePieceTokenizer()
    : processor_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {}

void SentencePieceTokenizer::load_model(const std::string& model_path) {
    const auto status = processor_->Load(model_path);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load SentencePiece model: " + 
            status.ToString());
    }
}

void SentencePieceTokenizer::train(
    const std::vector<std::string>& texts,
    const std::string& model_prefix,
    size_t vocab_size) {
    
    // Convert model_prefix to absolute path
    std::filesystem::path abs_model_path = std::filesystem::absolute(model_prefix);
    
    // Create model directory if it doesn't exist
    std::filesystem::path model_dir = abs_model_path.parent_path();
    if (!model_dir.empty()) {
        std::filesystem::create_directories(model_dir);
    }
    
    // Use a temporary file in the same directory as the model
    std::filesystem::path temp_file = model_dir / "temp_training_data.txt";
    std::ofstream ofs(temp_file);
    if (!ofs) {
        throw std::runtime_error("Failed to create training data file: " + temp_file.string());
    }
    
    // Add debug output
    std::cout << "Writing " << texts.size() << " texts to " << temp_file << std::endl;
    size_t total_chars = 0;
    size_t non_empty_texts = 0;
    
    for (const auto& text : texts) {
        if (!text.empty()) {
            ofs << text << "\n";
            total_chars += text.length();
            non_empty_texts++;
        }
    }
    ofs.close();

    std::cout << "Wrote " << non_empty_texts << " non-empty texts with total " 
              << total_chars << " characters" << std::endl;

    // Verify the training file was written correctly
    std::ifstream check_file(temp_file);
    if (!check_file || check_file.peek() == std::ifstream::traits_type::eof()) {
        throw std::runtime_error("Training file is empty or could not be read: " + temp_file.string());
    }

    // First run a test training to determine the maximum possible vocabulary size
    std::string probe_args = 
        "--input=" + temp_file.string() + " "
        "--model_prefix=" + abs_model_path.string() + "_probe "  // Use different prefix for probe
        "--character_coverage=0.9995 "
        "--model_type=unigram "
        "--normalization_rule_name=nmt_nfkc "
        "--input_format=text "
        "--split_by_whitespace=true "
        "--add_dummy_prefix=true "
        "--max_sentence_length=2048 "
        "--num_threads=4 "
        "--train_extremely_large_corpus=false "
        "--shrinking_factor=0.95 "
        "--num_sub_iterations=2 "
        "--max_sentencepiece_length=16 "
        "--vocab_size=8000";  // Start with large vocab to see what's possible

    // Run probe training to get actual possible vocab size
    auto probe_status = sentencepiece::SentencePieceTrainer::Train(probe_args);
    if (!probe_status.ok()) {
        std::string error = probe_status.ToString();
        size_t pos = error.find("Please set it to a value <= ");
        if (pos != std::string::npos) {
            pos += 27;  // Length of "Please set it to a value <= "
            size_t end_pos = error.find(".", pos);
            if (end_pos != std::string::npos) {
                // Extract the maximum allowed vocabulary size
                std::string max_size_str = error.substr(pos, end_pos - pos);
                size_t max_vocab_size = std::stoul(max_size_str);
                
                // Use 95% of the maximum to be safe
                size_t actual_vocab_size = static_cast<size_t>(max_vocab_size * 0.95);
                
                std::cout << "Determined maximum vocabulary size: " << actual_vocab_size 
                          << " (from probe: " << max_vocab_size << ")" << std::endl;

                // Now run the actual training with the correct vocabulary size
                std::string training_args = 
                    "--input=" + temp_file.string() + " "
                    "--model_prefix=" + abs_model_path.string() + " "
                    "--vocab_size=" + std::to_string(actual_vocab_size) + " "
                    "--character_coverage=0.9995 "
                    "--model_type=unigram "
                    "--normalization_rule_name=nmt_nfkc "
                    "--pad_id=" + std::to_string(tokens::PAD_ID) + " "
                    "--unk_id=" + std::to_string(tokens::UNK_ID) + " "
                    "--bos_id=" + std::to_string(tokens::BOS_ID) + " "
                    "--eos_id=" + std::to_string(tokens::EOS_ID) + " "
                    "--pad_piece=" + tokens::PAD_TOKEN + " "
                    "--unk_piece=" + tokens::UNK_TOKEN + " "
                    "--bos_piece=" + tokens::BOS_TOKEN + " "
                    "--eos_piece=" + tokens::EOS_TOKEN + " "
                    "--input_format=text "
                    "--split_by_whitespace=true "
                    "--add_dummy_prefix=true "
                    "--max_sentence_length=2048 "
                    "--num_threads=4 "
                    "--train_extremely_large_corpus=false "
                    "--shrinking_factor=0.95 "        // Slower shrinking to control vocab growth
                    "--num_sub_iterations=2 "         // More iterations for better convergence
                    "--max_sentencepiece_length=16 "  // Limit piece length
                    "--seed_sentencepiece_size=3000"; // Start with smaller seed pieces

                // Clean up probe files
                std::filesystem::remove(abs_model_path.string() + "_probe.model");
                std::filesystem::remove(abs_model_path.string() + "_probe.vocab");

                // Instead of returning directly, store the status and continue
                auto status = sentencepiece::SentencePieceTrainer::Train(training_args);
                if (status.ok()) {
                    // Clean up temp file
                    std::filesystem::remove(temp_file);
                    
                    // Load the trained model
                    try {
                        load_model(abs_model_path.string() + ".model");
                        return;  // Success - early return
                    } catch (const std::exception& e) {
                        throw std::runtime_error("Failed to load trained model: " + std::string(e.what()));
                    }
                }
            }
        }
    }

    // If probe didn't give us the info we need, use a very conservative estimate
    size_t actual_vocab_size = std::min(vocab_size, static_cast<size_t>(1500));  // Very conservative

    // Build training arguments with better control over sentence handling
    std::string training_args = 
        "--input=" + temp_file.string() + " "
        "--model_prefix=" + abs_model_path.string() + " "
        "--vocab_size=" + std::to_string(actual_vocab_size) + " "
        "--character_coverage=0.9995 "
        "--model_type=unigram "
        "--normalization_rule_name=nmt_nfkc "
        "--pad_id=" + std::to_string(tokens::PAD_ID) + " "
        "--unk_id=" + std::to_string(tokens::UNK_ID) + " "
        "--bos_id=" + std::to_string(tokens::BOS_ID) + " "
        "--eos_id=" + std::to_string(tokens::EOS_ID) + " "
        "--pad_piece=" + tokens::PAD_TOKEN + " "
        "--unk_piece=" + tokens::UNK_TOKEN + " "
        "--bos_piece=" + tokens::BOS_TOKEN + " "
        "--eos_piece=" + tokens::EOS_TOKEN + " "
        "--input_format=text "
        "--split_by_whitespace=true "
        "--add_dummy_prefix=true "
        "--max_sentence_length=2048 "
        "--num_threads=4 "
        "--train_extremely_large_corpus=false "
        "--shrinking_factor=0.95 "        // Slower shrinking to control vocab growth
        "--num_sub_iterations=2 "         // More iterations for better convergence
        "--max_sentencepiece_length=16 "  // Limit piece length
        "--seed_sentencepiece_size=3000"; // Start with smaller seed pieces

    std::cout << "Using vocabulary size: " << actual_vocab_size 
              << " (requested: " << vocab_size << ")" << std::endl;

    // Print args for debugging
    std::cout << "Training arguments:\n" << training_args << std::endl;

    // Train using the correct overload
    std::string error_message;
    const auto status = sentencepiece::SentencePieceTrainer::Train(training_args);
    
    if (!status.ok()) {
        std::cerr << "Training failed with error: " << status.ToString() << std::endl;
        std::cerr << "Training file location: " << std::filesystem::absolute(temp_file) << std::endl;
        
        // Check if training file exists and is readable
        if (!std::filesystem::exists(temp_file)) {
            std::cerr << "Training file does not exist!" << std::endl;
        } else {
            std::ifstream check(temp_file);
            if (!check) {
                std::cerr << "Training file exists but cannot be read!" << std::endl;
            }
        }
        
        // Clean up temp file
        std::filesystem::remove(temp_file);
        throw std::runtime_error("SentencePiece training failed: " + status.ToString());
    }

    // Clean up temp file
    std::filesystem::remove(temp_file);

    // Verify the model file was created
    std::string model_path = abs_model_path.string() + ".model";
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file was not created: " + model_path);
    }

    // Load the trained model
    try {
        load_model(model_path);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load trained model: " + std::string(e.what()));
    }
}

std::vector<int> SentencePieceTokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    const auto status = processor_->Encode(text, &ids);
    if (!status.ok()) {
        throw std::runtime_error("Encoding failed: " + status.ToString());
    }
    return ids;
}

std::string SentencePieceTokenizer::decode(const std::vector<int>& ids) const {
    std::string text;
    const auto status = processor_->Decode(ids, &text);
    if (!status.ok()) {
        throw std::runtime_error("Decoding failed: " + status.ToString());
    }
    return text;
} 
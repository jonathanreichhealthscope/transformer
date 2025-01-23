#pragma once
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>
#include "token_constants.hpp"

class SentencePieceTokenizer {
public:
    SentencePieceTokenizer();
    ~SentencePieceTokenizer() = default;

    // Initialize with a pre-trained model
    void load_model(const std::string& model_path);

    // Train a new model
    void train(const std::vector<std::string>& texts,
               const std::string& model_prefix,
               size_t vocab_size = 32000);

    // Core tokenization methods
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    
    // Special token handling
    int get_pad_token_id() const { return processor_->pad_id(); }
    int get_unk_token_id() const { return processor_->unk_id(); }
    int get_bos_token_id() const { return processor_->bos_id(); }
    int get_eos_token_id() const { return processor_->eos_id(); }
    
    // Vocabulary access
    size_t vocab_size() const { return processor_->GetPieceSize(); }
    std::string id_to_token(int id) const { return processor_->IdToPiece(id); }
    int token_to_id(const std::string& token) const { return processor_->PieceToId(token); }

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
}; 
#pragma once

#include <functional>
#include <queue>
#include <utility>
#include <vector>
#include "matrix.hpp"

/**
 * @brief Implements beam search decoding for sequence generation.
 * 
 * Beam search is a heuristic search algorithm that explores a graph by expanding
 * the most promising nodes in a limited set. In the context of text generation,
 * it maintains multiple hypotheses (beams) at each step and selects the most
 * likely sequences. Features include:
 * - Configurable beam width
 * - Length penalty for balanced sequence lengths
 * - Early stopping on EOS token
 * - Score normalization
 */
class BeamSearch {
  public:
    /**
     * @brief Constructs a beam search decoder.
     * @param beam_width Number of beams to maintain (higher means more diverse but slower)
     * @param length_penalty Penalty factor for sequence length (>1.0 favors longer sequences)
     * @param temperature Temperature for temperature scaling
     * @param diversity_strength Strength of diversity penalty
     * @param top_k Top-K sampling parameter
     * @param top_p Top-P sampling parameter
     */
    BeamSearch(size_t beam_width = 5, 
               float length_penalty = 0.6f,
               float temperature = 1.0f,
               float diversity_strength = 0.5f,
               size_t top_k = 40,
               float top_p = 0.9f);

    /**
     * @brief Represents a single hypothesis in beam search.
     * 
     * A hypothesis contains a sequence of tokens and its associated score.
     * Hypotheses can be compared based on their scores for beam selection.
     */
    struct Hypothesis {
        float score;                  ///< Log probability score of the hypothesis
        std::vector<int> tokens;     ///< Sequence of token IDs
        size_t last_token;           ///< Last token ID in the sequence
        size_t length;               ///< Length of the sequence
        bool finished;               ///< Whether the hypothesis has finished generating

        // Constructor for initializing with a sequence and score
        Hypothesis(const std::vector<int>& sequence, float score)
            : tokens(sequence), 
              score(score),
              last_token(sequence.empty() ? 0 : sequence.back()),
              length(sequence.size()),
              finished(false) {}

        // Constructor for initializing with individual components
        Hypothesis(float score = 0.0f, size_t last_token = 0,
                  size_t length = 0, bool finished = false)
            : score(score),
              last_token(last_token),
              length(length),
              finished(finished) {}

        /**
         * @brief Compares hypotheses based on their scores.
         * @param other Hypothesis to compare against
         * @return true if this hypothesis has lower score
         */
        bool operator<(const Hypothesis& other) const {
            return score < other.score;
        }
    };

    /**
     * @brief Represents a beam candidate during search
     */
    struct BeamCandidate {
        std::vector<size_t> sequence;
        float score;
        
        BeamCandidate(const std::vector<size_t>& seq, float s) 
            : sequence(seq), score(s) {}
    };

    /**
     * @brief Performs beam search decoding.
     * 
     * Starting from initial logits, generates sequences by maintaining the top-k
     * most likely hypotheses at each step. The search continues until either
     * max_length is reached or all beams produce EOS token.
     * 
     * @param initial_logits Initial token probabilities (log space)
     * @param next_token_fn Function to get next token probabilities given a sequence
     * @param max_length Maximum sequence length to generate
     * @param eos_token_id Token ID that marks end of sequence
     * @return Vector of final hypotheses, sorted by score
     */
    std::vector<Hypothesis>
    search(const std::vector<float>& initial_logits,
           std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
           size_t max_length, int eos_token_id);

  private:
    size_t beam_width_;     ///< Number of beams to maintain
    float length_penalty_;  ///< Penalty factor for sequence length
    float temperature;
    float diversity_strength;
    size_t top_k;
    float top_p;

    // Add special token IDs
    int pad_token_id_;   ///< ID for padding token
    int unk_token_id_;   ///< ID for unknown token
    int bos_token_id_;   ///< ID for beginning of sequence token
    int eos_token_id_;   ///< ID for end of sequence token
    int mask_token_id_;  ///< ID for mask token

    /**
     * @brief Applies length penalty to a hypothesis score.
     * 
     * The length penalty helps balance between short and long sequences:
     * score / (length ^ length_penalty)
     * 
     * @param score Raw hypothesis score
     * @param length Sequence length
     * @return Penalized score
     */
    float apply_length_penalty(float score, size_t length) const;

    // Helper methods for beam search
    void update_beams(std::vector<std::vector<int>>& sequences,
                     Matrix& beam_scores,
                     const Matrix& next_scores,
                     const std::vector<int>& next_tokens);

    bool is_search_complete(const std::vector<std::vector<int>>& sequences);

    std::vector<int> get_best_sequence(
        const std::vector<std::vector<int>>& sequences,
        const Matrix& beam_scores);

    /**
     * @brief CPU implementation of beam search
     * @param initial_logits Initial token probabilities
     * @param next_token_fn Function to get next token probabilities
     * @param max_length Maximum sequence length
     * @return Best sequence found
     */
    std::vector<int> cpu_beam_search(
        const std::vector<float>& initial_logits,
        std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
        size_t max_length);

    /**
     * @brief Calculates scores for the next tokens using temperature scaling
     */
    std::vector<float> calculateScores(const std::vector<float>& logits);
    
    /**
     * @brief Applies diversity penalty to beam candidates
     */
    void diversityPenalty(std::vector<BeamCandidate>& candidates, float strength);
    
    /**
     * @brief Calculates sequence overlap for diversity penalty
     */
    float calculateOverlap(const std::vector<size_t>& seq1, const std::vector<size_t>& seq2) {
        size_t overlap = 0;
        for (size_t token : seq1) {
            if (std::find(seq2.begin(), seq2.end(), token) != seq2.end()) {
                overlap++;
            }
        }
        return static_cast<float>(overlap) / std::max(seq1.size(), seq2.size());
    }

    /**
     * @brief Performs top-k sampling on probability distribution
     */
    std::vector<std::pair<float, size_t>> topKSampling(
        const std::vector<float>& probabilities, size_t k);

    /**
     * @brief Performs nucleus (top-p) sampling on probability distribution
     */
    std::vector<std::pair<float, size_t>> nucleusSampling(
        const std::vector<float>& probabilities, float p);
};
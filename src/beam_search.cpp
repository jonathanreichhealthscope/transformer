#include "../include/beam_search.hpp"
#include "../include/cuda/matrix_ops.cuh"
#include "../include/cuda/memory_manager.cuh"
#include "../include/utils.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

BeamSearch::BeamSearch(size_t beam_width, float length_penalty, float temperature,
                       float diversity_strength, size_t top_k, float top_p)
    : beam_width_(beam_width)
    , length_penalty_(length_penalty)
    , temperature(temperature)
    , diversity_strength(diversity_strength)
    , top_k(top_k)
    , top_p(top_p) {
    std::cout << "Initializing BeamSearch with width=" << beam_width
              << ", length_penalty=" << length_penalty 
              << ", temperature=" << temperature
              << ", diversity_strength=" << diversity_strength
              << ", top_k=" << top_k
              << ", top_p=" << top_p << std::endl;
    // Use consistent special token IDs
    pad_token_id_ = 0;
    unk_token_id_ = 1;
    bos_token_id_ = 2;
    eos_token_id_ = 3;
    mask_token_id_ = 4;
}

float BeamSearch::apply_length_penalty(float score, size_t length) const {
    float penalized_score = score / std::pow(length, length_penalty_);
    return penalized_score;
}

void BeamSearch::update_beams(std::vector<std::vector<int>>& sequences,
                              Matrix& beam_scores,
                              const Matrix& next_scores,
                              const std::vector<int>& next_tokens) {
    const float MIN_SCORE = -1e4f;  // Prevent -inf
    
    // Create temporary vectors to store candidates
    std::vector<std::pair<float, std::pair<size_t, int>>> candidates;
    candidates.reserve(beam_width_ * beam_width_);
    
    // Track token frequencies across all beams
    std::unordered_map<int, int> token_counts;
    for (const auto& seq : sequences) {
        for (int token : seq) {
            token_counts[token]++;
        }
    }
    
    // Gather all candidates from all beams
    for (size_t i = 0; i < beam_width_; i++) {
        float current_score = std::max(beam_scores(i, 0), MIN_SCORE);
        for (size_t j = 0; j < beam_width_; j++) {
            float next_score = std::max(next_scores(i, j), MIN_SCORE);
            float score = current_score + next_score;
            
            // Apply diversity penalty based on token frequency
            int next_token = next_tokens[j];
            float diversity_penalty = 0.0f;
            
            // Penalize tokens that appear frequently
            if (token_counts[next_token] > 0) {
                diversity_penalty = diversity_strength * std::log1p(token_counts[next_token]);
            }
            
            // Extra penalty for repeating the first token of any beam
            for (const auto& seq : sequences) {
                if (!seq.empty() && seq[0] == next_token) {
                    diversity_penalty += diversity_strength * 2.0f;
                }
            }
            
            score -= diversity_penalty;
            
            if (score > MIN_SCORE) {  // Only add valid candidates
                candidates.push_back({score, {i, next_token}});
            }
        }
    }
    
    // If we have no valid candidates, create a fallback candidate
    if (candidates.empty()) {
        candidates.push_back({MIN_SCORE, {0, eos_token_id_}});
    }
    
    // Sort candidates by score
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Create new sequences and scores with diversity enforcement
    std::vector<std::vector<int>> new_sequences;
    Matrix new_scores(beam_width_, 1);
    std::unordered_set<int> used_first_tokens;  // Track first tokens to ensure diversity
    
    size_t added = 0;
    size_t candidate_idx = 0;
    
    // Take top beam_width_ candidates while enforcing diversity
    while (added < beam_width_ && candidate_idx < candidates.size()) {
        const auto& [score, beam_token] = candidates[candidate_idx];
        const auto& [beam_idx, token] = beam_token;
        
        // Check if this would create a sequence with a repeated first token
        std::vector<int> new_sequence = sequences[beam_idx];
        new_sequence.push_back(token);
        
        int first_token = new_sequence[0];
        if (used_first_tokens.find(first_token) == used_first_tokens.end()) {
            used_first_tokens.insert(first_token);
            new_sequences.push_back(std::move(new_sequence));
            new_scores(added, 0) = score;
            added++;
        }
        candidate_idx++;
    }
    
    // Fill remaining beams if needed
    while (new_sequences.size() < beam_width_) {
        // Find the next valid candidate
        while (candidate_idx < candidates.size()) {
            const auto& [score, beam_token] = candidates[candidate_idx];
            const auto& [beam_idx, token] = beam_token;
            
            std::vector<int> new_sequence = sequences[beam_idx];
            new_sequence.push_back(token);
            
            new_sequences.push_back(std::move(new_sequence));
            new_scores(new_sequences.size() - 1, 0) = score;
            
            if (new_sequences.size() >= beam_width_) break;
            candidate_idx++;
        }
        
        // If we still need more sequences, duplicate the first one
        if (new_sequences.size() < beam_width_) {
            new_sequences.push_back(new_sequences[0]);
            new_scores(new_sequences.size() - 1, 0) = MIN_SCORE;
        }
    }
    
    // Update sequences and scores
    sequences = std::move(new_sequences);
    beam_scores = std::move(new_scores);
}

bool BeamSearch::is_search_complete(const std::vector<std::vector<int>>& sequences) {
    // Check if all sequences have reached the end token or max length
    const size_t MAX_LENGTH = 1024;  // Explicit constant
    for (const auto& seq : sequences) {
        if (seq.empty()) return false;
        
        // Continue searching if any sequence hasn't ended and hasn't reached max length
        if (seq.back() != eos_token_id_ && seq.size() < MAX_LENGTH) {
            return false;
        }
    }
    return true;  // All sequences have ended or reached max length
}

std::vector<int> BeamSearch::get_best_sequence(
    const std::vector<std::vector<int>>& sequences,
    const Matrix& beam_scores) {
    // Find sequence with highest score after length penalty
    float best_score = -std::numeric_limits<float>::infinity();
    size_t best_idx = 0;
    
    for (size_t i = 0; i < sequences.size(); i++) {
        float penalized_score = apply_length_penalty(beam_scores(i, 0), sequences[i].size());
        if (penalized_score > best_score) {
            best_score = penalized_score;
            best_idx = i;
        }
    }
    
    return sequences[best_idx];
}

std::vector<int> BeamSearch::cpu_beam_search(
    const std::vector<float>& initial_logits,
    std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
    size_t max_length) {
    // Initialize with top beam_width_ tokens
    std::vector<std::pair<std::vector<int>, float>> beams;
    std::vector<std::pair<float, int>> top_tokens;
    
    // Get initial top tokens
    for (size_t i = 0; i < initial_logits.size(); i++) {
        top_tokens.push_back({initial_logits[i], static_cast<int>(i)});
    }
    
    std::partial_sort(top_tokens.begin(), 
                      top_tokens.begin() + beam_width_,
                      top_tokens.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Initialize beams
    for (size_t i = 0; i < beam_width_; i++) {
        beams.push_back({{top_tokens[i].second}, top_tokens[i].first});
    }
    
    // Main search loop
    for (size_t step = 1; step < max_length; step++) {
        std::vector<std::pair<std::vector<int>, float>> new_beams;
        
        // Expand each beam
        for (const auto& [sequence, score] : beams) {
            if (sequence.back() == eos_token_id_) {  // Use consistent eos_token_id_
                new_beams.push_back({sequence, score});
                continue;
            }
            
            // Get next token logits from the model
            std::vector<float> next_logits = next_token_fn(sequence);
            
            // Apply temperature scaling and sampling
            std::vector<float> scaled_probs = calculateScores(next_logits);
            
            // Get top-k candidates
            auto candidate_pairs = topKSampling(scaled_probs, top_k);
            std::vector<size_t> candidate_indices;
            for (const auto& pair : candidate_pairs) {
                candidate_indices.push_back(pair.second);
            }
            
            // Apply nucleus sampling if enabled
            if (top_p < 1.0f) {
                auto nucleus_pairs = nucleusSampling(scaled_probs, top_p);
                std::vector<size_t> nucleus_indices;
                for (const auto& pair : nucleus_pairs) {
                    nucleus_indices.push_back(pair.second);
                }
                
                // Use intersection of top-k and nucleus sampling
                std::vector<size_t> filtered_indices;
                std::set_intersection(
                    candidate_indices.begin(), candidate_indices.end(),
                    nucleus_indices.begin(), nucleus_indices.end(),
                    std::back_inserter(filtered_indices)
                );
                candidate_indices = filtered_indices;
            }
            
            // Create beam candidates
            std::vector<BeamCandidate> candidates;
            for (size_t idx : candidate_indices) {
                // Convert int sequence to size_t sequence
                std::vector<size_t> new_sequence;
                new_sequence.reserve(sequence.size() + 1);
                for (int token : sequence) {
                    new_sequence.push_back(static_cast<size_t>(token));
                }
                new_sequence.push_back(idx);
                float new_score = score + std::log(scaled_probs[idx]);
                candidates.push_back(BeamCandidate(new_sequence, new_score));
            }
            
            // Apply diversity penalty
            diversityPenalty(candidates, diversity_strength);
            
            // Add top candidates to new beams
            std::sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) { return a.score > b.score; });
            
            size_t num_to_add = std::min(beam_width_, candidates.size());
            for (size_t i = 0; i < num_to_add; i++) {
                // Convert size_t sequence back to int sequence
                std::vector<int> int_sequence;
                int_sequence.reserve(candidates[i].sequence.size());
                for (size_t token : candidates[i].sequence) {
                    int_sequence.push_back(static_cast<int>(token));
                }
                new_beams.push_back({int_sequence, candidates[i].score});
            }
        }
        
        // Sort and prune beams
        std::partial_sort(new_beams.begin(),
                         new_beams.begin() + beam_width_,
                         new_beams.end(),
                         [](const auto& a, const auto& b) { 
                             return a.second > b.second; 
                         });
        
        beams.clear();
        beams.insert(beams.end(), 
                    new_beams.begin(),
                    new_beams.begin() + std::min(beam_width_, new_beams.size()));
        
        // Check if all beams have ended
        bool all_ended = true;
        for (const auto& [sequence, _] : beams) {
            if (sequence.back() != eos_token_id_) {  // Use consistent eos_token_id_
                all_ended = false;
                break;
            }
        }
        if (all_ended) break;
    }
    
    // Return sequence with highest score after length penalty
    return std::max_element(beams.begin(), beams.end(),
                           [this](const auto& a, const auto& b) {
                               float score_a = apply_length_penalty(a.second, a.first.size());
                               float score_b = apply_length_penalty(b.second, b.first.size());
                               return score_a < score_b;
                           })->first;
}

std::vector<BeamSearch::Hypothesis>
BeamSearch::search(const std::vector<float>& initial_logits,
                   std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
                   size_t max_length, int eos_token_id) {
    try {
        return search_cuda(initial_logits, next_token_fn, max_length, eos_token_id);
    } catch (const std::runtime_error& e) {
        std::cerr << "CUDA beam search failed: " << e.what() << "\nFalling back to CPU implementation" << std::endl;
        return search_cpu(initial_logits, next_token_fn, max_length, eos_token_id);
    }
}

void BeamSearch::diversityPenalty(std::vector<BeamCandidate>& candidates, float strength) {
    // Apply much stronger diversity penalty
    const float base_penalty = strength * 4.0f;  // Increased from 2.0f to 4.0f
    
    // Track unique tokens to penalize repetition across beams
    std::unordered_map<size_t, int> global_token_counts;
    
    // First pass: count all tokens across all candidates
    for (const auto& candidate : candidates) {
        for (const auto& token : candidate.sequence) {
            global_token_counts[token]++;
        }
    }
    
    // Second pass: apply penalties
    for (size_t i = 0; i < candidates.size(); i++) {
        float total_penalty = 0.0f;
        
        // Check for self-repetition within the sequence
        std::unordered_map<size_t, int> local_token_counts;
        for (const auto& token : candidates[i].sequence) {
            local_token_counts[token]++;
            if (local_token_counts[token] > 1) {
                total_penalty += base_penalty * (local_token_counts[token] - 1) * 2.0f;
            }
            
            // Add penalty based on global token frequency
            if (global_token_counts[token] > 1) {
                total_penalty += base_penalty * (global_token_counts[token] - 1);
            }
        }
        
        // Check for overlap with higher-scored candidates
        for (size_t j = 0; j < i; j++) {
            float overlap = calculateOverlap(candidates[i].sequence, candidates[j].sequence);
            total_penalty += base_penalty * overlap * 3.0f;  // Increased overlap penalty
        }
        
        // Apply stronger penalty for first token repetition
        if (i > 0 && !candidates[i].sequence.empty() && !candidates[0].sequence.empty()) {
            if (candidates[i].sequence[0] == candidates[0].sequence[0]) {
                total_penalty += base_penalty * 5.0f;  // Heavy penalty for same first token
            }
        }
        
        // Apply the penalties
        candidates[i].score -= total_penalty;
    }
}

std::vector<float> BeamSearch::calculateScores(const std::vector<float>& logits) {
    const float eps = 1e-10f;
    std::vector<float> probs(logits.size());
    
    // Get raw logits
    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = logits[i];
    }
    
    // Apply temperature scaling
    float max_logit = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] = std::exp((probs[i] - max_logit) / temperature);
        sum += probs[i];
    }
    
    // Normalize probabilities
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum;
    }
    
    // Add noise before top-k/p filtering
    for (size_t i = 0; i < probs.size(); i++) {
        float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;  // Increased noise
        probs[i] = std::max(probs[i] + noise * probs[i], eps);  // Scale noise by prob
    }
    
    // Re-normalize after noise
    sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum;
    }
    
    // Apply top-k filtering
    if (top_k > 0 && top_k < probs.size()) {
        std::vector<std::pair<float, size_t>> prob_idx;
        prob_idx.reserve(probs.size());
        for (size_t i = 0; i < probs.size(); i++) {
            prob_idx.push_back({probs[i], i});
        }
        
        // Sort by probability
        std::partial_sort(prob_idx.begin(), 
                         prob_idx.begin() + top_k,
                         prob_idx.end(),
                         std::greater<>());
        
        // Zero out probabilities below top-k
        std::vector<float> filtered_probs(probs.size(), 0.0f);
        for (size_t i = 0; i < top_k; i++) {
            filtered_probs[prob_idx[i].second] = prob_idx[i].first;
        }
        probs = std::move(filtered_probs);
    }
    
    // Apply nucleus (top-p) sampling
    if (top_p < 1.0f) {
        std::vector<std::pair<float, size_t>> sorted_probs;
        sorted_probs.reserve(probs.size());
        for (size_t i = 0; i < probs.size(); i++) {
            if (probs[i] > 0.0f) {
                sorted_probs.push_back({probs[i], i});
            }
        }
        
        // Sort by probability
        std::sort(sorted_probs.begin(), sorted_probs.end(), std::greater<>());
        
        // Find cutoff index
        float cumsum = 0.0f;
        size_t cutoff_idx = sorted_probs.size();
        for (size_t i = 0; i < sorted_probs.size(); i++) {
            cumsum += sorted_probs[i].first;
            if (cumsum > top_p) {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        // Zero out probabilities below cutoff
        std::vector<float> filtered_probs(probs.size(), 0.0f);
        for (size_t i = 0; i < cutoff_idx; i++) {
            filtered_probs[sorted_probs[i].second] = sorted_probs[i].first;
        }
        probs = std::move(filtered_probs);
    }
    
    // Final normalization
    sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum > 0.0f) {
        for (size_t i = 0; i < probs.size(); i++) {
            probs[i] /= sum;
        }
    } else {
        // Fallback to uniform distribution if all probs are zero
        float uniform_prob = 1.0f / probs.size();
        std::fill(probs.begin(), probs.end(), uniform_prob);
    }
    
    return probs;
}

std::vector<std::pair<float, size_t>> BeamSearch::topKSampling(
    const std::vector<float>& probabilities, size_t k) {
    std::vector<std::pair<float, size_t>> prob_idx;
    prob_idx.reserve(probabilities.size());
    
    // Add random noise to break ties and increase diversity
    std::vector<float> noisy_probs = probabilities;
    for (float& prob : noisy_probs) {
        float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.02f;
        prob = std::max(0.0f, prob + noise);
    }
    
    for (size_t i = 0; i < noisy_probs.size(); i++) {
        prob_idx.push_back({noisy_probs[i], i});
    }
    
    // Sort by probability in descending order
    std::partial_sort(
        prob_idx.begin(),
        prob_idx.begin() + std::min(k, prob_idx.size()),
        prob_idx.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    // Return top k pairs
    return std::vector<std::pair<float, size_t>>(
        prob_idx.begin(),
        prob_idx.begin() + std::min(k, prob_idx.size())
    );
}

std::vector<std::pair<float, size_t>> BeamSearch::nucleusSampling(
    const std::vector<float>& probabilities, float p) {
    std::vector<std::pair<float, size_t>> sorted_probs;
    sorted_probs.reserve(probabilities.size());
    
    for (size_t i = 0; i < probabilities.size(); i++) {
        sorted_probs.push_back({probabilities[i], i});
    }
    
    // Sort by probability in descending order
    std::sort(sorted_probs.begin(), sorted_probs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Find cutoff for cumulative probability >= p
    float cumsum = 0.0f;
    size_t cutoff_idx = sorted_probs.size();
    
    for (size_t i = 0; i < sorted_probs.size(); i++) {
        cumsum += sorted_probs[i].first;
        if (cumsum >= p) {
            cutoff_idx = i + 1;
            break;
        }
    }
    
    return std::vector<std::pair<float, size_t>>(
        sorted_probs.begin(),
        sorted_probs.begin() + cutoff_idx
    );
}

std::vector<BeamSearch::Hypothesis> BeamSearch::search_cuda(
    const std::vector<float>& initial_logits,
    std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
    size_t max_length,
    int eos_token_id) {
#ifdef USE_CUDA
    try {
        auto& memory_mgr = cuda::MemoryManager::instance();
        
        // Initialize beam scores and sequences
        Matrix beam_scores(beam_width_, 1);
        std::vector<std::vector<int>> sequences(beam_width_);
        
        // Get initial top-k candidates
        Matrix top_k_logits(beam_width_, 1);
        std::vector<int> top_k_indices(beam_width_);
        cuda::topk(initial_logits, top_k_logits, top_k_indices, beam_width_);
        
        // Initialize sequences with top-k tokens
        for (size_t i = 0; i < beam_width_; i++) {
            sequences[i].push_back(top_k_indices[i]);
            beam_scores(i, 0) = top_k_logits(i, 0);
        }
        
        // Main beam search loop
        for (size_t step = 1; step < max_length; step++) {
            Matrix next_scores;
            std::vector<int> next_tokens;
            
            // Get next token predictions
            Matrix model_output = Matrix::from_vector(next_token_fn(sequences[0]));
            cuda::beam_search_step(model_output, beam_scores, 
                                 next_scores, next_tokens, beam_width_);
            
            // Update sequences and scores
            update_beams(sequences, beam_scores, next_scores, next_tokens);
            
            // Check for completion
            if (is_search_complete(sequences)) break;
        }
        
        // Return all sequences as hypotheses
        std::vector<Hypothesis> hypotheses;
        for (size_t i = 0; i < sequences.size(); i++) {
            float penalized_score = apply_length_penalty(beam_scores(i, 0), sequences[i].size());
            hypotheses.push_back(Hypothesis{sequences[i], penalized_score});
        }
        
        // Sort hypotheses by score
        std::sort(hypotheses.begin(), hypotheses.end(),
                 [](const auto& a, const auto& b) { return a.score > b.score; });
        
        return hypotheses;
    } catch (const std::runtime_error& e) {
        throw;
    }
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
}

std::vector<BeamSearch::Hypothesis> BeamSearch::search_cpu(
    const std::vector<float>& initial_logits,
    std::function<std::vector<float>(const std::vector<int>&)> next_token_fn,
    size_t max_length,
    int eos_token_id) {
    
    // Safety checks
    if (initial_logits.empty()) {
        return {};
    }
    
    // Limit max_length to prevent excessive memory usage
    const size_t MAX_SAFE_LENGTH = 128;
    max_length = std::min(max_length, MAX_SAFE_LENGTH);
    
    // Limit beam width to prevent combinatorial explosion
    const size_t MAX_SAFE_BEAM_WIDTH = 10;
    size_t effective_beam_width = std::min(beam_width_, MAX_SAFE_BEAM_WIDTH);
    
    // Initialize with top beam_width_ tokens
    std::vector<std::pair<std::vector<int>, float>> beams;
    std::vector<std::pair<float, int>> top_tokens;
    top_tokens.reserve(std::min(initial_logits.size(), size_t(1000))); // Prevent excessive allocation
    
    // Get initial top tokens (safely)
    for (size_t i = 0; i < std::min(initial_logits.size(), size_t(1000)); i++) {
        top_tokens.push_back({initial_logits[i], static_cast<int>(i)});
    }
    
    if (top_tokens.empty()) {
        return {};
    }
    
    std::partial_sort(top_tokens.begin(), 
                      top_tokens.begin() + std::min(effective_beam_width, top_tokens.size()),
                      top_tokens.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Initialize beams (safely)
    for (size_t i = 0; i < std::min(effective_beam_width, top_tokens.size()); i++) {
        beams.push_back({{top_tokens[i].second}, top_tokens[i].first});
    }
    
    if (beams.empty()) {
        return {};
    }
    
    // Main search loop
    for (size_t step = 1; step < max_length; step++) {
        std::vector<std::pair<std::vector<int>, float>> new_beams;
        new_beams.reserve(effective_beam_width * 2); // Reserve reasonable space
        
        // Expand each beam
        for (const auto& [sequence, score] : beams) {
            if (!sequence.empty() && sequence.back() == eos_token_id) {
                new_beams.push_back({sequence, score});
                continue;
            }
            
            try {
                // Get next token logits from the model
                std::vector<float> next_logits = next_token_fn(sequence);
                if (next_logits.empty()) continue;
                
                // Apply temperature scaling and sampling
                std::vector<float> scaled_probs = calculateScores(next_logits);
                
                // Get top-k candidates (safely)
                auto candidates = topKSampling(scaled_probs, std::min(effective_beam_width, size_t(5)));
                
                // Add candidates to new beams (with size check)
                for (const auto& [prob, token_id] : candidates) {
                    if (new_beams.size() >= effective_beam_width * 2) break;
                    
                    std::vector<int> new_sequence = sequence;
                    new_sequence.push_back(token_id);
                    new_beams.push_back({new_sequence, score + std::log(prob)});
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in beam expansion: " << e.what() << std::endl;
                continue;
            }
        }
        
        if (new_beams.empty()) {
            break;
        }
        
        // Sort and prune beams (safely)
        size_t sort_size = std::min(effective_beam_width, new_beams.size());
        std::partial_sort(new_beams.begin(),
                         new_beams.begin() + sort_size,
                         new_beams.end(),
                         [](const auto& a, const auto& b) { 
                             return a.second > b.second; 
                         });
        
        beams.clear();
        beams.insert(beams.end(), 
                    new_beams.begin(),
                    new_beams.begin() + sort_size);
                    
        // Early stopping if all beams end with EOS
        bool all_finished = true;
        for (const auto& [seq, _] : beams) {
            if (seq.empty() || seq.back() != eos_token_id) {
                all_finished = false;
                break;
            }
        }
        if (all_finished) break;
    }
    
    // Convert to hypotheses (safely)
    std::vector<Hypothesis> hypotheses;
    hypotheses.reserve(std::min(beams.size(), effective_beam_width));
    
    for (size_t i = 0; i < std::min(beams.size(), effective_beam_width); i++) {
        const auto& [sequence, score] = beams[i];
        hypotheses.push_back(Hypothesis{sequence, score});
    }
    
    // Sort hypotheses by score
    std::sort(hypotheses.begin(), hypotheses.end(),
              [this](const auto& a, const auto& b) {
                  float score_a = apply_length_penalty(a.score, a.tokens.size());
                  float score_b = apply_length_penalty(b.score, b.tokens.size());
                  return score_a > score_b;
              });
    
    return hypotheses;
}
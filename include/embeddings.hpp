#pragma once
#include "matrix.hpp"
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

// Forward declarations
class TokenEmbedding;
class PositionalEncoding;

/**
 * @brief Token embedding layer that converts token IDs to dense vectors.
 * 
 * The TokenEmbedding class manages the embedding lookup table that maps token IDs
 * to continuous vector representations. Features include:
 * - Learnable embedding weights
 * - Gradient computation for training
 * - CUDA acceleration support
 * - Vocabulary projection for language modeling
 */
class TokenEmbedding {
  private:
    Matrix weights_;              ///< Embedding weight matrix of shape [vocab_size, embedding_dim]
    mutable Matrix weights_grad_; ///< Gradient matrix for embedding weights
    size_t vocab_size_;          ///< Size of the vocabulary
    size_t embedding_dim_;       ///< Dimension of each embedding vector

  public:
    /**
     * @brief Constructs a token embedding layer.
     * @param vocab_size Size of the vocabulary
     * @param embedding_dim Dimension of each embedding vector
     */
    TokenEmbedding(size_t vocab_size, size_t embedding_dim);

    /**
     * @brief Converts token IDs to embedding vectors.
     * @param tokens Vector of input token IDs
     * @return Matrix of shape [num_tokens, embedding_dim] containing embeddings
     */
    Matrix forward(const std::vector<int>& tokens);

    /**
     * @brief Projects hidden states back to vocabulary space.
     * @param hidden_states Matrix of hidden states
     * @return Matrix of logits over vocabulary
     */
    Matrix project_to_vocab(const Matrix& hidden_states);

    /**
     * @brief Computes gradients during backpropagation.
     * @param grad_output Gradient of the loss with respect to the output
     * @param input_tokens Original input token IDs
     */
    void backward(const Matrix& grad_output, const std::vector<int>& input_tokens);

    /**
     * @brief CUDA-accelerated forward pass.
     * @param tokens Vector of input token IDs
     * @param output Output matrix to store embeddings
     */
    virtual void forward_cuda(const std::vector<int>& tokens, Matrix& output);

    /**
     * @brief CUDA-accelerated vocabulary projection.
     * @param input Matrix of hidden states
     * @return Matrix of logits over vocabulary
     */
    virtual Matrix project_to_vocab_cuda(const Matrix& input);

    /**
     * @brief Gets the embedding weights matrix.
     * @return Reference to the weights matrix
     */
    Matrix& get_weights() { return weights_; }

    /**
     * @brief Gets the embedding weights matrix (const version).
     * @return Const reference to the weights matrix
     */
    const Matrix& get_weights() const { return weights_; }

    /**
     * @brief Gets the embedding weight matrix (const).
     * @return Const reference to embedding weights
     */
    const Matrix& get_embedding_table() const {
        return weights_;
    }

    /**
     * @brief Gets the embedding weight matrix (mutable).
     * @return Reference to embedding weights
     */
    Matrix& get_embedding_table() {
        return weights_;
    }

    /**
     * @brief Gets the gradient matrix (const).
     * @return Const reference to gradient matrix
     */
    const Matrix& get_gradient_table() const {
        return weights_grad_;
    }

    /**
     * @brief Gets the gradient matrix (mutable).
     * @return Reference to gradient matrix
     */
    Matrix& get_gradient_table() {
        return weights_grad_;
    }

    /**
     * @brief Gets the vocabulary size.
     * @return Size of vocabulary
     */
    size_t get_vocab_size() const {
        return vocab_size_;
    }

    /**
     * @brief Gets the embedding dimension.
     * @return Dimension of embedding vectors
     */
    size_t get_embedding_dim() const {
        return embedding_dim_;
    }

    /**
     * @brief Container for trainable parameters.
     */
    struct Parameters {
        std::vector<std::reference_wrapper<Matrix>> matrices;  ///< References to parameter matrices

        // Add iterator support
        auto begin() { return matrices.begin(); }
        auto end() { return matrices.end(); }
        auto begin() const { return matrices.begin(); }
        auto end() const { return matrices.end(); }
    };

    /**
     * @brief Gets references to trainable parameters.
     * @return Parameter container
     */
    Parameters& parameters() {
        params_.matrices.clear();
        params_.matrices.emplace_back(weights_);
        return params_;
    }

    /**
     * @brief Gets references to parameter gradients.
     * @return Gradient container
     */
    const Parameters& parameter_gradients() const {
        param_gradients_.matrices.clear();
        param_gradients_.matrices.emplace_back(std::ref(const_cast<Matrix&>(weights_grad_)));
        return param_gradients_;
    }

    /**
     * @brief Saves the embedding layer to a stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const;

    /**
     * @brief Loads an embedding layer from a stream.
     * @param is Input stream to load from
     * @return Unique pointer to loaded embedding layer
     */
    static std::unique_ptr<TokenEmbedding> load(std::istream& is);

  private:
    Parameters params_;                  ///< Container for trainable parameters
    mutable Parameters param_gradients_; ///< Container for parameter gradients
};

/**
 * @brief Positional encoding layer that adds position information to embeddings.
 * 
 * The PositionalEncoding class implements the sinusoidal position encoding used
 * in transformer models to provide sequence order information. Features:
 * - Fixed sinusoidal encodings
 * - Support for variable sequence lengths
 * - Efficient computation and caching
 */
class PositionalEncoding {
  private:
    Matrix encoding_matrix_;     ///< Pre-computed positional encodings
    size_t max_seq_length_;     ///< Maximum supported sequence length
    size_t hidden_size_;        ///< Size of the hidden dimension

  public:
    /**
     * @brief Default constructor.
     */
    PositionalEncoding() = default;

    /**
     * @brief Constructs a positional encoding layer.
     * @param max_seq_length Maximum sequence length to support
     * @param hidden_size Size of the hidden dimension
     */
    PositionalEncoding(size_t max_seq_length, size_t hidden_size);

    /**
     * @brief Virtual destructor.
     */
    virtual ~PositionalEncoding() = default;

    /**
     * @brief Adds positional encodings to input embeddings.
     * @param position_ids Matrix of position IDs
     * @return Matrix with positional information added
     */
    Matrix forward(const Matrix& position_ids);

    /**
     * @brief Saves the encoding layer to a stream.
     * @param os Output stream to save to
     */
    void save(std::ostream& os) const;

    /**
     * @brief Loads an encoding layer from a stream.
     * @param is Input stream to load from
     * @return Unique pointer to loaded encoding layer
     */
    static std::unique_ptr<PositionalEncoding> load(std::istream& is);

    /**
     * @brief Gets the positional encoding matrix.
     * @return Reference to the encoding matrix
     */
    Matrix& get_weights() { return encoding_matrix_; }

    /**
     * @brief Gets the positional encoding matrix (const version).
     * @return Const reference to the encoding matrix
     */
    const Matrix& get_weights() const { return encoding_matrix_; }

    /**
     * @brief Gets the encoding matrix (const).
     * @return Const reference to encoding matrix
     */
    const Matrix& get_encoding_matrix() const {
        return encoding_matrix_;
    }

    /**
     * @brief Gets the encoding matrix (mutable).
     * @return Reference to encoding matrix
     */
    Matrix& get_encoding_matrix() {
        return encoding_matrix_;
    }

    /**
     * @brief Gets the maximum sequence length.
     * @return Maximum supported sequence length
     */
    size_t get_max_seq_length() const {
        return max_seq_length_;
    }

    /**
     * @brief Gets the hidden dimension size.
     * @return Size of hidden dimension
     */
    size_t get_hidden_size() const {
        return hidden_size_;
    }

    /**
     * @brief Gets references to trainable parameters.
     * @return Vector of parameter references
     */
    std::vector<std::reference_wrapper<Matrix>>& parameters() {
        static std::vector<std::reference_wrapper<Matrix>> params;
        params.clear();
        params.emplace_back(encoding_matrix_);
        return params;
    }
};
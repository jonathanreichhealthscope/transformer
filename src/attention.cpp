#include "../include/attention.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

Vector MultiHeadAttention::apply_rope(const Vector &x, size_t position) const {
  std::cout << "\n=== MultiHeadAttention::apply_rope START ===" << std::endl;
  std::cout << "Input vector size: " << x.size() << std::endl;
  std::cout << "Position: " << position << std::endl;

  Vector result = x;
  std::cout << "Created result vector" << std::endl;

  // Apply rotary position embeddings
  std::cout << "Applying rotary embeddings..." << std::endl;
  for (size_t i = 0; i < x.size(); i += 2) {
    if (i + 1 >= x.size()) {
      std::cout << "Breaking at i=" << i << " (odd size)" << std::endl;
      break;
    }

    float x_i = x[i];
    float x_i1 = x[i + 1];

    float cos_theta = cos_cached(position, i / 2);
    float sin_theta = sin_cached(position, i / 2);

    result[i] = x_i * cos_theta - x_i1 * sin_theta;
    result[i + 1] = x_i * sin_theta + x_i1 * cos_theta;
  }

  std::cout << "=== MultiHeadAttention::apply_rope END ===\n" << std::endl;
  return result;
}

Matrix MultiHeadAttention::flash_attention(const Matrix &Q, const Matrix &K,
                                           const Matrix &V,
                                           const AttentionMask &mask) const {
  std::cout << "\n=== MultiHeadAttention::flash_attention START ===" << std::endl;
  
  std::cout << "Input matrices dimensions:" << std::endl;
  std::cout << "Q: " << Q.rows() << "x" << Q.cols() << std::endl;
  std::cout << "K: " << K.rows() << "x" << K.cols() << std::endl;
  std::cout << "V: " << V.rows() << "x" << V.cols() << std::endl;
  
  const size_t seq_length = Q.rows();
  const size_t block_size = window_size;
  std::cout << "Sequence length: " << seq_length << std::endl;
  std::cout << "Block size: " << block_size << std::endl;
  
  Matrix output(Q.rows(), V.cols(), 0.0f);
  std::cout << "Created output matrix: " << output.rows() << "x" << output.cols() << std::endl;

  // Process in blocks for better memory efficiency
  for (size_t b_start = 0; b_start < seq_length; b_start += block_size) {
    size_t b_end = std::min(b_start + block_size, seq_length);
    std::cout << "\nProcessing block [" << b_start << ", " << b_end << "]" << std::endl;

    // Create block views
    Matrix K_block(b_end - b_start, K.cols());
    Matrix V_block(b_end - b_start, V.cols());
    std::cout << "Created block matrices:" << std::endl;
    std::cout << "K_block: " << K_block.rows() << "x" << K_block.cols() << std::endl;
    std::cout << "V_block: " << V_block.rows() << "x" << V_block.cols() << std::endl;

    // Copy block data
    std::cout << "Copying block data..." << std::endl;
    for (size_t i = b_start; i < b_end; ++i) {
      for (size_t j = 0; j < K.cols(); ++j) {
        K_block(i - b_start, j) = K(i, j);
      }
      for (size_t j = 0; j < V.cols(); ++j) {
        V_block(i - b_start, j) = V(i, j);
      }
    }
    std::cout << "Block data copied" << std::endl;

    // Compute attention scores for this block
    std::cout << "Computing attention scores..." << std::endl;
    Matrix scores = matmul(Q, K_block.transpose());
    std::cout << "Scores shape: " << scores.rows() << "x" << scores.cols() << std::endl;
    
    std::cout << "Scaling scores..." << std::endl;
    scores *= 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::cout << "Scores range after scaling: [" << scores.min() << ", " << scores.max() << "]" << std::endl;

    // Apply mask if provided
    if (!mask.mask.empty()) {
      std::cout << "Applying attention mask..." << std::endl;
      for (size_t i = 0; i < scores.rows(); ++i) {
        for (size_t j = 0; j < scores.cols(); ++j) {
          if (mask.mask(i, j) == 0.0f) {
            scores(i, j) = -std::numeric_limits<float>::infinity();
          }
        }
      }
      std::cout << "Mask applied" << std::endl;
    } else {
      std::cout << "No mask to apply" << std::endl;
    }

    // Apply softmax
    std::cout << "Applying softmax..." << std::endl;
    scores.apply_softmax();
    std::cout << "Softmax applied" << std::endl;
    std::cout << "Scores range after softmax: [" << scores.min() << ", " << scores.max() << "]" << std::endl;

    // Compute weighted sum
    std::cout << "Computing weighted sum..." << std::endl;
    Matrix block_output = matmul(scores, V_block);
    std::cout << "Block output shape: " << block_output.rows() << "x" << block_output.cols() << std::endl;

    // Add to output
    std::cout << "Adding block output to final output..." << std::endl;
    for (size_t i = 0; i < output.rows(); ++i) {
      for (size_t j = 0; j < output.cols(); ++j) {
        output(i, j) += block_output(i, j);
      }
    }
    std::cout << "Block output added" << std::endl;
  }

  std::cout << "Final output shape: " << output.rows() << "x" << output.cols() << std::endl;
  std::cout << "Final output range: [" << output.min() << ", " << output.max() << "]" << std::endl;
  std::cout << "=== MultiHeadAttention::flash_attention END ===\n" << std::endl;
  return output;
}

Matrix MultiHeadAttention::forward(const Matrix &x, const AttentionMask &mask,
                                   const std::optional<KVCache> &kv_cache) {
  std::cout << "\n=== MultiHeadAttention::forward START ===" << std::endl;
  
  try {
    // Validate input matrix
    std::cout << "Validating input matrix..." << std::endl;
    if (x.empty()) {
      throw std::runtime_error("Input matrix is empty");
    }
    std::cout << "Input validation passed" << std::endl;
    
    // Print input stats
    std::cout << "\nInput matrix statistics:" << std::endl;
    std::cout << "- Shape: " << x.rows() << "x" << x.cols() << std::endl;
    std::cout << "- Range: [" << x.min() << ", " << x.max() << "]" << std::endl;
    float mean = 0.0f;
    for(size_t i = 0; i < x.size(); i++) {
        mean += x.data()[i];
    }
    mean /= x.size();
    std::cout << "- Mean: " << mean << std::endl;
    
    // Validate projection matrices
    std::cout << "\nValidating projection matrices..." << std::endl;
    if (query_proj.empty() || key_proj.empty() || value_proj.empty()) {
      throw std::runtime_error("Projection matrices not initialized");
    }
    std::cout << "Projection matrices validation passed" << std::endl;
    
    // Print projection matrix stats
    std::cout << "\nProjection matrices statistics:" << std::endl;
    std::cout << "Query projection:" << std::endl;
    std::cout << "- Shape: " << query_proj.rows() << "x" << query_proj.cols() << std::endl;
    std::cout << "- Range: [" << query_proj.min() << ", " << query_proj.max() << "]" << std::endl;
    
    std::cout << "Key projection:" << std::endl;
    std::cout << "- Shape: " << key_proj.rows() << "x" << key_proj.cols() << std::endl;
    std::cout << "- Range: [" << key_proj.min() << ", " << key_proj.max() << "]" << std::endl;
    
    std::cout << "Value projection:" << std::endl;
    std::cout << "- Shape: " << value_proj.rows() << "x" << value_proj.cols() << std::endl;
    std::cout << "- Range: [" << value_proj.min() << ", " << value_proj.max() << "]" << std::endl;
    
    // Project input to Q, K, V
    std::cout << "\nProjecting input to Q, K, V..." << std::endl;
    Matrix Q = matmul(x, query_proj);
    std::cout << "Q projection complete" << std::endl;
    Matrix K = matmul(x, key_proj);
    std::cout << "K projection complete" << std::endl;
    Matrix V = matmul(x, value_proj);
    std::cout << "V projection complete" << std::endl;
    
    // Add biases
    std::cout << "\nAdding biases..." << std::endl;
    for (size_t i = 0; i < Q.rows(); ++i) {
      for (size_t j = 0; j < Q.cols(); ++j) {
        Q(i, j) += query_bias[j];
        K(i, j) += key_bias[j];
        V(i, j) += value_bias[j];
      }
    }
    std::cout << "Biases added successfully" << std::endl;
    
    // Print Q, K, V statistics
    std::cout << "\nAfter projection statistics:" << std::endl;
    std::cout << "Q matrix:" << std::endl;
    std::cout << "- Shape: " << Q.rows() << "x" << Q.cols() << std::endl;
    std::cout << "- Range: [" << Q.min() << ", " << Q.max() << "]" << std::endl;
    
    std::cout << "K matrix:" << std::endl;
    std::cout << "- Shape: " << K.rows() << "x" << K.cols() << std::endl;
    std::cout << "- Range: [" << K.min() << ", " << K.max() << "]" << std::endl;
    
    std::cout << "V matrix:" << std::endl;
    std::cout << "- Shape: " << V.rows() << "x" << V.cols() << std::endl;
    std::cout << "- Range: [" << V.min() << ", " << V.max() << "]" << std::endl;
    
    // Reshape for attention computation
    std::cout << "\nPreparing for attention computation..." << std::endl;
    size_t batch_size = x.rows();
    size_t seq_len = x.rows();  // For self-attention, seq_len = batch_size
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Sequence length: " << seq_len << std::endl;
    
    // Validate attention mask dimensions if provided
    std::cout << "\nValidating attention mask..." << std::endl;
    if (!mask.mask.empty()) {
      std::cout << "Mask shape: " << mask.mask.rows() << "x" << mask.mask.cols() << std::endl;
      if (mask.mask.rows() != seq_len || mask.mask.cols() != seq_len) {
        throw std::runtime_error("Attention mask dimension mismatch");
      }
      std::cout << "Mask validation passed" << std::endl;
    } else {
      std::cout << "No mask provided" << std::endl;
    }
    
    std::cout << "\nComputing attention..." << std::endl;
    Matrix result = compute_attention(Q, K, V, mask);
    std::cout << "Attention computation complete" << std::endl;
    std::cout << "Result shape: " << result.rows() << "x" << result.cols() << std::endl;
    std::cout << "Result range: [" << result.min() << ", " << result.max() << "]" << std::endl;
    
    std::cout << "=== MultiHeadAttention::forward END ===\n" << std::endl;
    return result;
    
  } catch (const std::exception& e) {
    std::cerr << "\nERROR in attention forward pass: " << e.what() << std::endl;
    std::cerr << "=== MultiHeadAttention::forward FAILED ===\n" << std::endl;
    throw;
  }
}

void MultiHeadAttention::save(std::ostream &os) const {
  std::cout << "\n=== MultiHeadAttention::save START ===" << std::endl;
  
  // Save dimensions and configuration
  std::cout << "Saving configuration..." << std::endl;
  std::cout << "- Number of heads: " << num_heads << std::endl;
  std::cout << "- Head dimension: " << head_dim << std::endl;
  os.write(reinterpret_cast<const char *>(&num_heads), sizeof(num_heads));
  os.write(reinterpret_cast<const char *>(&head_dim), sizeof(head_dim));

  // Save projection matrices
  std::cout << "\nSaving projection matrices..." << std::endl;
  std::cout << "Query projection shape: " << query_proj.rows() << "x" << query_proj.cols() << std::endl;
  query_proj.save(os);
  std::cout << "Key projection shape: " << key_proj.rows() << "x" << key_proj.cols() << std::endl;
  key_proj.save(os);
  std::cout << "Value projection shape: " << value_proj.rows() << "x" << value_proj.cols() << std::endl;
  value_proj.save(os);
  std::cout << "Output projection shape: " << output_proj.rows() << "x" << output_proj.cols() << std::endl;
  output_proj.save(os);
  
  std::cout << "=== MultiHeadAttention::save END ===\n" << std::endl;
}

std::unique_ptr<MultiHeadAttention> MultiHeadAttention::load(std::istream &is) {
  std::cout << "\n=== MultiHeadAttention::load START ===" << std::endl;
  
  // Read configuration
  std::cout << "Reading configuration..." << std::endl;
  size_t num_heads, head_dim;
  is.read(reinterpret_cast<char *>(&num_heads), sizeof(num_heads));
  is.read(reinterpret_cast<char *>(&head_dim), sizeof(head_dim));
  std::cout << "- Number of heads: " << num_heads << std::endl;
  std::cout << "- Head dimension: " << head_dim << std::endl;

  std::cout << "\nCreating attention instance..." << std::endl;
  auto attention = std::make_unique<MultiHeadAttention>(
      num_heads * head_dim, // hidden_size
      num_heads, head_dim);
  std::cout << "Attention instance created" << std::endl;

  // Load projection matrices
  std::cout << "\nLoading projection matrices..." << std::endl;
  std::cout << "Loading query projection..." << std::endl;
  attention->query_proj = Matrix::load(is);
  std::cout << "Query projection loaded: " << attention->query_proj.rows() << "x" << attention->query_proj.cols() << std::endl;
  
  std::cout << "Loading key projection..." << std::endl;
  attention->key_proj = Matrix::load(is);
  std::cout << "Key projection loaded: " << attention->key_proj.rows() << "x" << attention->key_proj.cols() << std::endl;
  
  std::cout << "Loading value projection..." << std::endl;
  attention->value_proj = Matrix::load(is);
  std::cout << "Value projection loaded: " << attention->value_proj.rows() << "x" << attention->value_proj.cols() << std::endl;
  
  std::cout << "Loading output projection..." << std::endl;
  attention->output_proj = Matrix::load(is);
  std::cout << "Output projection loaded: " << attention->output_proj.rows() << "x" << attention->output_proj.cols() << std::endl;

  // Validate loaded matrices
  std::cout << "\nValidating loaded matrices..." << std::endl;
  auto validate_matrix = [](const Matrix& m, const std::string& name) {
    if (m.empty()) {
      throw std::runtime_error(name + " is empty after loading");
    }
    std::cout << name << " statistics:" << std::endl;
    std::cout << "- Shape: " << m.rows() << "x" << m.cols() << std::endl;
    std::cout << "- Range: [" << m.min() << ", " << m.max() << "]" << std::endl;
    if (std::isnan(m.min()) || std::isnan(m.max()) || 
        std::isinf(m.min()) || std::isinf(m.max())) {
      throw std::runtime_error("Invalid values in " + name + " after loading");
    }
  };

  validate_matrix(attention->query_proj, "Query projection");
  validate_matrix(attention->key_proj, "Key projection");
  validate_matrix(attention->value_proj, "Value projection");
  validate_matrix(attention->output_proj, "Output projection");

  std::cout << "=== MultiHeadAttention::load END ===\n" << std::endl;
  return attention;
}

MultiHeadAttention::MultiHeadAttention(size_t hidden_size_, size_t num_heads_, 
                                     size_t head_dim_, float dropout_prob_,
                                     bool use_flash_, bool use_rope_,
                                     bool use_sliding_window_, size_t window_size_,
                                     bool use_gqa_, size_t num_kv_heads_)
    : num_heads(num_heads_),
      head_dim(head_dim_),
      hidden_size(hidden_size_),
      dropout_prob(dropout_prob_),
      use_flash(use_flash_),
      use_rope(use_rope_),
      use_sliding_window(use_sliding_window_),
      window_size(window_size_),
      use_gqa(use_gqa_),
      num_kv_heads(num_kv_heads_),
      // Initialize matrices in initializer list
      query_proj(Matrix(hidden_size_, num_heads_ * head_dim_)),
      key_proj(Matrix(hidden_size_, num_heads_ * head_dim_)),
      value_proj(Matrix(hidden_size_, num_heads_ * head_dim_)),
      output_proj(Matrix(num_heads_ * head_dim_, hidden_size_)),
      // Initialize bias vectors
      query_bias(FloatVector(num_heads_ * head_dim_)),
      key_bias(FloatVector(num_heads_ * head_dim_)),
      value_bias(FloatVector(num_heads_ * head_dim_)),
      output_bias(FloatVector(hidden_size_)) {
    
    std::cout << "\n=== MultiHeadAttention::constructor START ===" << std::endl;
    
    // Print configuration
    std::cout << "Configuration:" << std::endl;
    std::cout << "- Hidden size: " << hidden_size << std::endl;
    std::cout << "- Number of heads: " << num_heads << std::endl;
    std::cout << "- Head dimension: " << head_dim << std::endl;
    std::cout << "- Dropout probability: " << dropout_prob << std::endl;
    std::cout << "- Use flash attention: " << std::boolalpha << use_flash << std::endl;
    std::cout << "- Use RoPE: " << use_rope << std::endl;
    std::cout << "- Use sliding window: " << use_sliding_window << std::endl;
    std::cout << "- Window size: " << window_size << std::endl;
    std::cout << "- Use GQA: " << use_gqa << std::endl;
    std::cout << "- Number of KV heads: " << num_kv_heads << std::endl;
    
    // Validate input dimensions
    std::cout << "\nValidating dimensions..." << std::endl;
    if (hidden_size == 0 || num_heads == 0 || head_dim == 0) {
        throw std::runtime_error("Invalid dimensions: hidden_size=" + std::to_string(hidden_size) +
                               ", num_heads=" + std::to_string(num_heads) +
                               ", head_dim=" + std::to_string(head_dim));
    }
    
    if (hidden_size % num_heads != 0) {
        throw std::runtime_error("hidden_size must be divisible by num_heads");
    }
    std::cout << "Dimension validation passed" << std::endl;

    // Initialize weights with smaller scale for numerical stability
    std::cout << "\nInitializing weights..." << std::endl;
    const float MAX_INIT_VAL = 0.1f;  // Limit maximum initial value
    
    float q_scale = std::min(std::sqrt(2.0f / (hidden_size + head_dim * num_heads)), MAX_INIT_VAL);
    float kv_scale = std::min(std::sqrt(2.0f / (hidden_size + head_dim)), MAX_INIT_VAL);
    float out_scale = std::min(std::sqrt(2.0f / (hidden_size * 2)), MAX_INIT_VAL);

    std::cout << "Initialization scales:" << std::endl;
    std::cout << "- Query scale: " << q_scale << std::endl;
    std::cout << "- Key/Value scale: " << kv_scale << std::endl;
    std::cout << "- Output scale: " << out_scale << std::endl;

    std::cout << "Randomizing projection matrices..." << std::endl;
    query_proj.randomize(-q_scale, q_scale);
    key_proj.randomize(-kv_scale, kv_scale);
    value_proj.randomize(-kv_scale, kv_scale);
    output_proj.randomize(-out_scale, out_scale);

    // Initialize biases with smaller values
    std::cout << "\nInitializing biases..." << std::endl;
    const float BIAS_INIT = 0.001f;  // Reduced from 0.01f
    std::cout << "Bias initialization value: " << BIAS_INIT << std::endl;
    
    for(size_t i = 0; i < query_bias.size(); i++) query_bias[i] = BIAS_INIT;
    for(size_t i = 0; i < key_bias.size(); i++) key_bias[i] = BIAS_INIT;
    for(size_t i = 0; i < value_bias.size(); i++) value_bias[i] = BIAS_INIT;
    for(size_t i = 0; i < output_bias.size(); i++) output_bias[i] = BIAS_INIT;

    // Validate initialization
    std::cout << "\nValidating initialization..." << std::endl;
    auto validate_matrix = [](const Matrix& m, const std::string& name) {
        if (m.empty()) {
            throw std::runtime_error(name + " is empty after initialization");
        }
        std::cout << name << " statistics:" << std::endl;
        std::cout << "- Shape: " << m.rows() << "x" << m.cols() << std::endl;
        std::cout << "- Range: [" << m.min() << ", " << m.max() << "]" << std::endl;
        if (std::isnan(m.min()) || std::isnan(m.max()) || 
            std::isinf(m.min()) || std::isinf(m.max())) {
            throw std::runtime_error("Invalid values in " + name + " after initialization");
        }
    };

    validate_matrix(query_proj, "Query projection");
    validate_matrix(key_proj, "Key projection");
    validate_matrix(value_proj, "Value projection");
    validate_matrix(output_proj, "Output projection");

    std::cout << "=== MultiHeadAttention::constructor END ===\n" << std::endl;
}

Matrix MultiHeadAttention::standard_attention(const Matrix &Q, const Matrix &K,
                                              const Matrix &V,
                                              const AttentionMask &mask) {
  std::cout << "\n=== MultiHeadAttention::standard_attention START ===" << std::endl;
  
  std::cout << "Input matrices dimensions:" << std::endl;
  std::cout << "Q: " << Q.rows() << "x" << Q.cols() << std::endl;
  std::cout << "K: " << K.rows() << "x" << K.cols() << std::endl;
  std::cout << "V: " << V.rows() << "x" << V.cols() << std::endl;
  
  std::cout << "Computing attention scores..." << std::endl;
  Matrix scores = matmul(Q, K.transpose());
  std::cout << "Raw scores shape: " << scores.rows() << "x" << scores.cols() << std::endl;
  std::cout << "Raw scores range: [" << scores.min() << ", " << scores.max() << "]" << std::endl;
  
  // Clamp extreme values in scores
  std::cout << "Clamping extreme values..." << std::endl;
  for(size_t i = 0; i < scores.size(); i++) {
    scores.data()[i] = std::clamp(scores.data()[i], -10.0f, 10.0f);
  }
  std::cout << "Scores range after clamping: [" << scores.min() << ", " << scores.max() << "]" << std::endl;

  // Add numerical stability to attention scaling
  std::cout << "Applying attention scaling..." << std::endl;
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  scale = std::min(scale, 10.0f);  // Prevent too large scaling
  std::cout << "Scale factor: " << scale << std::endl;
  scores *= scale;
  std::cout << "Scores range after scaling: [" << scores.min() << ", " << scores.max() << "]" << std::endl;

  if (!mask.mask.empty()) {
    std::cout << "Applying attention mask..." << std::endl;
    std::cout << "Mask shape: " << mask.mask.rows() << "x" << mask.mask.cols() << std::endl;
    for (size_t i = 0; i < scores.rows(); ++i) {
      for (size_t j = 0; j < scores.cols(); ++j) {
        if (mask.mask(i, j) == 0.0f) {
          scores(i, j) = -1e6f;  // Use finite value instead of infinity
        }
      }
    }
    std::cout << "Mask applied" << std::endl;
  } else {
    std::cout << "No mask to apply" << std::endl;
  }

  // Add numerical stability to softmax
  std::cout << "Applying numerically stable softmax..." << std::endl;
  for (size_t i = 0; i < scores.rows(); ++i) {
    float max_val = scores(i, 0);
    for (size_t j = 1; j < scores.cols(); ++j) {
      max_val = std::max(max_val, scores(i, j));
    }
    float sum = 0.0f;
    const float epsilon = 1e-10f;
    for (size_t j = 0; j < scores.cols(); ++j) {
      scores(i, j) = std::exp(scores(i, j) - max_val);
      sum += scores(i, j);
    }
    sum = std::max(sum, epsilon);
    for (size_t j = 0; j < scores.cols(); ++j) {
      scores(i, j) /= sum;
    }
  }
  std::cout << "Softmax applied" << std::endl;
  std::cout << "Scores range after softmax: [" << scores.min() << ", " << scores.max() << "]" << std::endl;

  // Validate no NaN in output
  std::cout << "Checking for NaN values..." << std::endl;
  size_t nan_count = 0;
  for(size_t i = 0; i < scores.size(); i++) {
    if(std::isnan(scores.data()[i])) {
      std::cerr << "NaN detected at index " << i << std::endl;
      scores.data()[i] = 0.0f;  // Replace NaN with zero
      nan_count++;
    }
  }
  if (nan_count > 0) {
    std::cout << "WARNING: Replaced " << nan_count << " NaN values with 0" << std::endl;
  } else {
    std::cout << "No NaN values found" << std::endl;
  }

  // Store attention scores for backward pass
  std::cout << "Storing attention scores for backward pass..." << std::endl;
  attention_scores = scores;
  
  std::cout << "Computing final output..." << std::endl;
  Matrix output = matmul(scores, V);
  std::cout << "Output shape: " << output.rows() << "x" << output.cols() << std::endl;
  std::cout << "Output range: [" << output.min() << ", " << output.max() << "]" << std::endl;
  
  std::cout << "=== MultiHeadAttention::standard_attention END ===\n" << std::endl;
  return output;
}

Matrix MultiHeadAttention::backward(const Matrix& grad_output,
                                  const Matrix& input,
                                  const Matrix& target_distribution) {
    std::cout << "\n=== MultiHeadAttention::backward START ===" << std::endl;
    
    try {
        // Store dimensions for debugging
        std::cout << "Configuration:" << std::endl;
        std::cout << "- Hidden size: " << hidden_size << std::endl;
        std::cout << "- Num heads: " << num_heads << std::endl;
        std::cout << "- Head dim: " << head_dim << std::endl;
        
        std::cout << "\nInput dimensions:" << std::endl;
        std::cout << "- Gradient output: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
        std::cout << "- Input: " << input.rows() << "x" << input.cols() << std::endl;
        std::cout << "- Target distribution: " << target_distribution.rows() << "x" << target_distribution.cols() << std::endl;
        
        // Create local copy of gradient that we can modify
        std::cout << "\nCreating local gradient copy..." << std::endl;
        Matrix grad = grad_output;
        std::cout << "Local gradient shape: " << grad.rows() << "x" << grad.cols() << std::endl;
        
        // Add gradient norm check with adaptive scaling
        std::cout << "Computing gradient norm..." << std::endl;
        float grad_norm = 0.0f;
        for(size_t i = 0; i < grad.size(); i++) {
            grad_norm += grad.data()[i] * grad.data()[i];
        }
        grad_norm = std::sqrt(grad_norm);
        std::cout << "Initial gradient norm: " << grad_norm << std::endl;
        
        const float MIN_GRAD_NORM = 1e-4f;  // Increased minimum gradient norm
        if (grad_norm < MIN_GRAD_NORM) {
            std::cout << "WARNING: Small gradient norm detected, applying scaling" << std::endl;
            // Scale up gradients to prevent vanishing
            float scale = MIN_GRAD_NORM / (grad_norm + 1e-8f);
            std::cout << "Scaling factor: " << scale << std::endl;
            for(size_t i = 0; i < grad.size(); i++) {
                grad.data()[i] *= scale;
            }
            std::cout << "Gradient scaled up" << std::endl;
        }

        // Validate dimensions
        std::cout << "\nValidating dimensions..." << std::endl;
        validate_dimensions(grad, input, target_distribution);
        std::cout << "Dimension validation passed" << std::endl;
        
        // Compute gradients with numerical stability
        std::cout << "\nComputing query gradients..." << std::endl;
        Matrix dQ = compute_query_gradients(grad, input);
        std::cout << "Query gradients shape: " << dQ.rows() << "x" << dQ.cols() << std::endl;
        
        std::cout << "Computing key gradients..." << std::endl;
        Matrix dK = compute_key_gradients(grad, input);
        std::cout << "Key gradients shape: " << dK.rows() << "x" << dK.cols() << std::endl;
        
        std::cout << "Computing value gradients..." << std::endl;
        Matrix dV = compute_value_gradients(grad, input);
        std::cout << "Value gradients shape: " << dV.rows() << "x" << dV.cols() << std::endl;
        
        // Stabilize gradients
        std::cout << "\nStabilizing gradients..." << std::endl;
        auto stabilize_gradients = [](Matrix& grad) {
            const float MAX_GRAD = 1.0f;
            const float EPSILON = 1e-6f;
            size_t clipped_count = 0;
            size_t epsilon_count = 0;
            
            for(size_t i = 0; i < grad.size(); i++) {
                if (std::abs(grad.data()[i]) > MAX_GRAD) {
                    grad.data()[i] = std::clamp(grad.data()[i], -MAX_GRAD, MAX_GRAD);
                    clipped_count++;
                }
                if (std::abs(grad.data()[i]) < EPSILON) {
                    grad.data()[i] = grad.data()[i] < 0 ? -EPSILON : EPSILON;
                    epsilon_count++;
                }
            }
            
            std::cout << "- Clipped " << clipped_count << " values to [-" << MAX_GRAD << ", " << MAX_GRAD << "]" << std::endl;
            std::cout << "- Applied epsilon to " << epsilon_count << " small values" << std::endl;
        };
        
        std::cout << "Stabilizing query gradients..." << std::endl;
        stabilize_gradients(dQ);
        std::cout << "Stabilizing key gradients..." << std::endl;
        stabilize_gradients(dK);
        std::cout << "Stabilizing value gradients..." << std::endl;
        stabilize_gradients(dV);

        // Combine gradients
        std::cout << "\nCombining gradients..." << std::endl;
        Matrix combined = combine_gradients(dQ, dK, dV);
        std::cout << "Combined gradients shape: " << combined.rows() << "x" << combined.cols() << std::endl;
        std::cout << "Combined gradients range: [" << combined.min() << ", " << combined.max() << "]" << std::endl;
        
        std::cout << "=== MultiHeadAttention::backward END ===\n" << std::endl;
        return combined;
        
    } catch (const std::exception& e) {
        std::cerr << "\nERROR in attention backward: " << e.what() << std::endl;
        std::cerr << "=== MultiHeadAttention::backward FAILED ===\n" << std::endl;
        throw;
    }
}

Matrix MultiHeadAttention::reshape_for_attention(const Matrix& x, size_t batch_size, 
                                               size_t num_heads, size_t seq_len, size_t head_size) const {
    // Reshape from [batch_size, seq_len, hidden_size] to 
    // [batch_size * num_heads, seq_len, head_size]
    std::cout << "In reshaped for attention" << std::endl;
    std::cout << "Input matrix shape: " << x.shape() << std::endl;
    std::cout << "num_heads: " << num_heads << std::endl;
    std::cout << "head_size: " << head_size << std::endl;
    std::cout << "seq_len: " << seq_len << std::endl;
    std::cout << "batch_size: " << batch_size << std::endl;
    Matrix reshaped(batch_size * num_heads, seq_len, num_heads);
    std::cout << "Reshaped matrix shape: " << reshaped.shape() << std::endl;
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t s = 0; s < x.rows(); s++) {
                for (size_t d = 0; d < head_size; d++) {
                    size_t src_idx = s * x.cols() + h * head_size + d;
                    size_t tgt_idx = (b * num_heads + h) * x.rows() + s;
                    reshaped.data()[tgt_idx * head_size + d] = x.data()[src_idx];
                }
            }
        }
    }
    return reshaped;
}


   

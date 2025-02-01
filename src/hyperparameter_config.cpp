TransformerConfig HyperparameterConfig::to_transformer_config() const {
    TransformerConfig config;
    
    // Architecture parameters
    config.num_layers = num_layers;
    config.num_heads = num_heads;
    config.hidden_size = hidden_size;
    config.intermediate_size = intermediate_size;
    config.head_dim = head_dim;
    
    // Learning rate parameters
    config.initial_lr = initial_lr;
    config.peak_lr = peak_lr;
    config.warmup_steps = warmup_steps;
    config.decay_factor = decay_factor;
    
    // Training parameters
    config.dropout_rate = dropout_rate;
    config.weight_decay = weight_decay;
    config.early_stopping_patience = early_stopping_patience;
    config.early_stopping_threshold = early_stopping_threshold;
    config.gradient_clip_threshold = gradient_clip_threshold;
    config.layer_norm_epsilon = layer_norm_epsilon;
    
    // Memory and optimization
    config.memory_pool_size = memory_pool_size;
    config.gradient_accumulation_steps = gradient_accumulation_steps;
    
    return config;
} 
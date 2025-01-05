# A decoder-style Transformer in C++

A pure C++ implementation of a decoder-only transformer model with CUDA support. It is based on the paper "Attention is All You Need" by Vaswani et al and has been trained on an example dataset found in the `data` directory called 'training_pairs.txt'. It performs a single token prediction for each input.

## Transformer Implementation Features

## Core Attention Mechanisms

- Standard Multi-Head Attention
- Grouped Query Attention (GQA)
- Flash Attention optimization
- Rotary Position Embeddings (RoPE)
- Sliding Window Attention
- Key-Value Cache support

## Architecture Components

- Layer Normalization
- Feed Forward Networks
- Dropout layers
- Residual connections
- Language Model Head
- Tokenizer with vocabulary management

## Training Features

- Batch processing
- Dynamic learning rate adjustment
- Gradient clipping
- Loss computation and backpropagation
- Training/Evaluation modes
- Gradient checkpointing
- Performance metrics tracking

## Optimization Features

- CUDA support for GPU acceleration
- OpenMP parallelisation
- Half-precision (FP16) support
- Memory pooling
- Gradient accumulation
- SAM (Sharpness-Aware Minimization) optimizer

## Advanced Features

- Quantization-Aware Training
- Adaptive cache replacement policies
- Token embedding with positional encoding
- Advanced attention mechanisms (block-sparse)
- Configurable model architecture

## Utility Features

- JSON configuration loading
- Model checkpointing and saving
- Performance metrics logging
- Validation data evaluation
- Token prediction and probability calculation
- Text preprocessing and tokenization

## Memory Management

- Memory pooling
- Cache management
- Gradient checkpointing
- Efficient matrix operations

## Development Features

- Comprehensive logging
- Error handling
- Configuration validation
- Performance profiling
- Debug output options

## Dependencies

- OpenMP: <https://github.com/OpenMP/openmp-api>
- CUDA: <https://developer.nvidia.com/cuda-downloads>
- nlohmann/json: <https://github.com/nlohmann/json>

## Building

To build the project, you can use the following commands:

```bash
mkdir build
cd build
cmake ..
make
```

## Training the model

After building the project, running `main.cpp` will train the model and save the model hyperparamters to whatever directory is specified in the `config/transformer_config.json` file. To execute the training on the sample dataset, run the following command
from the build directory:

```bash
./transformer
```

## Logging

The logging is done to a file called `transformer.log` in the `build` directory.

## Configuration

The configuration is done in the `config/transformer_config.json` file.

## Limitations

- The model training is performed on a very small dataset, so its predictions are certainly sub-optimal, given its constraints.
- It only works on a format that follows the training data i.e I like to cook in the |kitchen (| is the delimiter).

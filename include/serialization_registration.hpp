#pragma once
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>
#include "transformer.hpp"

// Register polymorphic types with cereal
CEREAL_REGISTER_TYPE(LayerNorm)
CEREAL_REGISTER_TYPE(MultiHeadAttention)
CEREAL_REGISTER_TYPE(FeedForward)
CEREAL_REGISTER_TYPE(TokenEmbedding)
CEREAL_REGISTER_TYPE(PositionalEncoding)
CEREAL_REGISTER_TYPE(TransformerLayer)

// Register base classes
CEREAL_REGISTER_POLYMORPHIC_RELATION(LayerNorm, LayerNorm)
CEREAL_REGISTER_POLYMORPHIC_RELATION(MultiHeadAttention, MultiHeadAttention)
CEREAL_REGISTER_POLYMORPHIC_RELATION(FeedForward, FeedForward)
CEREAL_REGISTER_POLYMORPHIC_RELATION(TokenEmbedding, TokenEmbedding)
CEREAL_REGISTER_POLYMORPHIC_RELATION(PositionalEncoding, PositionalEncoding)
CEREAL_REGISTER_POLYMORPHIC_RELATION(TransformerLayer, TransformerLayer) 
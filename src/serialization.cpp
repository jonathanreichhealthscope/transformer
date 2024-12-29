#include "../include/serialization.hpp"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <fstream>

void save_model(const std::string& path, const Transformer& model) {
    std::ofstream os(path, std::ios::binary);
    cereal::BinaryOutputArchive archive(os);
    
    // Save config
    archive(model.config);
    
    // Save layers
    archive(model.layers);
    
    // Save embeddings
    archive(model.token_embedding);
    archive(model.pos_encoding);
    
    // Save final layer norm
    archive(model.final_ln);
}

void load_model(const std::string& path, Transformer& model) {
    std::ifstream is(path, std::ios::binary);
    cereal::BinaryInputArchive archive(is);
    
    // Load config
    archive(model.config);
    
    // Load layers
    archive(model.layers);
    
    // Load embeddings
    archive(model.token_embedding);
    archive(model.pos_encoding);
    
    // Load final layer norm
    archive(model.final_ln);
    
#ifdef USE_CUDA
    if (model.config.use_cuda) {
        model.cuda_manager = std::make_unique<CudaManager>();
    }
#endif
} 
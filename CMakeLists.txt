cmake_minimum_required(VERSION 3.15)
project(transformer_cpp CUDA CXX)

# Set CUDA architecture policy
cmake_policy(SET CMP0104 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set default CUDA architectures if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80)
endif()

# Include FetchContent for downloading dependencies
include(FetchContent)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

# Download and build sentencepiece
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.1.97
)
FetchContent_MakeAvailable(sentencepiece)

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${sentencepiece_SOURCE_DIR}/src)

# Core source files
set(CORE_SOURCES
    src/components.cpp
    src/matrix_ops.cpp
    src/vector_ops.cpp
)

# Neural network components
set(NN_SOURCES
    src/attention.cpp
    src/attention_mask.cpp
    src/transformer.cpp
    src/layernorm.cpp
    src/feed_forward.cpp
    src/embeddings.cpp
)

# Training and utilities
set(UTIL_SOURCES
    src/tokenizer.cpp
    src/cache.cpp
    src/trainer.cpp
    src/memory_pool.cpp
    src/quantization.cpp
    src/serialization.cpp
    src/optimizer.cpp
)

# Advanced attention mechanisms
set(ATTENTION_SOURCES
    src/attention/block_sparse.cpp
    src/attention/advanced_attention.cpp
)

# Language model components
set(LM_SOURCES
    src/lm_head.cpp
    src/vocabulary.cpp
)

# Optimization components
set(OPTIMIZER_SOURCES
    src/optimizer/sam.cpp
)

# CUDA kernels
set(CUDA_SOURCES
    src/cuda/cuda_kernels.cu
    src/cuda/quantization_kernels.cu
    src/cuda/layernorm_kernels.cu
    src/cuda/attention_kernels.cu
)

# Create library
add_library(transformer_lib
    ${CORE_SOURCES}
    ${NN_SOURCES}
    ${UTIL_SOURCES}
    ${ATTENTION_SOURCES}
    ${LM_SOURCES}
    ${OPTIMIZER_SOURCES}
    ${CUDA_SOURCES}
)

# Set CUDA specific properties for the library
set_target_properties(transformer_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create executable
add_executable(transformer src/main.cpp)

# Link libraries
target_link_libraries(transformer_lib
    PUBLIC
    sentencepiece
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

target_link_libraries(transformer
    PRIVATE
    transformer_lib
)

# Compiler options
target_compile_definitions(transformer_lib
    PUBLIC
    USE_CUDA
)

if(MSVC)
    target_compile_options(transformer_lib
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/W3">
    )
else()
    target_compile_options(transformer_lib
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    )
endif()

# Installation
install(TARGETS transformer transformer_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.cuh"
)

# Optional dependencies
find_library(TCMALLOC_LIB tcmalloc)
if(TCMALLOC_LIB)
    message(STATUS "tcmalloc found: ${TCMALLOC_LIB}")
    target_link_libraries(transformer_lib PUBLIC ${TCMALLOC_LIB})
else()
    message(WARNING "tcmalloc not found, continuing without it")
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(transformer_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
cmake_minimum_required(VERSION 3.15)
project(transformer_cpp CUDA CXX)

# Set CUDA architecture policy
cmake_policy(SET CMP0104 NEW)

# Add Doxygen support
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set default CUDA architectures if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Include FetchContent for downloading dependencies
include(FetchContent)

# Find required packages
find_package(CUDAToolkit)

# Set CUDA availability flag
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found - enabling CUDA support")
    add_definitions(-DCUDA_AVAILABLE)
else()
    message(STATUS "CUDA not found - building without CUDA support")
endif()

# Find required packages
find_package(OpenMP)

# Download and build nlohmann/json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

# Include directories
if(CUDAToolkit_FOUND)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
include_directories(${CMAKE_SOURCE_DIR}/include)

# Core source files
set(CORE_SOURCES
    src/components.cpp
    src/vector_ops.cpp
    src/logger.cpp
    src/model_saver.cpp
    src/half_precision.cpp
    src/gradient_checkpoint.cpp
    src/utils.cpp
    src/performance_metrics.cpp
    src/gqa.cpp
    src/beam_search.cpp
    src/config.cpp
)

# Neural network components
set(NN_SOURCES
    src/attention.cpp
    src/transformer.cpp
    src/layer_norm.cpp
    src/feed_forward.cpp
    src/embeddings.cpp
)

# Training and utilities
set(UTIL_SOURCES
    src/tokenizer.cpp
    src/cache.cpp
    src/quantization.cpp
    src/optimizer.cpp
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

# Add the tensor source file to the sources list
set(SOURCES
    ${SOURCES}
    src/tensor.cpp
)

# Add the tensor header to the includes
set(HEADERS
    ${HEADERS}
    include/tensor.hpp
)

# Create library
add_library(transformer_lib
    ${CORE_SOURCES}
    ${NN_SOURCES}
    ${UTIL_SOURCES}
    ${ATTENTION_SOURCES}
    ${LM_SOURCES}
    ${OPTIMIZER_SOURCES}
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
    nlohmann_json::nlohmann_json
)

if(CUDAToolkit_FOUND)
    target_link_libraries(transformer_lib
        PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )
endif()

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

# Add CUDA kernels library only if CUDA is found
if(CUDAToolkit_FOUND)
    add_library(cuda_kernels STATIC
        src/cuda/matrix_ops.cu
        src/cuda/attention_ops.cu
        src/cuda/backward_ops.cu
        src/cuda/feed_forward_kernels.cu
        src/cuda/half_precision_kernels.cu
        src/cuda/cuda_utils.cu
        src/cuda/cuda_init.cu
        src/cuda/token_embedding_cuda.cu
        src/cuda/layernorm_cuda.cu
        src/cuda/gqa_kernels.cu
    )

    # Set CUDA properties for the kernels library
    set_target_properties(cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    # Link CUDA libraries to cuda_kernels
    target_link_libraries(cuda_kernels
        PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )

    # Add CUDA include directories
    target_include_directories(cuda_kernels
        PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include/cuda
    )

    # Link cuda_kernels to transformer_lib
    target_link_libraries(transformer_lib
        PUBLIC
        cuda_kernels
    )
endif()

# Add to your sources list:
target_sources(transformer_lib PRIVATE
    src/tensor.cpp
)

find_package(nlohmann_json 3.2.0 REQUIRED)
target_link_libraries(transformer_lib PRIVATE nlohmann_json::nlohmann_json)

# Add CUDA include directories
target_include_directories(transformer_lib
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include/cuda
)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    # Set CUDA architectures
    set(CMAKE_CUDA_ARCHITECTURES 86)
    
    # Add CUDA compilation flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    
    # Add CUDA sources to library
    target_sources(transformer_lib 
        PRIVATE 
        ${CUDA_SOURCES}
    )
    
    # Link CUDA libraries
    target_link_libraries(transformer_lib
        PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )
    
    # Define CUDA macro
    target_compile_definitions(transformer_lib
        PRIVATE
        USE_CUDA
    )
endif()
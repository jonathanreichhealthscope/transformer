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

set(CMAKE_CXX_STANDARD 20)
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

# Optional: Find OpenMP
if(MSVC)
    find_package(OpenMP QUIET)
else()
    find_package(OpenMP)
endif()

# Store OpenMP status for later use
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling parallel processing support")
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    endif()
else()
    message(WARNING "OpenMP not found - parallel processing will be disabled")
endif()

# Download and build nlohmann/json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(json)

# Include directories
if(CUDAToolkit_FOUND)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# Base include directories - single source of truth for all includes
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Add tiktoken include directory globally
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tiktoken/cpp/include
)

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
    src/tokenizer.cpp
    src/tiktoken_tokenizer.cpp
    src/matrix.cpp
    src/data_augmentation.cpp
    src/lm_head.cpp
    # Add training components
    src/training/loss_tracker.cpp
    src/training/gradient_manager.cpp
    src/training/learning_rate_scheduler.cpp
    src/training/training_state_manager.cpp
    src/training/training_monitor.cpp
    src/phrase_analysis.cpp
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
    src/cache.cpp
    src/quantization.cpp
    src/optimizer.cpp
)

# Language model components
set(LM_SOURCES
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

# Add CUDA compilation flags
if(CUDAToolkit_FOUND)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -arch=sm_60)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

# Create library with all components
add_library(transformer_lib
    ${CORE_SOURCES}
    ${NN_SOURCES}
    ${UTIL_SOURCES}
    ${LM_SOURCES}
    ${OPTIMIZER_SOURCES}
    ${SOURCES}
)

# Set CUDA specific properties for the library
set_target_properties(transformer_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Add OpenMP configuration if found
if(OpenMP_CXX_FOUND)
    target_compile_definitions(transformer_lib PUBLIC USE_OPENMP)
    target_compile_options(transformer_lib PRIVATE ${OpenMP_CXX_FLAGS})
    if(NOT MSVC)
        target_compile_options(transformer_lib PRIVATE -fopenmp)
    endif()
    target_link_libraries(transformer_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Create single executable
add_executable(transformer src/main.cpp)

# Link everything to the executable
target_link_libraries(transformer
    PRIVATE
    transformer_lib
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

# Link OpenMP if found
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
        src/cuda/beam_search_ops.cu
        src/cuda/tokenizer_kernels.cu
    )

    set_target_properties(cuda_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LANGUAGE CUDA
    )

    # Add CUDA include directories
    target_include_directories(cuda_kernels
        PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party
    )

    # Link CUDA libraries to cuda_kernels
    target_link_libraries(cuda_kernels
        PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )

    # Link cuda_kernels to transformer_lib
    target_link_libraries(transformer_lib
        PUBLIC
        cuda_kernels
    )
endif()

# Add SOURCES to the library instead
target_sources(transformer_lib 
    PRIVATE 
    ${SOURCES}
)

find_package(nlohmann_json 3.2.0 REQUIRED)
target_link_libraries(transformer_lib PRIVATE nlohmann_json::nlohmann_json)

# Make sure the include directory is in the include path
target_include_directories(transformer_lib 
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

add_definitions(
    -DPAD_TOKEN_ID=0
    -DUNK_TOKEN_ID=1
    -DBOS_TOKEN_ID=2
    -DEOS_TOKEN_ID=3
    -DMASK_TOKEN_ID=4
)

# Simplified tiktoken data copying
add_custom_command(TARGET transformer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/tiktoken_data"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/third_party/tiktoken/tiktoken/tiktoken_data"
        "${CMAKE_BINARY_DIR}/tiktoken_data"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/third_party/tiktoken/tiktoken/cl100k_base.vocab.json"
        "${CMAKE_SOURCE_DIR}/third_party/tiktoken/tiktoken/cl100k_base.merges.json"
        "${CMAKE_BINARY_DIR}/tiktoken_data/"
    COMMENT "Copying tiktoken data files to build directory"
)

if(CUDAToolkit_FOUND)
    # Set source file properties to compile with CUDA
    set_source_files_properties(
        src/lm_head.cpp
        PROPERTIES
        LANGUAGE CUDA
    )
endif()
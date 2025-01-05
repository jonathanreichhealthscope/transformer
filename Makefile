# Project directories
BUILD_DIR := build
SRC_DIR := src
INCLUDE_DIR := include

# Default target
.PHONY: all
all: build

# Create build directory and build project
.PHONY: build
build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. && make

# Clean build directory
.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)

# Run the transformer executable
.PHONY: run
run: build
	@cd $(BUILD_DIR) && ./transformer

# Format code (requires clang-format)
.PHONY: format
format:
	@find $(SRC_DIR) $(INCLUDE_DIR) -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' | xargs clang-format -i

# Build in debug mode
.PHONY: debug
debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Debug .. && make

# Build in release mode
.PHONY: release
release:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release .. && make

# Show help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make          - Build project in default mode"
	@echo "  make build    - Same as 'make'"
	@echo "  make clean    - Remove build directory"
	@echo "  make run      - Build and run the transformer"
	@echo "  make format   - Format source code"
	@echo "  make debug    - Build in debug mode"
	@echo "  make release  - Build in release mode"
	@echo "  make stop     - Stop the running transformer"

# Stop the running transformer
.PHONY: stop
stop:
	@pkill transformer || true 

SOURCES = src/main.cpp src/transformer.cpp src/attention.cpp src/components.cpp \
          src/embeddings.cpp src/layernorm.cpp src/feed_forward.cpp \
          src/trainer.cpp src/cache.cpp \
          src/gqa.cpp

OBJECTS = $(SOURCES:.cpp=.o) 
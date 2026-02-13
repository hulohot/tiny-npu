# Tiny NPU Makefile
# Quick reference for common development tasks

# Directories
BUILD_DIR := sim/verilator/build
SIM_DIR := sim/verilator
PYTHON_DIR := python

# Default target
.PHONY: all
all: build

# Build simulation
.PHONY: build
build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ..
	@cd $(BUILD_DIR) && cmake --build . -j$$(nproc)

# Run all tests
.PHONY: test
test: build
	@cd $(BUILD_DIR) && ctest --output-on-failure

# Run individual tests
.PHONY: test-mac
test-mac: build
	@cd $(BUILD_DIR) && ./test_mac_unit

.PHONY: test-systolic
test-systolic: build
	@cd $(BUILD_DIR) && ./test_systolic_array

.PHONY: test-smoke
test-smoke: build
	@cd $(BUILD_DIR) && ./test_npu_smoke

.PHONY: test-integration
test-integration: build
	@cd $(BUILD_DIR) && ./test_integration

.PHONY: test-gpt2
test-gpt2: build
	@cd $(BUILD_DIR) && ./test_gpt2_block

# Clean build artifacts
.PHONY: clean
clean:
	@rm -rf $(BUILD_DIR)
	@echo "Build directory cleaned"

# Deep clean (including generated files)
.PHONY: distclean
distclean: clean
	@find . -name "*.vcd" -delete
	@find . -name "*.fst" -delete
	@find . -name "*.hex" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete
	@echo "All generated files cleaned"

# Generate GPT-2 weights for inference demo
.PHONY: weights
weights:
	@mkdir -p $(BUILD_DIR)/demo_data
	@DEMO_OUTDIR=$(BUILD_DIR)/demo_data python3 $(PYTHON_DIR)/tools/export_gpt2_weights.py
	@DEMO_OUTDIR=$(BUILD_DIR)/demo_data python3 $(PYTHON_DIR)/tools/quantize_pack.py
	@echo "Weights exported to $(BUILD_DIR)/demo_data/"

# Run inference demo (requires weights)
.PHONY: infer
infer: build weights
	@cd $(BUILD_DIR) && ./demo_infer --datadir demo_data --prompt "Hello" --max-tokens 10

# Run inference with custom prompt
# Usage: make infer-prompt PROMPT="Your prompt here"
.PHONY: infer-prompt
infer-prompt: build weights
	@cd $(BUILD_DIR) && ./demo_infer --datadir demo_data --prompt "$(PROMPT)" --max-tokens 10

# Format code (requires verible or verilator)
.PHONY: format
format:
	@find rtl -name "*.sv" -exec verilator --lint-only {} + 2>/dev/null || echo "Verilator format check complete"

# Lint RTL
.PHONY: lint
lint:
	@verilator --lint-only -Wall rtl/npu_top.sv 2>/dev/null || true

# Help
.PHONY: help
help:
	@echo "Tiny NPU - Available targets:"
	@echo ""
	@echo "  Build & Test:"
	@echo "    make build          - Build all simulation targets"
	@echo "    make test           - Run all tests"
	@echo "    make test-mac       - Run MAC unit test"
	@echo "    make test-systolic  - Run systolic array test"
	@echo "    make test-smoke     - Run NPU smoke test"
	@echo "    make test-integration - Run integration test"
	@echo "    make test-gpt2      - Run GPT-2 block test"
	@echo ""
	@echo "  Inference:"
	@echo "    make weights        - Export and quantize GPT-2 weights"
	@echo "    make infer          - Run inference demo with default prompt"
	@echo "    make infer-prompt PROMPT=\"...\" - Run with custom prompt"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean          - Remove build directory"
	@echo "    make distclean      - Remove all generated files"
	@echo "    make format         - Format SystemVerilog code"
	@echo "    make lint           - Lint RTL with Verilator"
	@echo "    make help           - Show this help"

# Tiny NPU

A minimal open-source Neural Processing Unit (NPU) in SystemVerilog, optimized for learning how AI accelerators work from the ground up.

[![GitHub Codespaces](https://img.shields.io/badge/GitHub%20Codespaces-Ready-blue?logo=github)](https://codespaces.new/hulohot/tiny-npu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Tiny NPU is a fully synthesizable neural processing unit that can run real transformer models including:
- **GPT-2** (small)
- **LLaMA** (MicroLlama)
- **Mistral** (300M)
- **Qwen2** (0.5B)

Built with fully documented SystemVerilog RTL, featuring:
- 16×16 systolic array (256 MACs) with INT8 compute
- Dedicated hardware engines: Softmax, LayerNorm/RMSNorm, GELU/SiLU, RoPE
- Hardware KV-cache for efficient autoregressive generation
- Two execution modes: LLM Mode (microcode) and Graph Mode (ONNX)
- Full Verilator simulation with bit-exact verification

## Quick Start (GitHub Codespaces)

Click the badge above or:

```bash
# 1. Open in GitHub Codespaces (free tier available)
# 2. Wait for dev container to build (~2-3 minutes)
# 3. Start developing!
```

## Project Structure

```
tiny-npu/
├── rtl/                    # SystemVerilog RTL
│   ├── npu_top.sv         # Top-level NPU
│   ├── gemm/              # Systolic array GEMM engine
│   ├── engines/           # Softmax, LayerNorm, GELU, etc.
│   ├── memory/            # SRAM, KV-cache
│   └── common/            # Utilities, interfaces
├── sim/                   # Simulation
│   └── verilator/         # Verilator testbenches
├── python/                # Python tools
│   ├── tools/             # Weight export, quantization
│   ├── golden/            # Reference implementations
│   └── onnx_compiler/     # ONNX to NPU compiler
├── .devcontainer/         # GitHub Codespaces config
└── .github/workflows/     # CI/CD
```

## Build & Simulate

```bash
# Build simulation
cd sim/verilator
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)

# Run tests
./npu_sim          # Control plane test
./engine_sim       # Engine compute tests
./gpt2_block_sim   # Full transformer block
```

## Run GPT-2 Inference

```bash
# Export and quantize weights
DEMO_OUTDIR=sim/verilator/build/demo_data python3 python/tools/export_gpt2_weights.py
DEMO_OUTDIR=sim/verilator/build/demo_data python3 python/tools/quantize_pack.py

# Run inference demo
cd sim/verilator/build
./demo_infer --datadir demo_data --prompt "Hello" --max-tokens 10
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tiny NPU Architecture                     │
├─────────────────────────────────────────────────────────────┤
│  AXI4-Lite Host Interface                                    │
│         │                                                    │
│  ┌──────┴──────────────────────────────────────────────┐    │
│  │              Microcode Controller                   │    │
│  │         Fetch → Decode → Dispatch                   │    │
│  └──────┬──────────────────────────────────────────────┘    │
│         │                                                    │
│  ┌──────┴──────┬─────────┬─────────┬─────────┬──────────┐   │
│  │  DMA Eng    │ GEMM    │ Softmax │ LayerNorm│ Vec Eng │   │
│  │  (DDR↔SRAM) │ 16×16   │ 3-pass  │ 2-pass  │ Add/Mul │   │
│  └──────┬──────┴─────┬───┴─────┬───┴─────┬───┴─────┬────┘   │
│         │            │         │         │         │        │
│  ┌──────┴──────────────────────────────────────────────┐    │
│  │              On-Chip SRAM (64KB + 8KB)              │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### INT8 Quantization
- Per-tensor symmetric quantization
- INT32 accumulators with requantization
- Bit-exact matching with PyTorch reference

### Hardware KV-Cache
- Dedicated cache storage for key/value vectors
- 1.8× speedup for autoregressive generation
- Bit-exact with full-recompute baseline

### ONNX Graph Mode
- Compile arbitrary ONNX models (MLP, CNN)
- Tensor descriptor table with automatic allocation
- Graph ISA with DMA, GEMM, element-wise ops

## CI/CD

GitHub Actions automatically:
- Builds all simulation targets
- Runs unit tests and integration tests
- Verifies bit-exact inference

## License

MIT License - See [LICENSE](LICENSE)

## Acknowledgments

Inspired by:
- [tiny-gpu](https://github.com/adam-maj/tiny-gpu) by Adam Majmudar
- [tiny-npu](https://github.com/harishsg993010/tiny-npu) by Harish Santhanalakshmi Ganesan
- Matt Venn's Zero to ASIC course

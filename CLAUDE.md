# CLAUDE.md — AI Assistant Guide for tiny-npu

This file provides context for AI coding assistants (Claude, Copilot, etc.) working in this repository.

---

## Project Overview

**tiny-npu** is a learning-focused SystemVerilog NPU (Neural Processing Unit) prototype designed for transformer-style inference, specifically targeting tiny GPT-2 models. It includes:

- Full RTL for a 16×16 systolic array and five compute engines
- Verilator-based simulation with C++ testbenches
- Python golden reference implementations for bit-exact verification
- A real-weight LLM demo using HuggingFace GPT-2

**Target model:** 64D hidden dim, 4 heads, 256 FFN width, 4 layers, 16 seq_len, INT8 quantized.

---

## Repository Layout

```
tiny-npu/
├── rtl/                          # SystemVerilog RTL (~2,383 lines)
│   ├── npu_top.sv                # Top-level NPU with AXI4-Lite/AXI4 interfaces
│   ├── control/
│   │   └── microcode_controller.sv  # Instruction fetch/decode/dispatch + scoreboard
│   ├── gemm/
│   │   ├── mac_unit.sv           # INT8×8 → INT32 multiply-accumulate
│   │   ├── systolic_array.sv     # 16×16 weight-stationary systolic array
│   │   └── gemm_engine.sv        # Full GEMM with tiling and requantization
│   ├── engines/
│   │   ├── softmax_engine.sv     # 3-pass softmax (exp/sum/normalize) with causal mask
│   │   ├── layernorm_engine.sv   # Layer normalization (mean/variance/scale)
│   │   ├── gelu_engine.sv        # GELU via 256-entry LUT approximation
│   │   └── vec_engine.sv         # Element-wise ops (ADD/MUL/COPY/CLAMP)
│   └── memory/
│       ├── sram_top.sv           # Dual-bank SRAM: 64KB (SRAM0) + 8KB (SRAM1)
│       └── dma_engine.sv         # AXI4 master for DDR↔SRAM transfers
├── sim/verilator/                # Simulation infrastructure
│   ├── CMakeLists.txt            # Verilator build configuration
│   ├── cmake/                    # SRAM init generation helpers
│   └── testbenches/              # C++ testbenches (one per engine + integration)
│       ├── common/npu_utils.h    # Instruction packing, opcodes, shared utilities
│       ├── mac_unit_tb.cpp
│       ├── systolic_array_tb.cpp
│       ├── softmax_engine_tb.cpp
│       ├── layernorm_engine_tb.cpp
│       ├── gelu_engine_tb.cpp
│       ├── vec_engine_tb.cpp
│       ├── npu_tb.cpp            # NPU smoke test
│       ├── integration_tb.cpp    # Multi-engine pipeline test
│       ├── gpt2_block_tb.cpp     # Full GPT-2 block with real weights
│       └── demo_infer.cpp        # Inference demo binary
├── python/                       # Python tools and golden reference (~786 lines)
│   ├── run_tiny_llm_sim.py       # Interactive LLM demo (temperature, top-k, top-p)
│   ├── eval_first_token.py       # First-token agreement vs reference
│   ├── eval_prompt_variation.py  # Prompt-set output diversity check
│   ├── golden/reference.py       # Bit-exact golden implementations (INT8 GEMM, etc.)
│   ├── tests/
│   │   ├── framework.py          # TestCase/TestSuite base classes
│   │   └── test_tiny_llm_smoke.py
│   └── tools/
│       ├── export_gpt2_weights.py  # HuggingFace weight export
│       └── quantize_pack.py        # INT8 quantization packing
├── docs/
│   ├── ARCHITECTURE.md           # ISA, engine specs, test strategy (485 lines)
│   ├── ROADMAP.md                # Contributing priorities and milestones
│   └── CI_BRANCH_PROTECTION.md  # Branch protection setup guide
├── benchmarks/
│   ├── prompts/                  # Prompt sets for evaluation
│   └── results/                  # Generated benchmark outputs and CSVs
├── scripts/
│   ├── benchmark_deterministic.sh  # N-run repeatability checker
│   ├── lint_warning_summary.sh     # Verilator warning aggregator
│   ├── check_fsm_case_coverage.sh  # FSM case coverage verifier
│   └── configure_branch_protection.sh
├── .devcontainer/devcontainer.json  # GitHub Codespaces / Docker config
├── .github/workflows/ci.yml        # GitHub Actions CI pipeline
├── Makefile                         # All development task shortcuts
├── CONTRIBUTING.md                  # PR guidelines and conventions
└── README.md                        # Quick start guide
```

---

## Development Environment

### Prerequisites

- **Ubuntu 22.04** (or Codespaces devcontainer)
- **CMake** 3.14+
- **Verilator** (recent version)
- **GCC/G++** with C++17 support
- **Python** 3.11 with `numpy` installed
- Optional: `transformers`, `torch` (for LLM demo)

### Quick Setup (Codespaces / devcontainer)

The `.devcontainer/devcontainer.json` auto-installs all dependencies. For local setup:

```bash
sudo apt-get install cmake verilator python3-pip build-essential
pip3 install numpy
```

---

## Build System

### Makefile (Primary Interface)

The `Makefile` at the repo root is the primary interface for all development tasks:

```bash
make build                   # CMake configure (if needed) + build
make rebuild                 # Force reconfigure + full rebuild
make clean                   # Remove build artifacts
make distclean               # Remove build/ directory entirely
```

### CMake (Underlying Build)

Build files live under `sim/verilator/build/` (created by CMake). Binaries are placed there after build.

---

## Testing

### Test Suite (9 tests via ctest)

**Unit Tests (6):**
| Make Target | Test Name | What It Covers |
|-------------|-----------|----------------|
| `make test-mac` | `test_mac_unit` | Signed INT8 multiplication, accumulation, clear |
| `make test-systolic` | `test_systolic_array` | 16×16 matrix multiply correctness |
| `make run-softmax` | `test_softmax_engine` | Exp/sum/normalize with causal mask |
| `make run-layernorm` | `test_layernorm_engine` | Mean/variance/scale normalization |
| `make run-gelu` | `test_gelu_engine` | LUT-based GELU approximation |
| `make run-vec` | `test_vec_engine` | Element-wise ADD/MUL/COPY operations |

**Integration Tests (3):**
| Make Target | Test Name | What It Covers |
|-------------|-----------|----------------|
| `make test-smoke` | `test_npu_smoke` | Basic NPU initialization and control |
| `make test-integration` | `test_integration` | Multi-engine pipeline (full transformer block) |
| `make test-gpt2` | `test_gpt2_block` | GPT-2 block with real quantized weights |

### Running Tests

```bash
# All tests (recommended before any PR)
make test

# Individual tests (with rebuild)
make test-mac
make test-systolic
make test-smoke
make test-integration
make test-gpt2

# Individual tests (without rebuild, faster)
make run-mac
make run-systolic
make run-smoke
make run-integration
make run-gpt2

# Deterministic repeatability check (must match across N runs)
make benchmark-deterministic
RUNS=5 make benchmark-deterministic
```

### Python Evaluation Harnesses

```bash
# First-token agreement between reference and sim
python3 -m python.eval_first_token --prepare

# Prompt variation diversity check
python3 -m python.eval_prompt_variation

# Smoke test (auto-skips if model dependencies missing)
python3 -m unittest python/tests/test_tiny_llm_smoke.py

# Interactive LLM demo
python3 -m python.run_tiny_llm_sim --interactive --max-new-tokens 16
```

Results are written to `benchmarks/results/`.

---

## Linting

```bash
make lint              # Run Verilator lint on all RTL files
make lint-summary      # Aggregate Verilator warnings into CSV
make check-fsm-case    # Verify FSM case statement coverage
```

The CI pipeline also runs Verible (downloaded at runtime) for additional verilog linting.

---

## CI/CD Pipeline

Three required checks defined in `.github/workflows/ci.yml`:

1. **stable-regression** — Builds and runs `test_mac_unit`, `test_npu_smoke`, `test_integration`, `test_gpt2_block`
2. **full-ctest** — Builds and runs all 9 tests via `ctest --output-on-failure`
3. **lint** — Verible linting + Verilator warning aggregation

**Triggers:** Push/PR to `main`, push to `develop`, daily at 06:00 UTC, manual dispatch.

**Action pinning:** All third-party GitHub Actions are pinned to immutable commit SHAs (never floating tags like `@v4`).

**Branch protection on `main`:**
- All 3 CI checks must pass
- 1 PR review required, stale reviews dismissed
- Linear history enforced
- No force pushes or branch deletions

---

## ISA Summary

Instructions are **128-bit fixed-width**. Opcodes (from `npu_utils.h`):

| Opcode | Mnemonic | Description |
|--------|----------|-------------|
| `0x0` | `NOP` | No operation |
| `0x1` | `DMA` | DDR↔SRAM transfer |
| `0x2` | `GEMM` | Matrix multiply (systolic array) |
| `0x3` | `SOFTMAX` | Softmax over row |
| `0x4` | `LAYERNORM` | Layer normalization |
| `0x5` | `GELU` | GELU activation |
| `0x6` | `VEC_ADD` | Element-wise addition |
| `0x7` | `VEC_MUL` | Element-wise multiply |
| `0x8` | `VEC_COPY` | Memory copy |
| `0x9` | `BARRIER` | Pipeline sync |
| `0xA` | `END` | Program halt |

Full ISA encoding details are in `docs/ARCHITECTURE.md`.

---

## Memory Map

| Bank | Size | Usage |
|------|------|-------|
| SRAM0 | 64 KB | Weights, activations, attention matrices |
| SRAM1 | 8 KB | Betas, residuals, KV-cache scratch |

---

## Code Conventions

### SystemVerilog RTL

- Use `logic` for all signals (never bare `wire` or `reg`)
- Sequential: `always_ff @(posedge clk or negedge rst_n)` with active-low reset
- Combinational: `always_comb`
- FSM: `typedef enum logic [N:0] { ... } state_t;` with explicit state names
- Parameter naming: `UPPER_SNAKE_CASE` (e.g., `DATA_WIDTH`, `ARRAY_SIZE`)
- Signal naming: `snake_case`
- AXI4 port prefix: `s_axi_*` for slave, `m_axi_*` for master
- Timescale: `` `timescale 1ns/1ps `` on all files
- Include module-level header comments explaining purpose, ports, and algorithm

**Example FSM pattern:**
```systemverilog
typedef enum logic [1:0] {
    IDLE   = 2'b00,
    ACTIVE = 2'b01,
    DONE   = 2'b10
} state_t;

state_t state, next_state;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= IDLE;
    else        state <= next_state;
end
```

### C++ Testbenches

- Use Verilator's generated `V<module>` class
- Enable waveform traces via `VerilatedVcdC` (`.vcd` output)
- Use `npu_utils.h` for instruction packing — never hand-encode instruction bits
- Test vectors must be deterministic (no random seeds in verification logic)
- Print clear PASS/FAIL per test case with expected vs actual values

### Python

- Type hints on all function signatures
- NumPy arrays for all numerical operations
- Docstrings on all public functions
- Module-level imports; use `try/except ImportError` for optional dependencies
- Prefer `pathlib.Path` over `os.path` for file operations

### Commit Style (Conventional Commits)

```
feat: add causal mask support to softmax engine
fix: correct requantization overflow in gemm_engine
test: add vec_engine boundary condition cases
docs: update ISA table with BARRIER semantics
chore: pin CI action to SHA for reproducibility
```

---

## Pre-PR Checklist

Before opening a pull request, run:

```bash
make rebuild                  # Full clean build
make test                     # All 9 tests must pass
make benchmark-deterministic  # Outputs must be stable across runs
make lint                     # Zero new lint warnings
make check-fsm-case           # All FSM cases covered
```

Reference `CONTRIBUTING.md` for full PR requirements including what to write in the PR description.

---

## Key Design Principles

1. **Testability first** — Every engine has a standalone testbench; no RTL change without a passing test
2. **Determinism** — Simulation outputs must be bit-identical across runs; randomness is forbidden in verification
3. **Modularity** — Engines are independently instantiable with their own `valid`/`ready` handshake
4. **Simplicity** — Prefer simple, readable RTL over clever optimizations; this is a learning project
5. **Golden reference** — Python `reference.py` defines the numerically correct behavior; RTL must match it

---

## Current Limitations (as of 2026-03)

- Full token generation path is not yet wired in RTL; first-token only
- Multi-token autoregressive decode and KV-cache not fully implemented in hardware
- Simulated output uses INT8 projection-only emulation

See `docs/ROADMAP.md` for planned work and `docs/ARCHITECTURE.md` for full technical specs.

---

## Useful References

| File | Description |
|------|-------------|
| `docs/ARCHITECTURE.md` | Complete ISA, engine algorithms, test strategy |
| `docs/ROADMAP.md` | Milestone status, priority work items |
| `Makefile` | All available development commands |
| `CONTRIBUTING.md` | PR guidelines, code review process |
| `sim/verilator/testbenches/common/npu_utils.h` | Instruction encoding helpers |
| `python/golden/reference.py` | Authoritative golden implementations |

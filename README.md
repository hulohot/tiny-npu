# Tiny NPU

A learning-focused SystemVerilog NPU prototype with Verilator-based regression tests.

[![GitHub Codespaces](https://img.shields.io/badge/GitHub%20Codespaces-Ready-blue?logo=github)](https://codespaces.new/hulohot/tiny-npu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What is implemented today

Validated by CI/tests in `sim/verilator`:
- RTL modules for MAC, systolic array, controller/memory/engines scaffolding
- Verilator regression binaries:
  - `test_mac_unit`
  - `test_systolic_array`
  - `test_npu_smoke`
  - `test_integration`
  - `test_gpt2_block`
- Deterministic output harness (`make benchmark-deterministic`)

## What is roadmap (not yet production-complete)

- Full end-to-end LLM inference fidelity/performance
- Complete microcode/engine feature parity with architecture spec
- FPGA timing/resource closure and hardware bring-up
- Expanded lint/warning cleanup across all RTL modules

See `docs/ARCHITECTURE.md` and roadmap issues for details.

## Quick Start

```bash
# Build simulation
cmake -S sim/verilator -B sim/verilator/build
cmake --build sim/verilator/build -j$(nproc)

# Run all tests
ctest --test-dir sim/verilator/build --output-on-failure

# Deterministic baseline harness
make benchmark-deterministic
```

## CI

GitHub Actions runs three checks:
- `stable-regression`
- `full-ctest`
- `lint`

Branch protection hardening guide: `docs/CI_BRANCH_PROTECTION.md`

## Repository layout

```
tiny-npu/
├── rtl/                    # SystemVerilog RTL
├── sim/verilator/          # Verilator testbenches + CMake
├── python/                 # Model/data helper tooling
├── docs/                   # Architecture + process docs
├── benchmarks/             # Deterministic benchmark harness + baseline
└── .github/workflows/      # CI
```

## Minimal real-weights LLM demo (interactive + first-token compare)

This repository includes a minimal end-to-end path that uses **real HuggingFace GPT-2-family weights** (default: `sshleifer/tiny-gpt2`) and reports:
- **reference generation** from full HF model
- **simulated token** from an INT8 projection-only path (real hidden-state + real `lm_head`, first-token only)

### 1) Prepare artifacts in `demo_data`

```bash
python -m python.run_tiny_llm_sim --prepare --prompt "hello"
```

This runs export + quantization/packing. Pack assumptions are recorded in `demo_data/quant_manifest.json`.

### 2) One-shot run (JSON output)

```bash
python -m python.run_tiny_llm_sim \
  --prompt "Hello tiny NPU" \
  --max-new-tokens 16 \
  --temperature 0.9 --top-k 40 --top-p 0.95 --seed 42
```

Optional smoke check integration (if Verilator build exists):

```bash
python -m python.run_tiny_llm_sim --prompt "Hello tiny NPU" --run-verilator-smoke
```

### 3) Interactive mode

```bash
python -m python.run_tiny_llm_sim --interactive --max-new-tokens 16 --temperature 0.9 --top-k 40 --top-p 0.95 --seed 42
```

### 4) Smoke/regression check

```bash
python -m unittest python/tests/test_tiny_llm_smoke.py
```

> Note: this smoke test auto-skips when model dependencies/download are unavailable in the environment.

### Current limitations

- RTL path is **not yet wired** for full GPT-2 token generation.
- `simulated` remains first-token projection-only INT8 emulation, not full block-by-block RTL execution.
- Multi-token autoregressive decode and KV-cache handling are not yet implemented in hardware flow.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

## License

MIT License - See [LICENSE](LICENSE)

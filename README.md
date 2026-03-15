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

# Deterministic baseline harness (defaults to RUNS=3)
make benchmark-deterministic

# Override repeat count (N runs + hash compare)
RUNS=5 make benchmark-deterministic
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

## Minimal real-weights LLM demo (interactive + reference vs simulated decode)

This repository includes a minimal end-to-end path that uses **real HuggingFace GPT-2-family weights** (default: `sshleifer/tiny-gpt2`) and reports:
- **reference generation** from full HF model
- **simulated generation** from an INT8 projection decode path (multi-token software approximation using real hidden states + real `lm_head`)

### 1) Install runtime dependencies (one-time)

```bash
bash scripts/setup_llm_env.sh
source .venv/bin/activate
```

(Or manually: `python -m pip install -r requirements-llm.txt`.)

### 2) Prepare artifacts in `demo_data`

```bash
python -m python.run_tiny_llm_sim --prepare --prompt "hello"
```

This runs export + quantization/packing. Pack assumptions are recorded in `demo_data/quant_manifest.json`.

### 3) One-shot run (JSON output)

```bash
python -m python.run_tiny_llm_sim \
  --prompt "Hello tiny NPU" \
  --max-new-tokens 16 \
  --temperature 0.9 --top-k 40 --top-p 0.95 --seed 42 \
  --sim-max-new-tokens 16 \
  --sim-temperature 0.0 --sim-top-k 0 --sim-top-p 1.0 --sim-seed 123
```

Optional smoke check integration (if Verilator build exists):

```bash
python -m python.run_tiny_llm_sim --prompt "Hello tiny NPU" --run-verilator-smoke
```

### 4) Interactive mode

```bash
python -m python.run_tiny_llm_sim \
  --interactive \
  --max-new-tokens 16 --temperature 0.9 --top-k 40 --top-p 0.95 --seed 42 \
  --sim-max-new-tokens 16 --sim-temperature 0.0 --sim-top-k 0 --sim-top-p 1.0 --sim-seed 123
```

### 5) Smoke/regression check

```bash
python -m unittest python/tests/test_tiny_llm_smoke.py
```

> Note: this smoke test auto-skips when model dependencies/download are unavailable in the environment.

### 6) First-token evaluation harness (reference vs simulated)

```bash
python3 -m python.eval_first_token --prepare
# writes:
#   benchmarks/results/first_token_eval/first_token_eval.csv
#   benchmarks/results/first_token_eval/summary.json
```

This gives you a prompt-set match rate so improvements can be measured over time.

### 7) Prompt-set variation check (interactive quality)

```bash
python3 -m python.eval_prompt_variation
# writes:
#   benchmarks/results/prompt_variation/prompt_variation.csv
#   benchmarks/results/prompt_variation/summary.json
```

This reports unique first-token count and variation ratio across a prompt set.

### Current limitations

- RTL path is **not yet wired** for full GPT-2 token generation.
- `simulated` uses INT8 projection decode with hidden states from the full HF model at each step; this is **not** full block-by-block RTL execution.
- KV-cache behavior and true hardware-timed autoregressive decode are not yet implemented in the RTL path.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

## License

MIT License - See [LICENSE](LICENSE)

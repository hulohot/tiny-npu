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

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

## License

MIT License - See [LICENSE](LICENSE)

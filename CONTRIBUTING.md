# Contributing to Tiny NPU

Thanks for contributing.

## Development setup

```bash
git clone https://github.com/hulohot/tiny-npu.git
cd tiny-npu
cmake -S sim/verilator -B sim/verilator/build
cmake --build sim/verilator/build -j"$(nproc)"
```

## Run tests before opening a PR

```bash
cd sim/verilator/build
./test_mac_unit
./test_npu_smoke
./test_integration
./test_gpt2_block
ctest --output-on-failure -E "Systolic_Array"
```

> Note: `Systolic_Array` is currently tracked as a known failing test in CI while it is being debugged.

## Coding guidelines

- Keep modules synthesizable unless explicitly test-only.
- Prefer small PRs with one clear purpose.
- Add or update tests for logic changes.
- Document architectural changes in `docs/ARCHITECTURE.md`.

## Commit style

Use conventional commit prefixes when possible:
- `feat:` new functionality
- `fix:` bug fix
- `test:` test updates
- `docs:` documentation
- `chore:` maintenance

## Pull requests

Include in every PR:
- What changed
- Why it changed
- How it was tested (exact commands)
- Any known limitations or follow-up work

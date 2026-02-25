# Contributing to Tiny NPU

Thanks for contributing.

## Development setup

```bash
git clone https://github.com/hulohot/tiny-npu.git
cd tiny-npu
cmake -S sim/verilator -B sim/verilator/build
cmake --build sim/verilator/build -j"$(nproc)"
```

## Required pre-PR checks

```bash
make rebuild
make test
make benchmark-deterministic
```

Recommended:
```bash
verilator --lint-only -Wall $(find rtl -name '*.sv' | tr '\n' ' ')
```

## CI checks on PRs

PRs into `main` are expected to pass:
- `stable-regression`
- `full-ctest`
- `lint`

Branch protection setup: `docs/CI_BRANCH_PROTECTION.md`

## Coding guidelines

- Keep modules synthesizable unless explicitly test-only.
- Prefer small PRs with one clear purpose.
- Add or update tests for logic changes.
- Document architecture/process changes in `docs/`.

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

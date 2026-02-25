# Deterministic Benchmark Harness

Run deterministic benchmark/output checks for core regression binaries:

```bash
make benchmark-deterministic
```

Outputs:
- `benchmarks/results/*.out` raw stdout/stderr captures
- `benchmarks/results/deterministic_summary.csv` SHA256 digest per test output

Use this to catch accidental non-determinism in simulation-facing behavior.

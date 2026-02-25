# Deterministic Baseline

Generated on branch `quality-drive` using:

```bash
make rebuild
make test
make benchmark-deterministic
```

Current expected output hashes (`benchmarks/results/deterministic_summary.csv`):

- `test_mac_unit`: `1d5c5c4870c7acae25bc9a05c514895ae03193c695a27fb92d1c98482433f9aa`
- `test_systolic_array`: `e8f6d7a4f59f2d9884dc998a90f68662810fb39dbd6279bfab926cb6e8ede509`
- `test_npu_smoke`: `24df99b0b4144a22819a8878de9e2fa553c114387122b152129c2fdcb6767b6a`
- `test_integration`: `907dd837af07f6e6b3cf681f4150b7f1e580cf736cb3e7d848ead6d603a2db97`
- `test_gpt2_block`: `f564f439542ef5681035a21c5b8caab6f47e222f45f35687b002b7a12f6b3a9c`

If outputs change, update this file in the same PR and explain why.

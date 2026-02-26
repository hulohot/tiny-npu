# Tiny NPU Contributor Roadmap

Prioritized issue queue (quality drive).

## P0 (next)
1. #4 RTL warnings: eliminate remaining WIDTHTRUNC/WIDTHEXPAND in gemm_engine
2. #10 Verification: add dedicated unit tests for softmax/layernorm/gelu/vec engines

## P1 (high)
3. #2 CI: add scheduled nightly full regression + artifact retention
4. #3 CI: pin third-party action versions to immutable SHAs
5. #5 RTL warnings: complete case/default coverage in all FSM modules
6. #6 RTL hygiene: resolve multi-driver and dead-signal warnings in top-level wiring
7. #7 Benchmarking: add repeatability check target (N runs + hash compare)
8. #9 Docs: split ARCHITECTURE into Implemented vs Target sections by module
9. #13 Roadmap: milestone plan for warning debt burn-down (P0/P1/P2)

## P2 (medium)
10. #8 Benchmarking: publish per-test runtime/cycle baseline table
11. #11 Tooling: add `make lint-verilator` target with warning summary report
12. #12 Process: add PR checklist for CI checks, deterministic baseline, and docs updates

## Milestone #15 progress snapshot (2026-02-26, refreshed)

Milestone issue: #15 (RTL reliability + verification depth)

- ✅ Core prerequisites completed: #4, #6, #10
- ✅ Follow-up hardening completed: #2, #3, #5, #7, #11, #25
- ✅ CI gates active and green on main (`stable-regression`, `full-ctest`, `lint`)

At this point, milestone #15 criteria are satisfied and the issue can be closed after this roadmap update is merged.

## Working agreement
- Keep PRs scoped (1–3 related issues each).
- Keep `stable-regression`, `full-ctest`, and `lint` green before merge.
- Update benchmark baseline docs when outputs legitimately change.

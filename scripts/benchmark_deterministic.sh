#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/sim/verilator/build"
OUT_DIR="${ROOT_DIR}/benchmarks/results"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${BUILD_DIR}/test_mac_unit" ]]; then
  echo "Build artifacts not found. Run: make rebuild"
  exit 1
fi

runs=(
  test_mac_unit
  test_systolic_array
  test_npu_smoke
  test_integration
  test_gpt2_block
)

summary_csv="${OUT_DIR}/deterministic_summary.csv"
echo "test,sha256" > "${summary_csv}"

for t in "${runs[@]}"; do
  outfile="${OUT_DIR}/${t}.out"
  (cd "${BUILD_DIR}" && "./${t}") > "${outfile}" 2>&1
  hash="$(sha256sum "${outfile}" | awk '{print $1}')"
  echo "${t},${hash}" >> "${summary_csv}"
  echo "${t}: ${hash}"
done

echo "Wrote ${summary_csv}"

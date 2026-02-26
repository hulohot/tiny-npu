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

RUNS="${RUNS:-3}"
if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || [[ "${RUNS}" -lt 1 ]]; then
  echo "RUNS must be a positive integer (got: ${RUNS})"
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
echo "test,run,sha256" > "${summary_csv}"

overall_status=0
for t in "${runs[@]}"; do
  ref_hash=""
  mismatch=0

  for i in $(seq 1 "${RUNS}"); do
    outfile="${OUT_DIR}/${t}.run${i}.out"
    (cd "${BUILD_DIR}" && "./${t}") > "${outfile}" 2>&1
    hash="$(sha256sum "${outfile}" | awk '{print $1}')"
    echo "${t},${i},${hash}" >> "${summary_csv}"

    if [[ -z "${ref_hash}" ]]; then
      ref_hash="${hash}"
      echo "${t}: run ${i}/${RUNS} hash=${hash} (baseline)"
    elif [[ "${hash}" != "${ref_hash}" ]]; then
      mismatch=1
      overall_status=1
      echo "${t}: run ${i}/${RUNS} hash=${hash} (MISMATCH vs ${ref_hash})"
    else
      echo "${t}: run ${i}/${RUNS} hash=${hash} (match)"
    fi
  done

  if [[ "${mismatch}" -eq 0 ]]; then
    echo "${t}: PASS (${RUNS}/${RUNS} hashes matched)"
  else
    echo "${t}: FAIL (hash mismatch detected across ${RUNS} runs)"
  fi
  echo
 done

echo "Wrote ${summary_csv}"
if [[ "${overall_status}" -eq 0 ]]; then
  echo "Deterministic benchmark repeatability: PASS"
else
  echo "Deterministic benchmark repeatability: FAIL"
fi

exit "${overall_status}"

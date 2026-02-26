#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/benchmarks/results"
mkdir -p "${OUT_DIR}"

REPORT_TXT="${OUT_DIR}/verilator_warnings.txt"
SUMMARY_CSV="${OUT_DIR}/verilator_warning_summary.csv"

TMP_LOG="$(mktemp)"
trap 'rm -f "${TMP_LOG}"' EXIT

# Capture warnings without turning them into fatal errors.
set +e
find "${ROOT_DIR}/rtl" -name '*.sv' -print0 \
  | xargs -0 verilator --lint-only -Wall -Wno-fatal >"${TMP_LOG}" 2>&1
verilator_rc=$?
set -e

cp "${TMP_LOG}" "${REPORT_TXT}"

warning_lines=$(grep -E '^%Warning-[A-Z0-9_]+' "${TMP_LOG}" || true)
if [[ -z "${warning_lines}" ]]; then
  echo "warning_class,count" > "${SUMMARY_CSV}"
  echo "TOTAL,0" >> "${SUMMARY_CSV}"
  echo "No Verilator warnings found"
  echo "Wrote ${SUMMARY_CSV}"
  exit 0
fi

classes=$(echo "${warning_lines}" | sed -E 's/^%Warning-([A-Z0-9_]+):.*/\1/' | sort)

echo "warning_class,count" > "${SUMMARY_CSV}"
echo "${classes}" | uniq -c | awk '{print $2 "," $1}' >> "${SUMMARY_CSV}"
total=$(echo "${warning_lines}" | wc -l)
echo "TOTAL,${total}" >> "${SUMMARY_CSV}"

cat "${SUMMARY_CSV}"
echo "Wrote ${SUMMARY_CSV}"
echo "Raw warnings: ${REPORT_TXT}"

# Preserve non-zero only for tool invocation errors, not warnings.
if [[ "${verilator_rc}" -ne 0 ]] && ! grep -q '^%Warning-' "${TMP_LOG}"; then
  exit "${verilator_rc}"
fi

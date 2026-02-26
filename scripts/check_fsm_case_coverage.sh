#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Checking FSM case/default coverage (CASEINCOMPLETE/CASEOVERLAP)..."

find "${ROOT_DIR}/rtl" -name '*.sv' -print0 \
  | xargs -0 verilator --lint-only -Wall -Wno-fatal --Werror-CASEINCOMPLETE --Werror-CASEOVERLAP >/tmp/fsm_case_lint.log 2>&1 || {
    cat /tmp/fsm_case_lint.log
    echo "FSM case coverage check: FAIL"
    exit 1
  }

cat /tmp/fsm_case_lint.log
echo "FSM case coverage check: PASS"

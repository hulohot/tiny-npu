#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python binary '$PYTHON_BIN' not found" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-llm.txt

echo "LLM environment ready in $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"

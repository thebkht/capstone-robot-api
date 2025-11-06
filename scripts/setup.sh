#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON:-python3}"

if [[ ! -x "$(command -v ${PYTHON_BIN})" ]]; then
  echo "Python interpreter '${PYTHON_BIN}' not found" >&2
  exit 1
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[setup] creating virtual environment at ${VENV_PATH}"
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
else
  echo "[setup] reusing existing virtual environment at ${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

pip install --upgrade pip setuptools wheel
pip install -e "${PROJECT_ROOT}[dev]"

echo "[setup] environment ready"

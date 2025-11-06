#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
UVICORN_BIN="${UVICORN_BIN:-uvicorn}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

cd "${PROJECT_ROOT}"

exec "${UVICORN_BIN}" app.main:app --host "${HOST}" --port "${PORT}"


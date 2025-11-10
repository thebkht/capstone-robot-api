#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
UVICORN_BIN="${UVICORN_BIN:-uvicorn}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
LOG_CONFIG="${LOG_CONFIG:-${PROJECT_ROOT}/scripts/logging.ini}"

ensure_tailscale() {
  if ! command -v tailscale >/dev/null 2>&1; then
    echo "tailscale binary not found; skipping remote access configuration" >&2
    return
  fi

  echo "Configuring tailscale remote access" >&2

  # 1) make sure we're up and logged in
  # remove --advertise-exit-node if you don't need it
  if ! sudo -n tailscale up --ssh --accept-dns=false; then
    echo "tailscale up failed (continuing without remote access); ensure sudo is passwordless or run as root" >&2
    return
  fi

  # 2) create the serve config inside the tailnet
  if ! sudo -n tailscale serve http 8000; then
    echo "tailscale serve setup failed (continuing without remote access)" >&2
    return
  fi

  # 3) make it public
  if ! sudo -n tailscale funnel 8000; then
    echo "tailscale funnel setup failed (continuing without public access); check tailnet funnel settings" >&2
  fi
}

# activate venv if present
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

# do tailscale first (before starting uvicorn)
ensure_tailscale

cd "${PROJECT_ROOT}"

LOGGING_ARGS=(--host "${HOST}" --port "${PORT}")

if [[ -n "${LOG_CONFIG}" && -f "${LOG_CONFIG}" ]]; then
  LOGGING_ARGS+=(--log-config "${LOG_CONFIG}")
elif [[ -n "${LOG_LEVEL}" ]]; then
  LOGGING_ARGS+=(--log-level "${LOG_LEVEL}")
fi

# IMPORTANT: exec should be LAST
exec "${UVICORN_BIN}" app.main:app "${LOGGING_ARGS[@]}"

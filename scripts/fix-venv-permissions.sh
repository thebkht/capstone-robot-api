#!/usr/bin/env bash
# Script to fix virtual environment permissions and install missing packages
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

echo "Fixing virtual environment permissions..."
sudo chown -R jetson:jetson "${VENV_PATH}"

echo "Installing missing camera packages..."
source "${VENV_PATH}/bin/activate"
pip install depthai opencv-python

echo "Done! You can now restart the service:"
echo "  sudo systemctl restart capstone-robot-api.service"


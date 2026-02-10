#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

print_msg() {
  printf '[setup_uv] %s\n' "$1"
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  print_msg "uv not found, installing uv..."
  if ! command -v curl >/dev/null 2>&1; then
    print_msg "curl is required to install uv automatically."
    print_msg "Install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi

  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Common default install path for uv.
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    print_msg "uv installation finished, but uv is still not in PATH."
    print_msg "Please add ~/.local/bin to PATH and re-run this script."
    exit 1
  fi
}

main() {
  print_msg "Project root: $ROOT_DIR"
  ensure_uv

  print_msg "Creating virtual environment (.venv) with Python ${PYTHON_VERSION}..."
  uv venv --python "${PYTHON_VERSION}" .venv

  print_msg "Installing dependencies from requirements.txt..."
  uv pip install --python .venv/bin/python -r requirements.txt

  print_msg "Done."
  print_msg "Activate env: source .venv/bin/activate"
  print_msg "Run inference: CUDA_VISIBLE_DEVICES=0 .venv/bin/python new_infer.py --config-file ./configs/inference/inference_fusion.yaml"
}

main "$@"

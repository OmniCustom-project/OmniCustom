#!/usr/bin/env bash
# set -euo pipefail

PYTHON_BIN="python"
if [ -x "./.venv/bin/python" ]; then
  PYTHON_BIN="./.venv/bin/python"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$PYTHON_BIN" infer.py --config-file ./configs/inference/inference_fusion.yaml

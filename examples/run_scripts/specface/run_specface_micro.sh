#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
CONFIG="${ROOT_DIR}/examples/llm_serving/specface/specface_micro_config.json"
PYTHON_BIN="${ROOT_DIR}/.venv-specface/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python3
fi

"${PYTHON_BIN}" "${ROOT_DIR}/experiments/specface/specface_micro.py" --config "${CONFIG}"

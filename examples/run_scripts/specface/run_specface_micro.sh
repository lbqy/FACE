#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
CONFIG="${ROOT_DIR}/examples/llm_serving/specface/specface_micro_config.json"

python3 "${ROOT_DIR}/experiments/specface/specface_micro.py" --config "${CONFIG}"

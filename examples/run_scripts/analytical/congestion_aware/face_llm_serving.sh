#!/bin/bash
set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
EXAMPLE_DIR=$(realpath "${SCRIPT_DIR}/../../../")
ASTRA_SIM_BIN="${EXAMPLE_DIR}/../build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"

SYSTEM="${EXAMPLE_DIR}/system/native_collectives/Ring_4chunks.json"
REMOTE_MEMORY="${EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json"
NETWORK="${EXAMPLE_DIR}/network/analytical/Ring_16npus.yml"
LLM_CONFIG="${EXAMPLE_DIR}/llm_serving/face/face_wsc_config.json"

"${ASTRA_SIM_BIN}" \
  --workload-mode=llm-serving \
  --workload-configuration=empty \
  --llm-server-configuration="${LLM_CONFIG}" \
  --system-configuration="${SYSTEM}" \
  --network-configuration="${NETWORK}" \
  --remote-memory-configuration="${REMOTE_MEMORY}"

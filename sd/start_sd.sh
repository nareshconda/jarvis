#!/usr/bin/env bash
# start_sd.sh — run Stable Diffusion (AUTOMATIC1111 or Forge) so Jarvis can use it.
#
# Jarvis talks to the A1111-compatible HTTP API, so the web UI MUST be started with:
#   --api      enable the /sdapi/v1/* endpoints Jarvis calls
#   --listen   bind 0.0.0.0 so the Jarvis server can reach it over the LAN
#   --port     must match DEVICEn_SD_URL in Jarvis's .env (default 7860)
#
#   SD_DIR=~/stable-diffusion-webui-forge ./start_sd.sh
set -euo pipefail
SD_DIR="${SD_DIR:-$HOME/stable-diffusion-webui-forge}"
PORT="${SD_PORT:-7860}"

ARGS=(--api --listen --port "$PORT")

case "$(uname -s)" in
  Darwin)  # Apple Silicon (MPS)
    ARGS+=(--skip-torch-cuda-test --upcast-sampling --no-half-vae) ;;
  *)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    if   [ "${VRAM_MB:-0}" -ge 11000 ]; then ARGS+=(--xformers)             # 12 GB+
    elif [ "${VRAM_MB:-0}" -ge 7000  ]; then ARGS+=(--xformers --medvram)   # 8 GB
    else                                     ARGS+=(--lowvram)              # smaller cards
    fi ;;
esac
# Extra flags from the environment, e.g. SD_EXTRA_ARGS="--ckpt-dir /data/models"
# shellcheck disable=SC2206
ARGS+=(${SD_EXTRA_ARGS:-})

cd "$SD_DIR"
echo "Starting Stable Diffusion: ./webui.sh ${ARGS[*]}"
exec ./webui.sh "${ARGS[@]}"

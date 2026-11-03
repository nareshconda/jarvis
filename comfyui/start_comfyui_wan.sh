#!/usr/bin/env bash
# start_comfyui_wan.sh — run standalone ComfyUI (Wan 2.2) on the LAN for JARVIS.
#
# Video engine for NVIDIA cards WITHOUT BF16 (RTX 20xx / Turing / SM75), which
# can't run FreeVideo's MiniMax H3 (that needs BF16 / SM80+). It serves the RAW
# ComfyUI API on 0.0.0.0:8188 for Jarvis (DEVICEn_VIDEO_URL, DEVICEn_VIDEO_ENGINE=wan).
#
#   chmod +x start_comfyui_wan.sh
#   ./start_comfyui_wan.sh
#
# Reach it from JARVIS at http://<THIS_PC_IP>:8188 . Log: ~/ComfyUI/comfyui-lan.log

set -euo pipefail

PORT="${COMFY_PORT:-8188}"
COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"
VENV="${COMFY_VENV:-$COMFY_DIR/venv}"
LOG="$COMFY_DIR/comfyui-lan.log"

# 8 GB cards: --lowvram streams model layers through system RAM so the
# two 14B experts fit. If you still OOM, switch to --novram (slower) or use
# smaller GGUF quants (Q3_K_M) and set WAN_HIGH_GGUF/WAN_LOW_GGUF in JARVIS.
EXTRA_ARGS="${COMFY_EXTRA_ARGS:---lowvram}"

PY="$VENV/bin/python"
[ -x "$PY" ]              || { echo "❌ venv python missing at $PY — create it (see setup steps)."; exit 1; }
[ -f "$COMFY_DIR/main.py" ] || { echo "❌ ComfyUI not found at $COMFY_DIR."; exit 1; }

echo "▶ ComfyUI (Wan 2.2) on 0.0.0.0:$PORT"
echo "  python : $PY"
echo "  comfy  : $COMFY_DIR"
echo "  extra  : $EXTRA_ARGS"
echo "  log    : $LOG"
echo

cd "$COMFY_DIR"
exec "$PY" main.py --listen 0.0.0.0 --port "$PORT" $EXTRA_ARGS 2>&1 | tee "$LOG"

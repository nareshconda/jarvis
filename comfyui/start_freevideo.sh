#!/bin/zsh
# start_freevideo.sh — launch FreeVideo's ComfyUI on the LAN so JARVIS can reach it.
#
#   chmod +x start_freevideo.sh
#   ./start_freevideo.sh
#
# Serves ComfyUI on http://0.0.0.0:8188 (reachable from the Pi at
# http://<THIS_MAC_IP>:8188). Logs to ~/FreeVideo/freevideo-lan.log.

set -e

PORT=8188
COMFY_DIR="$HOME/FreeVideo/ComfyUI"
LOG="$HOME/FreeVideo/freevideo-lan.log"

# Auto-detect FreeVideo's uv-managed ComfyUI env. The folder has a hash
# (comfyui-<hash>) that changes when FreeVideo updates, so glob for it instead
# of hard-coding the path.
ENV_DIR=$(ls -d "$HOME"/FreeVideo/FreeVideo-engine/envs/comfyui-*/ 2>/dev/null | head -1)
if [ -z "$ENV_DIR" ]; then
  echo "❌ Couldn't find FreeVideo's ComfyUI env under ~/FreeVideo/FreeVideo-engine/envs/"
  echo "   Open the FreeVideo app once and let it finish 'Install & launch', then retry."
  exit 1
fi
PY="${ENV_DIR%/}/bin/python"
if [ ! -x "$PY" ]; then
  echo "❌ Python not found/executable at: $PY"
  exit 1
fi

# Use a SEPARATE database file so we don't clash with the FreeVideo app's own
# ComfyUI (that's what caused the "Could not acquire lock on ...comfyui.db").
DB="$COMFY_DIR/user/comfyui-lan.db"

echo "▶ FreeVideo ComfyUI (LAN) on port $PORT"
echo "  python : $PY"
echo "  comfy  : $COMFY_DIR"
echo "  log    : $LOG"
echo

cd "$COMFY_DIR"
"$PY" main.py \
  --listen 0.0.0.0 \
  --port "$PORT" \
  --database-url "sqlite:///$DB" \
  2>&1 | tee "$LOG"

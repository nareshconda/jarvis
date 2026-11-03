#!/usr/bin/env bash
# start_freevideo_lan.sh — expose FreeVideo's ComfyUI on the LAN (headless Linux).
#
# Runs the *raw* ComfyUI that FreeVideo installed, bound to 0.0.0.0:8188, so
# JARVIS can hit it exactly like Stable Diffusion / the Mac FreeVideo.
# We deliberately DO NOT use `freevideo server --listen 0.0.0.0`, because that
# puts a token-protected proxy in front of ComfyUI and JARVIS's plain
# POST /prompt calls would get 401'd.
#
#   chmod +x start_freevideo_lan.sh
#   ./start_freevideo_lan.sh
#
# Serves on http://0.0.0.0:8188  (reach it from JARVIS at http://<THIS_PC_IP>:8188)
# Log: ~/FreeVideo/freevideo-lan.log

set -euo pipefail

PORT="${FREEVIDEO_PORT:-8188}"
ROOT="${FREEVIDEO_ROOT:-$HOME/FreeVideo}"
LOG="$ROOT/freevideo-lan.log"

# Extra ComfyUI args. On an 8 GB card (RTX 2060 Super) FreeVideo's own weight
# streaming + int8 usually fits; if you ever hit CUDA OOM, run with:
#   FREEVIDEO_EXTRA_ARGS="--lowvram" ./start_freevideo_lan.sh
EXTRA_ARGS="${FREEVIDEO_EXTRA_ARGS:-}"

echo "▶ FreeVideo ComfyUI (LAN) — looking under: $ROOT"

# --- locate ComfyUI's main.py (layout can vary between FreeVideo versions) ---
COMFY_MAIN=""
for p in \
  "$ROOT/ComfyUI/main.py" \
  "$ROOT/FreeVideo-engine/ComfyUI/main.py"; do
  [ -f "$p" ] && COMFY_MAIN="$p" && break
done
if [ -z "$COMFY_MAIN" ]; then
  COMFY_MAIN="$(find "$ROOT" -maxdepth 5 -name main.py -path '*ComfyUI*' 2>/dev/null | head -1 || true)"
fi
if [ -z "$COMFY_MAIN" ]; then
  echo "❌ Couldn't find ComfyUI's main.py under $ROOT"
  echo "   Run ./setup.sh once (it installs ComfyUI + the models), then retry."
  exit 1
fi
COMFY_DIR="$(dirname "$COMFY_MAIN")"

# --- locate the Python that FreeVideo built its env with -----------------
PY=""
for cand in \
  "$ROOT"/FreeVideo-engine/envs/comfyui-*/bin/python \
  "$ROOT"/FreeVideo-engine/envs/comfyui-*/bin/python3 \
  "$COMFY_DIR"/../.venv/bin/python \
  "$COMFY_DIR"/venv/bin/python \
  "$ROOT"/.venv/bin/python; do
  [ -x "$cand" ] && PY="$cand" && break
done
if [ -z "$PY" ]; then
  PY="$(find "$ROOT" -maxdepth 5 -path '*comfyui-*/bin/python' -type f 2>/dev/null | head -1 || true)"
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "❌ Couldn't find FreeVideo's ComfyUI python under $ROOT"
  echo "   Run ./setup.sh once, then retry (it creates the env)."
  exit 1
fi

# Separate DB so we never clash with FreeVideo's own ComfyUI instance
# (that clash is what throws 'Could not acquire lock on ...comfyui.db').
DB="$COMFY_DIR/user/comfyui-lan.db"
mkdir -p "$COMFY_DIR/user"

echo "  python : $PY"
echo "  comfy  : $COMFY_DIR"
echo "  port   : $PORT   (0.0.0.0)"
echo "  log    : $LOG"
echo "  extra  : ${EXTRA_ARGS:-<none>}"
echo

cd "$COMFY_DIR"
exec "$PY" main.py \
  --listen 0.0.0.0 \
  --port "$PORT" \
  --database-url "sqlite:///$DB" \
  $EXTRA_ARGS \
  2>&1 | tee "$LOG"

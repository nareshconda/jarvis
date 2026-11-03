#!/bin/zsh
# run_jarvis_voice.sh — load env, activate the venv, start the voice listener.
# Lives in ~/jarvis_assistant alongside jarvis_voice_mac.py and the venv.

APP_DIR="$HOME/jarvis_assistant"
VENV="$APP_DIR/venv"

# --- config (edit here) -------------------------------------------------
export JARVIS_URL="http://JARVIS_SERVER_IP:5006/api/voice/phone"   # <- your server
export JARVIS_LLM_MODE="mac"
export JARVIS_WAKE="jarvis,hey jarvis,jaavis,jervis,jawvis,jargis,jarvish,jovis"
export JARVIS_STT_MODEL="small.en"        # small.en = better wake accuracy on the M6
export JARVIS_CMD_WAIT="8"                # seconds to start talking after the chime
export JARVIS_SILENCE_SECS="2"            # pause that ends a command
export JARVIS_MAX_CMD_SECS="20"           # hard cap on command length
export KOKORO_URL="http://127.0.0.1:8880/v1/audio/speech"
export JARVIS_LOG_FILE="$APP_DIR/jarvis_voice.log"
export HF_HOME="$HOME/.cache/huggingface"
# -----------------------------------------------------------------------

cd "$APP_DIR" || exit 1

# Activate the venv (you use: source venv/bin/activate). exec so launchd tracks
# the python process directly and KeepAlive can restart it cleanly.
source "$VENV/bin/activate"
exec python3 "$APP_DIR/jarvis_voice_mac.py"

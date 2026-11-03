#!/usr/bin/env bash
# Install the Jarvis PC agent as a systemd *user* service (starts with your desktop session).
#   ./install.sh <AGENT_TOKEN>      (use the same AGENT_TOKEN as the server's .env)
set -euo pipefail
TOKEN="${1:-}"
[ -n "$TOKEN" ] || { echo "usage: $0 <AGENT_TOKEN from the server's .env>"; exit 1; }
DIR="$HOME/.local/share/jarvis-agent"
mkdir -p "$DIR" "$HOME/.config/systemd/user" "$HOME/.config/jarvis"
cp "$(dirname "$0")/jarvis_pc_agent.py" "$DIR/"
python3 -m pip install --user --break-system-packages flask >/dev/null 2>&1 || python3 -m pip install --user flask
umask 077
cat > "$HOME/.config/jarvis/agent.env" <<ENV
AGENT_TOKEN=$TOKEN
AGENT_PORT=8765
AGENT_BIND=0.0.0.0
ENV
sed "s#@DIR@#$DIR#g" "$(dirname "$0")/jarvis-agent.service" > "$HOME/.config/systemd/user/jarvis-agent.service"
systemctl --user daemon-reload
systemctl --user enable --now jarvis-agent.service
echo "Jarvis PC agent running on port 8765.  Logs: journalctl --user -u jarvis-agent -f"
echo "Desktop control extras (KDE Wayland): see pc_agent/README.md"

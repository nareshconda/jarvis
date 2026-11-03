# PC agent

A small, token-protected HTTP server that runs **on each PC** Jarvis should control.
Jarvis calls it to run allow-listed commands, launch apps, create files and, on a
Linux desktop, see the screen and use the mouse and keyboard ("Agent" mode).

It only accepts requests carrying `AGENT_TOKEN`, only runs commands on its allow-list,
and should only be reachable on your LAN — never port-forward it.

## Install (Linux)

```bash
cd pc_agent
./install.sh <AGENT_TOKEN>      # same value as AGENT_TOKEN in the server's .env
```

That copies the agent to `~/.local/share/jarvis-agent`, writes `~/.config/jarvis/agent.env`
and starts a systemd **user** service that starts and stops with your desktop session.

```bash
systemctl --user status jarvis-agent
journalctl --user -u jarvis-agent -f
```

Check it from the server: `python setup.py check` lists every PC agent it can reach.

## Apps

Installed apps are discovered automatically from `.desktop` launchers (system, user,
Flatpak, Snap) and your Steam library, refreshed every 10 minutes. To add extra spoken
names or commands, create `~/.config/jarvis/apps.json`:

```json
{ "my script": "bash /home/me/bin/backup.sh", "notes": "obsidian" }
```

## Desktop control (optional — KDE Plasma on Wayland)

Agent mode can look at the screen and click/type. It needs:

```bash
# Arch / CachyOS
sudo pacman -S ydotool kdotool python-evdev python-atspi spectacle wl-clipboard
# Debian / Ubuntu
sudo apt install ydotool python3-evdev python3-pyatspi spectacle wl-clipboard   # kdotool: see its GitHub releases

systemctl --user enable --now ydotool          # virtual keyboard daemon
sudo usermod -aG input "$USER"                 # access to /dev/uinput (log out/in after)
```

Then run the built-in checks (from the server):

```bash
curl -s -X POST http://PC_IP:8765/run -H "Authorization: Bearer $AGENT_TOKEN" \
     -H 'Content-Type: application/json' -d '{"action":"gui_doctor"}' | jq -r .stdout
curl -s -X POST http://PC_IP:8765/run -H "Authorization: Bearer $AGENT_TOKEN" \
     -H 'Content-Type: application/json' -d '{"action":"gui_calibrate"}' | jq -r .stdout
```

`gui_calibrate` should end with `RESULT: OK — pointer lands on target.`

**Kill switch:** `touch ~/.jarvis_agent_stop` instantly pauses all mouse/keyboard actions;
delete the file to resume. The agent can never release it itself.

**Browsers:** for click-by-name (`ui_tree` / `click_element`), restart Firefox once after
installing; launch Chromium-based browsers with `--force-renderer-accessibility`.

## Settings (`~/.config/jarvis/agent.env`)

| Variable | Default | Meaning |
|---|---|---|
| `AGENT_TOKEN` | — | shared secret (required) |
| `AGENT_PORT` / `AGENT_BIND` | `8765` / `0.0.0.0` | where it listens |
| `JARVIS_GUI_CONTROL` | `1` | `0` disables mouse/keyboard/screen actions |
| `JARVIS_ABS_POINTER` | `1` | exact absolute pointer device (needs python-evdev) |
| `JARVIS_APPS_FILE` | `~/.config/jarvis/apps.json` | extra apps |
| `JARVIS_SCREEN_MAX_SIDE` | `1600` | screenshot size sent to the brain |
| `JARVIS_AGENT_TERMINAL` | `1` | show a terminal with the agent's activity log |

#!/usr/bin/env python3
"""
jarvis_pc_agent.py — Jarvis PC Tool Server
============================================
A small, token-protected HTTP/JSON server that exposes a BOUNDED set of safe
actions on this machine. Jarvis's agent loop calls these endpoints to actually
*do* things on your PCs. Run the same file on every PC —
only the env vars (token, port) differ.

SECURITY MODEL (read this):
  - Every request must carry the correct bearer token (AGENT_TOKEN).
  - Only commands on the ALLOWLIST run. The agent CANNOT run arbitrary shell.
    To add a capability, add it to ALLOWED_COMMANDS below.
  - This server should bind to the LAN only and never be exposed to the internet.

v10 — DESKTOP CONTROL ("computer use"):
  Adds eyes + hands so Jarvis can operate apps on screen: mouse/keyboard via
  ydotool, window control via kdotool (KWin), exact UI elements via the AT-SPI
  accessibility tree, layout-safe text entry via clipboard paste, and a kill
  switch (touch ~/.jarvis_agent_stop). Run the `gui_doctor` action to verify.

Run:
    pip install flask --break-system-packages
    AGENT_TOKEN="some-long-secret" AGENT_PORT=8765 python3 jarvis_pc_agent.py
"""

import os, sys, subprocess, shlex, json, platform, re, time, socket, signal, tempfile, threading, datetime
from flask import Flask, request, jsonify

# -----------------------------------------------------------------------------
# Config (override via environment)
# -----------------------------------------------------------------------------
AGENT_TOKEN = os.getenv("AGENT_TOKEN", "")          # REQUIRED — shared secret with Jarvis
AGENT_PORT  = int(os.getenv("AGENT_PORT", "8765"))
AGENT_BIND  = os.getenv("AGENT_BIND", "0.0.0.0")    # LAN; never public
HOSTNAME    = platform.node()

# Visible terminal: when enabled (default), the agent opens a terminal window on
# this PC's display tailing the live activity log, so you can SEE it working.
SHOW_TERMINAL = os.getenv("JARVIS_AGENT_TERMINAL", "1") == "1"
AGENT_LOG = os.path.expanduser("~/.jarvis_agent_activity.log")

# -----------------------------------------------------------------------------
# RUNNING-APP REGISTRY — remember what run_python_app launched, so a LATER,
# separate agent task can stop it WITHOUT the model having to recall the path
# (which it can't — each agent run is stateless, so it used to guess and fail).
# -----------------------------------------------------------------------------
_REGISTRY_PATH = os.path.expanduser("~/.jarvis_running_apps.json")

def _reg_load():
    try:
        with open(_REGISTRY_PATH) as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def _reg_save(entries):
    try:
        with open(_REGISTRY_PATH, "w") as f:
            json.dump(entries, f)
    except Exception as e:
        print(f"[registry] save failed: {e}")

def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)      # signal 0 = existence check, doesn't kill
        return True
    except Exception:
        return False

def _reg_prune():
    """Drop entries whose process is gone. Returns the live list."""
    live = [e for e in _reg_load() if e.get("pid") and _pid_alive(e["pid"])]
    _reg_save(live)
    return live

def _reg_add(entry):
    live = _reg_prune()
    # de-dupe by dir (a fresh launch of the same dir replaces the old record)
    live = [e for e in live if e.get("dir") != entry.get("dir")]
    live.append(entry)
    _reg_save(live)

# Per-action timeout overrides (seconds). Default is 30s; package operations
# legitimately take much longer (update_system at 30s always timed out).
ACTION_TIMEOUTS = {
    "update_system":   900,
    "install_package": 300,
}

if not AGENT_TOKEN:
    print("WARNING: AGENT_TOKEN is empty — set it before exposing on the LAN!")

# -----------------------------------------------------------------------------
# ALLOWLIST — the ONLY commands the agent may run.
# Key = short action name the agent uses. Value = the actual command template.
# Use {arg} placeholders for safe, validated argument substitution.
#
# NOTE: each action is defined ONCE here (the previous file had ~15 duplicates;
# Python silently kept the last copy — a maintenance trap, now cleaned up).
# -----------------------------------------------------------------------------
ALLOWED_COMMANDS = {
    # --- Read-only / safe status ---
    "uptime":         "uptime",
    "cpu_usage":      "top -bn1 | grep 'Cpu(s)'",                  # overall CPU usage
    "cpu_hogs":       "ps -eo comm,%cpu --sort=-%cpu | head -10",   # top 10 by CPU
    "disk_free":      "df -h /",
    "mem_free":       "free -h",
    "gpu_status":     "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader",
    "list_home":      "ls -la \"$HOME\"",
    "whoami":         "whoami",
    "running_apps":   "ps -eo comm --sort=-%mem | head -20",
    "ollama_list":    "ollama list",

    # --- App launching (Linux desktop). {app} validated against APP_MAP. ---
    "launch_app":     "systemd-run --user --quiet sh -c {app}",
    "close_app":      "pkill -9 -i -f {app}",   # -i ignore case, -f match full cmdline

    # --- File & project management ---
    # NOTE: create_file is handled NATIVELY in /run (see below) so code with
    # quotes/newlines/$ is written safely. Do NOT add `echo {content} > {path}`
    # back as a template — it corrupts generated code.
    "read_file":      "cat {path} | head -n 200",
    "move_file":      "mv {src} {dest}",
    "delete_file":    "rm -f {path}",
    "create_dir":     "mkdir -p {path}",
    "find_file":      "locate {filename} | head -n 20",
    "git_pull":       "cd {path} && git pull",

    # --- Python virtual environments & project running ---
    "create_venv":    "python3 -m venv {path}",
    "pip_install":    "{venv_path}/bin/pip install {packages}",
    # run_python_app is launched DETACHED in /run (see DETACHED_ACTIONS) so it
    # never blocks/times out. `exec` makes the launched process be python itself.
    "run_python_app": "cd {dir} && exec {dir}/venv/bin/python {dir}/app.py",
    # Verify / manage a running app
    "check_port":      "ss -ltn | grep -q :{port} && echo LISTENING on {port} || echo NOT_LISTENING on {port}",
    "read_app_log":    "tail -n 40 {dir}/app.log",
    # stop_python_app & list_apps are handled NATIVELY in /run via the running-app
    # registry (so they work without a correct dir). Templates kept as fallbacks.
    "stop_python_app": "pkill -f {dir}/venv/bin/python || echo no_process",
    "list_apps":       "true",
    # screen_capture is handled NATIVELY (returns the screen as base64 for the
    # vision model). Advertised here so the agent knows to use it for "what's on
    # my screen" instead of the file-only 'screenshot'.
    "screen_capture":  "true",

    # --- Desktop interaction (Jarvis can talk to you) ---
    "notify":         "notify-send 'Jarvis' {message} --icon=dialog-information",
    "clipboard_set":  "wl-copy {text}",
    "screenshot":     "spectacle -b -o {path}",

    # --- Media & audio control (Pipewire / Wayland) ---
    "volume_set":     "wpctl set-volume @DEFAULT_AUDIO_SINK@ {level}%",
    "volume_mute":    "wpctl set-mute @DEFAULT_AUDIO_SINK@ toggle",
    "media_playpause":"playerctl play-pause",
    "media_next":     "playerctl next",
    "media_prev":     "playerctl previous",

    # --- Display & monitor control (KDE Wayland) ---
    # Agent uses list_screens to find the ID (DP-1, HDMI-A-1...), then set_screen
    "list_screens":   "kscreen-doctor -o",
    "set_screen":     "kscreen-doctor output.{output_name}.{state}",  # state = enable|disable

    # --- Audio device control (Pipewire) ---
    "list_audio":     "wpctl status",
    "set_audio":      "wpctl set-default {sink_id}",

    # --- System management ---
    # CRITICAL: --noconfirm, otherwise pacman hangs on Y/N and times out.
    "update_system":  "sudo pacman -Syu --noconfirm",
    "install_package":"sudo pacman -S --noconfirm --needed {packages}",
    "check_logs":     "journalctl -n 50 --no-pager",

    # --- Power management ---
    "suspend_pc":     "systemctl suspend",
    "reboot_pc":      "systemctl reboot",
    "lock_screen":    "loginctl lock-session",
}

# Actions that start a long-running process. Launched detached (no wait, no 30s
# timeout) so the HTTP call returns immediately and the process keeps running.
DETACHED_ACTIONS = {"run_python_app"}

# Apps the agent may launch, by spoken name -> exec command.
# -----------------------------------------------------------------------------
# Installed apps are DISCOVERED, not hard-coded:
#   * every .desktop launcher (system, user, Flatpak)
#   * every installed Steam game (steam://rungameid/<id>)
#   * plus optional extras in JARVIS_APPS_FILE (JSON {"spoken name": "command"})
# Refreshed every 10 minutes, so newly installed apps show up on their own.
# -----------------------------------------------------------------------------
import glob as _glob
APPS_FILE = os.path.expanduser(os.getenv("JARVIS_APPS_FILE", "~/.config/jarvis/apps.json"))
_DESKTOP_DIRS = [
    "/usr/share/applications", "/usr/local/share/applications",
    "~/.local/share/applications",
    "/var/lib/flatpak/exports/share/applications",
    "~/.local/share/flatpak/exports/share/applications",
    "/var/lib/snapd/desktop/applications",
]
_FIELD_CODES = re.compile(r"\s+(%[fFuUdDnNickvm]|@@[uf]?)(?=\s|$)")

def _parse_desktop(path):
    name = exe = None
    hidden = False
    in_entry = False
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("["):
                    in_entry = (line == "[Desktop Entry]")
                    continue
                if not in_entry or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k == "Name" and name is None:
                    name = v.strip()
                elif k == "Exec" and exe is None:
                    exe = v.strip()
                elif k in ("NoDisplay", "Hidden") and v.strip().lower() == "true":
                    hidden = True
                elif k == "Type" and v.strip() != "Application":
                    hidden = True
    except Exception:
        return None, None
    if hidden or not name or not exe:
        return None, None
    exe = _FIELD_CODES.sub("", exe).replace("@@", "").strip()
    return name.lower(), exe

def _steam_games():
    games = {}
    roots = [os.path.expanduser(p) for p in ("~/.steam/steam", "~/.local/share/Steam",
                                             "~/.var/app/com.valvesoftware.Steam/.local/share/Steam")]
    libs = set()
    for r in roots:
        vdf = os.path.join(r, "steamapps", "libraryfolders.vdf")
        if os.path.isdir(os.path.join(r, "steamapps")):
            libs.add(os.path.join(r, "steamapps"))
        try:
            for m in re.finditer(r'"path"\s+"([^"]+)"', open(vdf, errors="replace").read()):
                libs.add(os.path.join(m.group(1), "steamapps"))
        except Exception:
            pass
    for lib in libs:
        for acf in _glob.glob(os.path.join(lib, "appmanifest_*.acf")):
            try:
                txt = open(acf, errors="replace").read()
                appid = re.search(r'"appid"\s+"(\d+)"', txt).group(1)
                name = re.search(r'"name"\s+"([^"]+)"', txt).group(1)
                if not re.search(r"(proton|steam linux runtime|steamworks)", name, re.I):
                    games[name.lower()] = f"steam steam://rungameid/{appid}"
            except Exception:
                continue
    return games

APP_MAP = {}
_APP_MAP_TS = {"t": 0.0}

def refresh_app_map(force=False):
    """Rebuild APP_MAP in place (so every reference sees the new list)."""
    if not force and time.time() - _APP_MAP_TS["t"] < 600:
        return APP_MAP
    found = {}
    for d in _DESKTOP_DIRS:
        for path in _glob.glob(os.path.join(os.path.expanduser(d), "*.desktop")):
            n, x = _parse_desktop(path)
            if n and n not in found:
                found[n] = x
    found.update(_steam_games())
    try:
        with open(APPS_FILE, encoding="utf-8") as f:
            for k, v in (json.load(f) or {}).items():
                found[str(k).lower()] = str(v)
    except FileNotFoundError:
        pass
    except Exception as ex:
        print(f"[jarvis_pc_agent] {APPS_FILE}: {ex}")
    APP_MAP.clear()
    APP_MAP.update(found)
    _APP_MAP_TS["t"] = time.time()
    return APP_MAP

refresh_app_map(force=True)

# =============================================================================
# v10 — DESKTOP CONTROL ("computer use"): eyes + hands for the agent
# =============================================================================
# KDE Wayland blocks fake input from ordinary apps, so each job uses the tool
# that Wayland actually allows:
#   * mouse / keyboard -> ydotool  (kernel uinput; needs ydotoold + /dev/uinput)
#   * windows          -> kdotool  (asks KWin to list / focus / close windows)
#   * UI elements      -> AT-SPI   (accessibility tree: exact buttons/fields by
#                                   name; "press" them without moving the mouse)
#   * text entry       -> clipboard paste (layout-independent — ydotool's typing
#                         assumes a US keymap and mangles @ " # £ on a UK layout)
# Coordinates: x,y are NORMALISED 0-1000 across the whole desktop by default
# (what vision models emit). Pass coords="px" for logical desktop pixels.
# Kill switch: while ~/.jarvis_agent_stop exists, every GUI action refuses.

GUI_ENABLED   = os.getenv("JARVIS_GUI_CONTROL", "1") == "1"
GUI_STOP_FLAG = os.path.expanduser("~/.jarvis_agent_stop")
SCREEN_MAX_SIDE = int(os.getenv("JARVIS_SCREEN_MAX_SIDE", "1600"))

def _session_env_defaults():
    """The agent may be started without the desktop session's environment
    (systemd, cron, ssh). Fill in what Wayland / D-Bus / ydotool need."""
    rt = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    os.environ.setdefault("XDG_RUNTIME_DIR", rt)
    if "WAYLAND_DISPLAY" not in os.environ:
        for n in ("wayland-0", "wayland-1"):
            if os.path.exists(os.path.join(rt, n)):
                os.environ["WAYLAND_DISPLAY"] = n
                break
    if "DBUS_SESSION_BUS_ADDRESS" not in os.environ and os.path.exists(os.path.join(rt, "bus")):
        os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={rt}/bus"
    if "YDOTOOL_SOCKET" not in os.environ:
        for cand in (os.path.join(rt, ".ydotool_socket"), "/tmp/.ydotool_socket"):
            if os.path.exists(cand):
                os.environ["YDOTOOL_SOCKET"] = cand
                break

_session_env_defaults()

# ---- live session env -------------------------------------------------------
# If the agent starts before Plasma (boot) or survives a logout/login, its env
# points at a dead session: spectacle falls back to X11 and crashes, AT-SPI
# talks to old sockets, windows can't be found. Re-sync from the CURRENT
# session (`systemctl --user show-environment`, which Plasma updates on login).
_SESSION_KEYS = ("WAYLAND_DISPLAY", "DISPLAY", "XDG_SESSION_ID", "XDG_SESSION_TYPE",
                 "XDG_CURRENT_DESKTOP", "DBUS_SESSION_BUS_ADDRESS", "XAUTHORITY",
                 "AT_SPI_BUS_ADDRESS", "KDE_FULL_SESSION")
_SESSION_SYNC = {"ts": 0.0, "first": None}

def _session_sync(force=False):
    if not force and time.time() - _SESSION_SYNC["ts"] < 20:
        return
    _SESSION_SYNC["ts"] = time.time()
    try:
        r = subprocess.run(["systemctl", "--user", "show-environment"],
                           capture_output=True, text=True, timeout=4)
        live = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
    except Exception:
        return
    rt = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    for k in _SESSION_KEYS:
        if live.get(k):
            os.environ[k] = live[k]
    wd = os.environ.get("WAYLAND_DISPLAY", "")
    if wd and os.path.exists(os.path.join(rt, wd)):
        os.environ["QT_QPA_PLATFORM"] = "wayland"     # spectacle/Qt: never fall back to xcb
    marker = (os.environ.get("WAYLAND_DISPLAY"), live.get("XDG_SESSION_ID"),
              live.get("AT_SPI_BUS_ADDRESS"))
    if _SESSION_SYNC["first"] is None:
        _SESSION_SYNC["first"] = marker
    elif marker != _SESSION_SYNC["first"] and any(marker):
        # new login since we started: AT-SPI/pyatspi can't be re-pointed in-process,
        # so restart ourselves cleanly (same PID under systemd, fresh connections)
        print("[jarvis_pc_agent] desktop session changed — restarting to re-attach", flush=True)
        threading.Timer(1.0, lambda: os.execv(sys.executable, [sys.executable] + sys.argv)).start()


def _gui_run(argv, timeout=10, input_text=None, detach_output=False):
    """Run a GUI helper. Returns (rc, stdout, stderr) — stdout NOT stripped.
    detach_output=True is required for wl-copy: it forks a background server
    that would otherwise hold our pipes open and hang this call."""
    _session_sync()
    try:
        if detach_output:
            p = subprocess.run(argv, input=input_text, text=True, timeout=timeout,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return p.returncode, "", ""
        p = subprocess.run(argv, input=input_text, text=True, timeout=timeout,
                           capture_output=True)
        return p.returncode, (p.stdout or ""), (p.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", f"{argv[0]} is not installed"
    except subprocess.TimeoutExpired:
        return 124, "", f"{argv[0]} timed out"
    except Exception as e:
        return 1, "", f"{argv[0]}: {e}"


def _ydo(argv, timeout=10):
    rc, out, err = _gui_run(["ydotool"] + argv, timeout=timeout)
    if rc != 0 and ("socket" in err.lower() or "connect" in err.lower()):
        err += (" | ydotoold isn't reachable — run `systemctl --user enable --now ydotool` "
                "and check /dev/uinput permissions (see setup).")
    return rc, out, err


def _kdo(argv, timeout=8):
    return _gui_run(["kdotool"] + argv, timeout=timeout)


# ------------------------------------------------------------ screen geometry
_GEOM_CACHE = {"ts": 0.0, "g": None}
_LAST_SHOT = {"w": 0, "h": 0}

def _desktop_geometry(force=False):
    """Bounding box of all enabled monitors in LOGICAL pixels (KDE scaling
    applied), from kscreen-doctor. Cached for 30s."""
    if not force and _GEOM_CACHE["g"] and time.time() - _GEOM_CACHE["ts"] < 30:
        return _GEOM_CACHE["g"]
    g = None
    rc, out, _ = _gui_run(["kscreen-doctor", "-j"], timeout=8)
    if rc == 0 and out.strip():
        try:
            data = json.loads(out)
            boxes = []
            for o in data.get("outputs", []):
                if not o.get("enabled"):
                    continue
                mode = next((m for m in o.get("modes", [])
                             if str(m.get("id")) == str(o.get("currentModeId"))), None)
                sz = (mode or {}).get("size") or o.get("size") or {}
                w, h = float(sz.get("width") or 0), float(sz.get("height") or 0)
                if not (w and h):
                    continue
                if int(o.get("rotation") or 1) in (2, 8):        # left / right
                    w, h = h, w
                sc = float(o.get("scale") or 1) or 1.0
                pos = o.get("pos") or {}
                x, y = float(pos.get("x") or 0), float(pos.get("y") or 0)
                boxes.append((x, y, x + w / sc, y + h / sc, o.get("name")))
            if boxes:
                minx, miny = min(b[0] for b in boxes), min(b[1] for b in boxes)
                maxx, maxy = max(b[2] for b in boxes), max(b[3] for b in boxes)
                g = {"x": int(minx), "y": int(miny),
                     "w": int(round(maxx - minx)), "h": int(round(maxy - miny)),
                     "outputs": [{"name": b[4], "x": int(b[0]), "y": int(b[1]),
                                  "w": int(round(b[2] - b[0])), "h": int(round(b[3] - b[1]))}
                                 for b in boxes],
                     "source": "kscreen-doctor"}
        except Exception:
            g = None
    if not g:
        g = {"x": 0, "y": 0, "w": _LAST_SHOT["w"], "h": _LAST_SHOT["h"],
             "outputs": [], "source": "last-screenshot" if _LAST_SHOT["w"] else "unknown"}
    _GEOM_CACHE.update(ts=time.time(), g=g)
    return g


def _target_xy(args):
    """args x,y -> absolute logical desktop pixels."""
    if "x" not in args or "y" not in args:
        raise ValueError("missing x / y")
    x, y = float(args["x"]), float(args["y"])
    if str(args.get("coords", "norm")).lower() in ("px", "pixel", "pixels"):
        return x, y
    if args.get("window_id"):
        # x,y are 0-1000 over a WINDOW screenshot (what the brain was shown)
        wg = _parse_geo(_kdo(["getwindowgeometry", str(args["window_id"])])[1])
        if not wg.get("w"):
            raise ValueError("that window is gone — take a new screen_capture")
        x = max(0.0, min(1000.0, x)); y = max(0.0, min(1000.0, y))
        return wg["x"] + x / 1000.0 * wg["w"], wg["y"] + y / 1000.0 * wg["h"]
    g = _desktop_geometry()
    if not (g["w"] and g["h"]):
        raise ValueError("unknown screen size — run screen_capture or gui_doctor first")
    x = max(0.0, min(1000.0, x))
    y = max(0.0, min(1000.0, y))
    return g["x"] + x / 1000.0 * g["w"], g["y"] + y / 1000.0 * g["h"]


def _to_norm(ax, ay):
    g = _desktop_geometry()
    if not (g["w"] and g["h"]):
        return None, None
    return (int(round((ax - g["x"]) / g["w"] * 1000)),
            int(round((ay - g["y"]) / g["h"] * 1000)))


POINTER_TOL = int(os.getenv("JARVIS_POINTER_TOL", "3"))     # px

# ---- absolute pointer (virtual "tablet") ------------------------------------
# ydotool's --absolute is fake: it slams the cursor to (0,0) and then sends a
# relative move — KWin drops/accelerates that, so it lands at (1,1) or way off.
# A uinput device with ABS_X/ABS_Y (like a VM's USB tablet) is mapped by KWin
# onto the whole desktop with no acceleration: the cursor goes exactly there.
# Needs python-evdev (`sudo pacman -S python-evdev`) + /dev/uinput access
# (same access ydotoold already uses).
_ABS_RANGE = 32767
_ABS_DEV = {"dev": None, "tried": False, "err": ""}

def _abs_dev():
    if _ABS_DEV["dev"] is not None or _ABS_DEV["tried"]:
        return _ABS_DEV["dev"]
    _ABS_DEV["tried"] = True
    if os.getenv("JARVIS_ABS_POINTER", "1") != "1":
        _ABS_DEV["err"] = "disabled (JARVIS_ABS_POINTER=0)"
        return None
    try:
        from evdev import UInput, ecodes as e, AbsInfo
        caps = {
            e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
            e.EV_ABS: [(e.ABS_X, AbsInfo(0, 0, _ABS_RANGE, 0, 0, 0)),
                       (e.ABS_Y, AbsInfo(0, 0, _ABS_RANGE, 0, 0, 0))],
            e.EV_REL: [e.REL_WHEEL, e.REL_HWHEEL],
        }
        _ABS_DEV["dev"] = UInput(caps, name="jarvis-abs-pointer", vendor=0x1234, product=0x5678)
        time.sleep(1.0)                      # let KWin/libinput pick the new device up
    except Exception as ex:
        _ABS_DEV["err"] = f"{type(ex).__name__}: {ex}"
        _ABS_DEV["dev"] = None
    return _ABS_DEV["dev"]

def _abs_move(ax, ay):
    dev = _abs_dev()
    if dev is None:
        return False
    from evdev import ecodes as e
    g = _desktop_geometry()
    w, h = max(1, g["w"] - 1), max(1, g["h"] - 1)
    vx = int(round(max(0, min(w, ax - g["x"])) / w * _ABS_RANGE))
    vy = int(round(max(0, min(h, ay - g["y"])) / h * _ABS_RANGE))
    dev.write(e.EV_ABS, e.ABS_X, vx)
    dev.write(e.EV_ABS, e.ABS_Y, vy)
    dev.syn()
    return True

def _abs_click(button="left", count=1):
    dev = _abs_dev()
    if dev is None:
        return False
    from evdev import ecodes as e
    code = {"left": e.BTN_LEFT, "right": e.BTN_RIGHT, "middle": e.BTN_MIDDLE}.get(button, e.BTN_LEFT)
    for i in range(max(1, min(3, int(count)))):
        dev.write(e.EV_KEY, code, 1); dev.syn(); time.sleep(0.03)
        dev.write(e.EV_KEY, code, 0); dev.syn()
        if i < count - 1:
            time.sleep(0.07)
    return True

def _btn_state(button, down):
    dev = _abs_dev()
    if dev is not None:
        from evdev import ecodes as e
        code = {"left": e.BTN_LEFT, "right": e.BTN_RIGHT, "middle": e.BTN_MIDDLE}.get(button, e.BTN_LEFT)
        dev.write(e.EV_KEY, code, 1 if down else 0); dev.syn()
        return
    _ydo(["click", (_BTN_DOWN if down else _BTN_UP)[button]])

def _abs_scroll(dx, dy):
    dev = _abs_dev()
    if dev is None:
        return False
    from evdev import ecodes as e
    if dy:
        dev.write(e.EV_REL, e.REL_WHEEL, int(dy))
    if dx:
        dev.write(e.EV_REL, e.REL_HWHEEL, int(dx))
    dev.syn()
    return True
_LAST_POINTER = {"x": None, "y": None, "ts": 0.0, "err": None, "verified": False}

def _cursor_pos():
    """Where the pointer REALLY is (KWin logical coords), via kdotool."""
    rc, out, _ = _kdo(["getmouselocation"], timeout=4)
    m = re.search(r"x:\s*(-?\d+(?:\.\d+)?)\D+y:\s*(-?\d+(?:\.\d+)?)", out or "")
    return (float(m.group(1)), float(m.group(2))) if (rc == 0 and m) else None

def _move(ax, ay, closed=True):
    """Closed-loop pointer move. ydotool's --absolute is open-loop: pointer
    acceleration and display scaling make it land off-target, so read back the
    real cursor position and nudge it until it's within POINTER_TOL px."""
    g = _desktop_geometry()
    if _abs_move(ax, ay):
        rc, out, err = 0, "", ""
    else:
        rc, out, err = _ydo(["mousemove", "--absolute",
                             "-x", str(int(round(ax - g["x"]))), "-y", str(int(round(ay - g["y"])))])
    if rc != 0 or not closed:
        return rc, out, err
    verified, miss = False, None
    aim_x, aim_y = ax, ay
    for _ in range(5):
        time.sleep(0.06)
        p = _cursor_pos()
        if p is None:
            break                                   # can't read it back — trust the move
        dx, dy = ax - p[0], ay - p[1]
        miss = (dx * dx + dy * dy) ** 0.5
        if abs(dx) <= POINTER_TOL and abs(dy) <= POINTER_TOL:
            verified = True
            break
        if _ABS_DEV["dev"] is not None:
            # absolute device: aim past the error (compensates any fixed offset)
            aim_x, aim_y = aim_x + dx, aim_y + dy
            _abs_move(aim_x, aim_y)
        else:
            _ydo(["mousemove", "-x", str(int(round(dx))), "-y", str(int(round(dy)))])
    _LAST_POINTER.update(x=ax, y=ay, ts=time.time(), err=miss, verified=verified)
    return rc, out, err

def _pointer_note():
    if _LAST_POINTER["verified"]:
        return " (pointer verified on target)"
    if _LAST_POINTER["err"] is not None:
        return f" (WARNING: pointer ended {int(_LAST_POINTER['err'])}px off target)"
    return " (pointer position unverified)"

def _gui_calibrate():
    """Move to 5 points; report open-loop error and closed-loop result."""
    g = _desktop_geometry()
    dev = _abs_dev()
    lines = [f"desktop {g.get('w')}x{g.get('h')} at ({g.get('x')},{g.get('y')}) via {g.get('source')}",
             "pointer backend: " + ("absolute uinput device (exact)" if dev is not None else
                                    f"ydotool relative (fallback — abs device unavailable: {_ABS_DEV['err']})")]
    worst = 0.0
    for nx, ny in ((100, 100), (900, 100), (500, 500), (100, 900), (900, 900)):
        ax, ay = _target_xy({"x": nx, "y": ny})
        if not _abs_move(ax, ay):
            _ydo(["mousemove", "--absolute", "-x", str(int(ax - g["x"])), "-y", str(int(ay - g["y"]))])
        time.sleep(0.1)
        p0 = _cursor_pos()
        _move(ax, ay)
        p1 = _cursor_pos()
        if p0 is None:
            lines.append(f"norm({nx},{ny}) -> target ({int(ax)},{int(ay)}): cannot read cursor "
                         "(kdotool getmouselocation unsupported?)")
            continue
        e0 = ((ax - p0[0]) ** 2 + (ay - p0[1]) ** 2) ** 0.5
        e1 = ((ax - p1[0]) ** 2 + (ay - p1[1]) ** 2) ** 0.5 if p1 else -1
        worst = max(worst, e1)
        lines.append(f"norm({nx},{ny}) -> target ({int(ax)},{int(ay)}): first landed "
                     f"({int(p0[0])},{int(p0[1])}) err={e0:.0f}px; after correction err={e1:.0f}px")
    lines.append("RESULT: " + ("OK — pointer lands on target." if worst <= POINTER_TOL * 2
                                else "pointer is still off — see the fixes in the setup notes."))
    return "\n".join(lines)


_BTN_CLICK = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}   # down+up
_BTN_DOWN  = {"left": "0x40", "right": "0x41", "middle": "0x42"}
_BTN_UP    = {"left": "0x80", "right": "0x81", "middle": "0x82"}

def _click(button="left", count=1):
    if _abs_click(button, count):
        return 0, "", ""
    code = _BTN_CLICK.get(button, _BTN_CLICK["left"])
    for i in range(max(1, min(3, int(count)))):
        rc, out, err = _ydo(["click", code])
        if rc != 0:
            return rc, out, err
        if i < count - 1:
            time.sleep(0.07)                       # inside the double-click window
    return 0, "", ""


# Linux input-event keycodes for `ydotool key <code>:<1|0>`
_KEYCODES = {
    "esc": 1, "escape": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8,
    "8": 9, "9": 10, "0": 11, "minus": 12, "-": 12, "equal": 13, "=": 13,
    "backspace": 14, "tab": 15, "q": 16, "w": 17, "e": 18, "r": 19, "t": 20,
    "y": 21, "u": 22, "i": 23, "o": 24, "p": 25, "[": 26, "]": 27,
    "enter": 28, "return": 28, "ctrl": 29, "control": 29, "a": 30, "s": 31,
    "d": 32, "f": 33, "g": 34, "h": 35, "j": 36, "k": 37, "l": 38, ";": 39,
    "'": 40, "`": 41, "shift": 42, "\\": 43, "z": 44, "x": 45, "c": 46, "v": 47,
    "b": 48, "n": 49, "m": 50, ",": 51, ".": 52, "/": 53, "alt": 56,
    "space": 57, "capslock": 58, "f1": 59, "f2": 60, "f3": 61, "f4": 62,
    "f5": 63, "f6": 64, "f7": 65, "f8": 66, "f9": 67, "f10": 68, "f11": 87,
    "f12": 88, "rightctrl": 97, "printscreen": 99, "print": 99, "altgr": 100,
    "home": 102, "up": 103, "pageup": 104, "pgup": 104, "left": 105,
    "right": 106, "end": 107, "down": 108, "pagedown": 109, "pgdn": 109,
    "insert": 110, "delete": 111, "del": 111, "mute": 113, "volumedown": 114,
    "volumeup": 115, "super": 125, "meta": 125, "win": 125, "cmd": 125,
    "menu": 127, "playpause": 164,
}

def _key_combo(combo, repeat=1):
    """'ctrl+shift+t' -> press in order, release in reverse."""
    parts = [p.strip().lower() for p in re.split(r"\s*\+\s*", str(combo).strip()) if p.strip()]
    if not parts:
        raise ValueError("empty key combo")
    codes = []
    for p in parts:
        if p not in _KEYCODES:
            raise ValueError(f"unknown key '{p}' (try e.g. ctrl, shift, alt, super, enter, "
                             "tab, esc, up, f5, a-z, 0-9)")
        codes.append(_KEYCODES[p])
    seq = [f"{c}:1" for c in codes] + [f"{c}:0" for c in reversed(codes)]
    for i in range(max(1, min(20, int(repeat)))):
        rc, out, err = _ydo(["key", "-d", "20"] + seq)
        if rc != 0:
            return rc, out, err
        if i:
            time.sleep(0.05)
    return 0, "", ""


# ------------------------------------------------------------ windows (KWin)
def _parse_geo(txt):
    # KWin reports fractional positions/sizes on fractionally-scaled outputs
    num = r"(-?\d+(?:\.\d+)?)"
    pos = re.search(r"Position:\s*" + num + r"\s*,\s*" + num, txt or "")
    size = re.search(r"Geometry:\s*" + num + r"\s*x\s*" + num, txt or "")
    if pos and size:
        return {"x": int(round(float(pos.group(1)))), "y": int(round(float(pos.group(2)))),
                "w": int(round(float(size.group(1)))), "h": int(round(float(size.group(2))))}
    return {}


def _list_windows():
    rc, out, err = _kdo(["search", "--name", "."])
    ids = [l.strip() for l in out.splitlines() if l.strip()]
    if not ids:
        return [], (err or "no windows found (is kdotool installed?)")
    names = []
    rc2, out2, _ = _kdo(["search", "--name", ".", "getwindowname", "%@"], timeout=12)
    if rc2 == 0:
        names = [l.strip() for l in out2.splitlines()]
    if len(names) != len(ids):                          # chaining unsupported -> per window
        names = [(_kdo(["getwindowname", w])[1] or "").strip() for w in ids[:25]]
        ids = ids[:25]
    return [{"id": w, "name": n} for w, n in zip(ids, names) if n], ""


def _active_window():
    rc, wid, err = _kdo(["getactivewindow"])
    wid = wid.strip()
    if rc != 0 or not wid:
        return {}
    name = (_kdo(["getwindowname", wid])[1] or "").strip()
    geo = _parse_geo(_kdo(["getwindowgeometry", wid])[1])
    return {"id": wid, "name": name, **geo}


def _is_log_terminal(w):
    return (w or {}).get("name", "").strip().lower() == "jarvis agent"


def _find_window(query):
    wins, err = _list_windows()
    q = (query or "").strip().lower()
    if not q:
        return None, wins, "missing window name"
    wins = [w for w in wins if not _is_log_terminal(w)]
    exact = [w for w in wins if w["name"].lower() == q]
    part = [w for w in wins if q in w["name"].lower()]
    pick = (exact or part or [None])[0]
    return pick, wins, ("" if pick else f"no window matching '{query}'")


# ------------------------------------------------------- accessibility (AT-SPI)
_ATSPI_LOCK = threading.Lock()
_UI_SNAP = {"ts": 0.0, "items": []}
_INTERACTIVE_ROLES = {
    "push button", "toggle button", "button", "check box", "radio button",
    "menu item", "check menu item", "radio menu item", "menu", "combo box",
    "text", "entry", "password text", "link", "page tab", "list item",
    "tree item", "spin button", "slider", "icon", "table cell", "tool bar",
}

def _pyatspi():
    _session_sync()
    try:
        import pyatspi            # pacman: python-atspi
        return pyatspi
    except Exception:
        return None


def _atspi_target_window(pa, app_hint=""):
    """The ACTIVE top-level window (or the one whose app/title matches app_hint)."""
    desktop = pa.Registry.getDesktop(0)
    hint = (app_hint or "").strip().lower()
    fallback = None
    for i in range(desktop.childCount):
        app_ = desktop.getChildAtIndex(i)
        if app_ is None:
            continue
        aname = (app_.name or "").lower()
        for j in range(app_.childCount):
            win = app_.getChildAtIndex(j)
            if win is None:
                continue
            try:
                active = win.getState().contains(pa.STATE_ACTIVE)
            except Exception:
                continue
            if hint:
                if hint in aname or hint in (win.name or "").lower():
                    if active:
                        return app_, win
                    fallback = fallback or (app_, win)
            elif active:
                return app_, win
    return fallback or (None, None)


def _atspi_walk(pa, root, node_limit=4000, max_items=200):
    items, stack, seen = [], [(root, 0)], 0
    while stack and seen < node_limit and len(items) < max_items:
        acc, depth = stack.pop()
        seen += 1
        try:
            st = acc.getState()
            if depth and not (st.contains(pa.STATE_SHOWING) and st.contains(pa.STATE_VISIBLE)):
                continue                                  # hidden subtree — skip it
            role = acc.getRoleName()
        except Exception:
            continue
        if role in _INTERACTIVE_ROLES:
            name = (acc.name or "").strip()
            if not name and role in ("text", "entry"):
                try:
                    t = acc.queryText()
                    name = (t.getText(0, min(t.characterCount, 60)) or "").strip()
                except Exception:
                    pass
            try:
                ext = acc.queryComponent().getExtents(pa.WINDOW_COORDS)
                ex, ey, ew, eh = ext.x, ext.y, ext.width, ext.height
            except Exception:
                ex = ey = ew = eh = 0
            if ew > 0 and eh > 0:
                items.append({
                    "_acc": acc, "role": role, "name": name[:80],
                    "wx": ex, "wy": ey, "w": ew, "h": eh,
                    "focused": st.contains(pa.STATE_FOCUSED),
                    "enabled": st.contains(pa.STATE_ENABLED) or st.contains(pa.STATE_SENSITIVE),
                    "checked": st.contains(pa.STATE_CHECKED),
                })
        if depth < 60:
            try:
                n = min(acc.childCount, 400)
                for k in range(n - 1, -1, -1):            # reversed -> document order
                    c = acc.getChildAtIndex(k)
                    if c is not None:
                        stack.append((c, depth + 1))
            except Exception:
                pass
    return items


UI_NODE_LIMIT = int(os.getenv("JARVIS_UI_NODE_LIMIT", "15000"))   # web pages are deep
UI_MAX_ITEMS  = int(os.getenv("JARVIS_UI_MAX_ITEMS", "400"))

def _ui_snapshot(app_hint=""):
    """Dump the interactive elements of the active (or named) window, with
    approximate SCREEN positions. On Wayland, apps only know window-relative
    coordinates, so we add the KWin frame position + title-bar offset."""
    pa = _pyatspi()
    if pa is None:
        raise RuntimeError("pyatspi missing — `sudo pacman -S python-atspi at-spi2-core`")
    with _ATSPI_LOCK:
        app_, win = _atspi_target_window(pa, app_hint)
        if win is None:
            raise RuntimeError("no accessible active window found — is accessibility "
                               "enabled for this app? (see setup: QT_LINUX_ACCESSIBILITY_ALWAYS_ON)")
        items = _atspi_walk(pa, win, node_limit=UI_NODE_LIMIT, max_items=UI_MAX_ITEMS)
        try:
            wext = win.queryComponent().getExtents(pa.WINDOW_COORDS)
            cw, ch = wext.width, wext.height
        except Exception:
            cw = ch = 0
    frame = _active_window() if not app_hint else {}
    if app_hint:                                   # named app: use its window geometry
        pick, _, _ = _find_window(app_hint)
        if pick:
            frame = {**pick, **_parse_geo(_kdo(["getwindowgeometry", pick["id"]])[1])}
    fx, fy = frame.get("x", 0), frame.get("y", 0)
    fw, fh = frame.get("w", 0), frame.get("h", 0)
    off_x = max(0, (fw - cw) // 2) if (fw and cw) else 0
    off_y = max(0, fh - ch - off_x) if (fh and ch) else 0     # title bar ≈ extra height
    for idx, it in enumerate(items):
        it["idx"] = idx
        it["sx"] = fx + off_x + it["wx"] + it["w"] / 2.0
        it["sy"] = fy + off_y + it["wy"] + it["h"] / 2.0
        it["nx"], it["ny"] = _to_norm(it["sx"], it["sy"])
    _UI_SNAP.update(ts=time.time(), items=items,
                    app=(app_.name if app_ is not None else ""),
                    window=(win.name if win is not None else ""))
    return items


def _public_item(it):
    return {k: v for k, v in it.items() if not k.startswith("_")}


def _resolve_element(args):
    """Pick an element by index (from the last ui_tree) or by name (fresh dump)."""
    if args.get("index") not in (None, ""):
        if time.time() - _UI_SNAP["ts"] > 120:
            raise ValueError("ui_tree snapshot is stale — call ui_tree again first")
        i = int(args["index"])
        items = _UI_SNAP["items"]
        if not (0 <= i < len(items)):
            raise ValueError(f"index {i} out of range (0..{len(items) - 1})")
        return items[i]
    name = str(args.get("name") or "").strip().lower()
    if not name:
        raise ValueError("give index (from ui_tree) or name")
    items = _ui_snapshot(args.get("app", ""))
    role = str(args.get("role") or "").strip().lower()
    pool = [it for it in items if not role or it["role"] == role]
    for test in (lambda n: n == name, lambda n: n.startswith(name), lambda n: name in n):
        hits = [it for it in pool if test(it["name"].lower())]
        if hits:
            return hits[0]
    raise ValueError(f"no element named '{args.get('name')}' in "
                     f"{('app ' + repr(args.get('app'))) if args.get('app') else 'the active window'} "
                     f"({len(items)} elements scanned) — call ui_tree to see what's there, or "
                     "click it from the screenshot")


def _press_element(it):
    """Invoke the element's own action (no mouse needed). Returns action name or ''."""
    try:
        ai = it["_acc"].queryAction()
        names = [(ai.getName(i) or "").lower() for i in range(ai.nActions)]
        for pref in ("click", "press", "activate", "jump", "toggle", "open"):
            if pref in names:
                ai.doAction(names.index(pref))
                return pref
        if ai.nActions > 0:
            ai.doAction(0)
            return names[0] or "action0"
    except Exception:
        pass
    return ""


def _focused_is_password():
    """Best-effort: is keyboard focus on a password field? Then we refuse to type."""
    pa = _pyatspi()
    if pa is None:
        return False
    try:
        with _ATSPI_LOCK:
            _, win = _atspi_target_window(pa)
            if win is None:
                return False
            for it in _atspi_walk(pa, win, node_limit=2500, max_items=400):
                if it["focused"] and it["role"] == "password text":
                    return True
    except Exception:
        pass
    return False


_TERMINAL_HINTS = ("konsole", "alacritty", "kitty", "xterm", "terminal", "foot", "wezterm", "termius")

def _type_text(text, method="paste", enter=False):
    if _focused_is_password():
        return False, ("Refusing to type into a PASSWORD field. Passwords are entered by the "
                       "user — hand control back.")
    if method == "type":                                  # raw keystrokes (US keymap!)
        rc, _, err = _ydo(["type", "-d", "6", "--", text], timeout=10 + len(text) // 15)
        if rc != 0:
            return False, f"ydotool type failed: {err}"
    else:                                                 # clipboard paste (default)
        rc0, old, _ = _gui_run(["wl-paste", "-n", "-t", "text/plain"], timeout=3)
        old = old if rc0 == 0 else None
        rc, _, _ = _gui_run(["wl-copy"], input_text=text, timeout=5, detach_output=True)
        if rc != 0:
            return False, "wl-copy failed (install wl-clipboard)"
        time.sleep(0.12)
        wname = (_active_window().get("name") or "").lower()
        combo = "ctrl+shift+v" if any(t in wname for t in _TERMINAL_HINTS) else "ctrl+v"
        rc, _, err = _key_combo(combo)
        time.sleep(0.35)                                  # let the app read the clipboard
        if old is not None:                               # restore the user's clipboard
            _gui_run(["wl-copy"], input_text=old, timeout=5, detach_output=True)
        if rc != 0:
            return False, f"paste failed: {err}"
    if enter:
        _key_combo("enter")
    return True, f"typed {len(text)} chars via {method}" + (" + Enter" if enter else "")


# ------------------------------------------------------------ screen capture
def _capture_screen(max_side=None, window=None):
    """Full-desktop screenshot -> dict with base64 PNG + geometry for mapping
    the vision model's coordinates back onto the real screen.
    window=<title substring>: crop to that window (it must be visible, not
    covered/minimised) — far fewer pixels for a vision model to read."""
    import base64 as _b64
    _session_sync()                               # spectacle needs the LIVE Wayland session
    max_side = int(max_side or SCREEN_MAX_SIDE)
    crop_win = None
    if window == "@active":
        aw = _active_window()
        if not aw.get("id") or _is_log_terminal(aw):
            window = None                     # nothing usable focused -> whole desktop
        else:
            crop_win = {"id": aw["id"], "name": aw.get("name", "")}
    elif window:
        crop_win, _, werr = _find_window(window)
        if not crop_win:
            return {"ok": False, "error": werr or f"no window matching '{window}'"}
    if crop_win:
        crop_win = {**crop_win, **_parse_geo(_kdo(["getwindowgeometry", crop_win["id"]])[1])}
        if not crop_win.get("w"):
            if window == "@active":
                crop_win = None           # can't crop -> whole desktop is still useful
            else:
                return {"ok": False, "error": f"couldn't read geometry of '{crop_win['name']}'"}
    shot = os.path.join(tempfile.gettempdir(), f"jarvis_screen_{int(time.time() * 1000)}.png")
    candidates = [
        ["spectacle", "-b", "-n", "-f", "-o", shot],   # KDE: all monitors, no UI/notification
        ["grim", shot],                                 # wlroots compositors
        ["scrot", "-o", shot],                          # X11
        ["import", "-window", "root", shot],            # ImageMagick X11
        ["gnome-screenshot", "-f", shot],               # GNOME
    ]
    last_err = ""
    for cmd in candidates:
        if not shutil.which(cmd[0]):
            continue
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            if os.path.isfile(shot) and os.path.getsize(shot) > 1000:
                break
            last_err = (p.stderr or p.stdout or f"{cmd[0]} produced no image").strip()[:200]
        except Exception as e:
            last_err = f"{cmd[0]}: {e}"
    if not (os.path.isfile(shot) and os.path.getsize(shot) > 1000):
        return {"ok": False, "error": f"Screen capture failed. {last_err}"}
    ow = oh = sw = sh = 0
    try:
        from PIL import Image
        img = Image.open(shot).convert("RGB")
        ow, oh = img.size
        g = _desktop_geometry()
        sc = ow / float(g.get("w") or ow)                # logical -> physical pixels
        # mark where the LAST click landed, so the brain can see if it missed
        if _LAST_POINTER["x"] is not None and time.time() - _LAST_POINTER["ts"] < 20:
            from PIL import ImageDraw
            mx = (_LAST_POINTER["x"] - g.get("x", 0)) * sc
            my = (_LAST_POINTER["y"] - g.get("y", 0)) * sc
            r = max(10, int(14 * sc))
            d = ImageDraw.Draw(img)
            d.ellipse((mx - r, my - r, mx + r, my + r), outline=(255, 0, 0), width=max(3, r // 4))
            d.line((mx - 2 * r, my, mx + 2 * r, my), fill=(255, 0, 0), width=2)
            d.line((mx, my - 2 * r, mx, my + 2 * r), fill=(255, 0, 0), width=2)
        if crop_win:
            box = (max(0, int((crop_win["x"] - g.get("x", 0)) * sc)),
                   max(0, int((crop_win["y"] - g.get("y", 0)) * sc)),
                   min(ow, int((crop_win["x"] - g.get("x", 0) + crop_win["w"]) * sc)),
                   min(oh, int((crop_win["y"] - g.get("y", 0) + crop_win["h"]) * sc)))
            if box[2] > box[0] and box[3] > box[1]:
                img = img.crop(box)
        img.thumbnail((max_side, max_side))
        sw, sh = img.size
        # JPEG: ~5x smaller and much faster to encode than optimised PNG
        img.convert("RGB").save(shot, "JPEG", quality=int(os.getenv("JARVIS_SHOT_QUALITY", "82")))
    except Exception:
        pass                                            # Pillow optional
    try:
        with open(shot, "rb") as f:
            b64 = _b64.b64encode(f.read()).decode()
    except Exception as e:
        return {"ok": False, "error": f"read failed: {e}"}
    finally:
        try:
            os.remove(shot)
        except Exception:
            pass
    if ow and oh:
        _LAST_SHOT.update(w=ow, h=oh)
    if crop_win:
        return {"ok": True, "image_b64": b64, "window": crop_win["name"],
                "window_id": crop_win["id"], "sent_w": sw, "sent_h": sh,
                "coords_note": "x,y 0-1000 over THIS window image; pass window_id with clicks"}
    return {"ok": True, "image_b64": b64, "orig_w": ow, "orig_h": oh,
            "sent_w": sw, "sent_h": sh, "geometry": _desktop_geometry(),
            "coords_note": "x,y for GUI actions are normalised 0-1000 over this whole image"}


# ------------------------------------------------------------ GUI dispatcher
GUI_ACTIONS = {
    "screen_capture", "screen_info", "gui_doctor", "gui_pause", "gui_resume",
    "move_mouse", "click", "double_click", "right_click", "drag", "scroll",
    "key_combo", "type_text", "list_windows", "active_window", "focus_window",
    "close_window", "ui_tree", "click_element", "set_text", "open_url", "wait",
    "gui_calibrate",
}
# read-only actions keep working while paused (so Jarvis can still SEE and report)
_GUI_ALWAYS_OK = {"screen_capture", "screen_info", "gui_doctor", "gui_resume", "wait"}
_GUI_PAUSE_STAMP = "paused-by-jarvis-agent"


def _stop_flag_info():
    """Who engaged the kill switch, and when — so a flag that keeps coming back
    can be traced. gui_pause writes a stamp; an EMPTY file means something
    outside this agent created it (a `touch`, a sync client, a script...)."""
    try:
        st = os.stat(GUI_STOP_FLAG)
        when = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%H:%M:%S")
        try:
            body = open(GUI_STOP_FLAG, errors="replace").read(300).strip()
        except Exception:
            body = ""
        who = (f"by this agent's gui_pause ({body})" if body.startswith(_GUI_PAUSE_STAMP)
               else f"by something OUTSIDE the agent (file content: {body[:80]!r})" if body
               else "by something OUTSIDE the agent (empty file, e.g. `touch`)")
        return f"engaged at {when} {who}"
    except FileNotFoundError:
        return "off"

def _gui_resp(action, ok, stdout="", stderr="", **extra):
    body = {"ok": bool(ok), "host": HOSTNAME, "action": action,
            "returncode": 0 if ok else 1, "stdout": stdout, "stderr": stderr}
    body.update(extra)
    _log_activity(f"  -> {action} ok={bool(ok)} {(stdout or stderr)[:160]}")
    return jsonify(body), 200


def _gui_doctor():
    _session_env_defaults()
    lines, fixes = [], []
    def tool(name, pkg):
        ok = bool(shutil.which(name))
        lines.append(f"{'OK ' if ok else 'MISSING'}  {name}")
        if not ok:
            fixes.append(f"install {pkg}")
        return ok
    tool("ydotool", "`sudo pacman -S ydotool`")
    tool("kdotool", "`paru -S kdotool` (AUR)")
    tool("wl-copy", "`sudo pacman -S wl-clipboard`")
    tool("kscreen-doctor", "`sudo pacman -S libkscreen`")
    tool("spectacle", "`sudo pacman -S spectacle`")
    sock = os.environ.get("YDOTOOL_SOCKET", "")
    lines.append(f"ydotool socket: {sock or '(not found)'}")
    rc, _, err = _ydo(["mousemove", "-x", "0", "-y", "0"])   # harmless: moves by 0
    lines.append("ydotoold: " + ("OK" if rc == 0 else f"FAIL ({err[:120]})"))
    if rc != 0:
        fixes.append("`systemctl --user enable --now ydotool` + uinput permissions")
    try:
        lines.append(f"/dev/uinput writable: {os.access('/dev/uinput', os.W_OK)}")
    except Exception:
        pass
    aw = _active_window()
    lines.append(f"kdotool active window: {aw.get('name') or 'FAIL'}")
    pa = _pyatspi()
    if pa is None:
        lines.append("AT-SPI: pyatspi MISSING")
        fixes.append("`sudo pacman -S python-atspi at-spi2-core`")
    else:
        try:
            with _ATSPI_LOCK:
                d = pa.Registry.getDesktop(0)
                apps = [(d.getChildAtIndex(i).name or "?") for i in range(min(d.childCount, 30))
                        if d.getChildAtIndex(i) is not None]
            lines.append(f"AT-SPI apps visible ({len(apps)}): {', '.join(apps[:20])}")
            if len(apps) <= 1:
                fixes.append("enable accessibility (QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1, re-login)")
        except Exception as e:
            lines.append(f"AT-SPI error: {e}")
    g = _desktop_geometry(force=True)
    lines.append(f"desktop: {g['w']}x{g['h']} at ({g['x']},{g['y']}) via {g['source']}; "
                 f"outputs={[(o['name'], o['w'], o['h']) for o in g['outputs']]}")
    lines.append(f"kill switch: {_stop_flag_info()}")
    return lines, fixes


def handle_gui(action, args):
    if not GUI_ENABLED:
        return _gui_resp(action, False, stderr="GUI control is disabled (JARVIS_GUI_CONTROL=0).")
    if action not in _GUI_ALWAYS_OK and os.path.exists(GUI_STOP_FLAG):
        return _gui_resp(action, False, stderr=(
            "GUI control is PAUSED by the user (kill switch engaged). Stop and tell the "
            f"user; only the user may resume it (delete {GUI_STOP_FLAG}). "
            f"Kill switch {_stop_flag_info()}."))
    after = str(args.get("screenshot_after", "")).lower() in ("1", "true", "yes")

    def done(ok, stdout="", stderr="", **extra):
        if ok and after:
            time.sleep(float(args.get("settle", 0.6)))
            shot = _capture_screen()
            if shot.get("ok"):
                extra.update({k: shot[k] for k in ("image_b64", "orig_w", "orig_h",
                                                   "sent_w", "sent_h", "geometry")})
        return _gui_resp(action, ok, stdout, stderr, **extra)

    try:
        if action == "screen_capture":
            shot = _capture_screen(args.get("max_side"), args.get("window"))
            if not shot.get("ok"):
                return _gui_resp(action, False, stderr=shot.get("error", "capture failed"))
            b64 = shot.pop("image_b64")
            return _gui_resp(action, True,
                             (f"Captured window '{shot['window']}' ({len(b64) // 1024} KB, "
                              f"{shot['sent_w']}x{shot['sent_h']})." if shot.get("window") else
                              f"Captured screen ({len(b64) // 1024} KB base64, "
                              f"{shot['orig_w']}x{shot['orig_h']} -> {shot['sent_w']}x{shot['sent_h']})."),
                             image_b64=b64, **{k: v for k, v in shot.items() if k != "ok"})
        if action == "gui_calibrate":
            return done(True, _gui_calibrate())
        if action == "gui_doctor":
            lines, fixes = _gui_doctor()
            return done(True, "\n".join(lines) +
                        ("\n\nTo fix:\n- " + "\n- ".join(fixes) if fixes else "\n\nAll good."))
        if action == "screen_info":
            g = _desktop_geometry(force=True)
            return done(True, json.dumps({"geometry": g, "active_window": _active_window()}),
                        geometry=g)
        if action == "gui_pause":
            with open(GUI_STOP_FLAG, "w") as fh:
                fh.write(f"{_GUI_PAUSE_STAMP} at {datetime.datetime.now():%H:%M:%S} "
                         f"from {request.remote_addr}\n")
            return done(True, "GUI control paused (kill switch engaged).")
        if action == "gui_resume":
            was = _stop_flag_info()
            try:
                os.remove(GUI_STOP_FLAG)
            except FileNotFoundError:
                pass
            _log_activity(f"  kill switch released (was: {was}) by {request.remote_addr}")
            return done(True, f"GUI control resumed (was: {was}).")

        if action == "move_mouse":
            ax, ay = _target_xy(args)
            rc, _, err = _move(ax, ay)
            return done(rc == 0, f"pointer -> ({int(ax)},{int(ay)})", err)

        if action in ("click", "double_click", "right_click"):
            button = "right" if action == "right_click" else str(args.get("button", "left"))
            count = 2 if action == "double_click" else int(args.get("count", 1))
            where = ""
            if "x" in args and "y" in args:
                ax, ay = _target_xy(args)
                rc, _, err = _move(ax, ay)
                if rc != 0:
                    return done(False, stderr=err)
                time.sleep(0.05)
                where = f" at ({int(ax)},{int(ay)})"
            rc, _, err = _click(button, count)
            return done(rc == 0, f"{button} click x{count}{where}" +
                        (_pointer_note() if where else ""), err)

        if action == "drag":
            x1, y1 = _target_xy({"x": args.get("x1"), "y": args.get("y1"),
                                 "coords": args.get("coords", "norm"), "window_id": args.get("window_id")})
            x2, y2 = _target_xy({"x": args.get("x2"), "y": args.get("y2"),
                                 "coords": args.get("coords", "norm"), "window_id": args.get("window_id")})
            _move(x1, y1); time.sleep(0.05)
            _btn_state("left", True); time.sleep(0.08)
            steps = 8                                   # glide so apps register the drag
            for s in range(1, steps + 1):
                _move(x1 + (x2 - x1) * s / steps, y1 + (y2 - y1) * s / steps,
                      closed=(s == steps))
                time.sleep(0.02)
            _btn_state("left", False)
            rc, err = 0, ""
            return done(rc == 0, f"dragged ({int(x1)},{int(y1)}) -> ({int(x2)},{int(y2)})", err)

        if action == "scroll":
            if "x" in args and "y" in args:
                ax, ay = _target_xy(args)
                _move(ax, ay)
            amount = max(1, min(30, int(args.get("amount", 3))))
            direction = str(args.get("direction", "down")).lower()
            dx, dy = {"down": (0, -amount), "up": (0, amount),
                      "left": (-amount, 0), "right": (amount, 0)}.get(direction, (0, -amount))
            if _abs_scroll(dx, dy):
                rc, err = 0, ""
            else:
                rc, _, err = _ydo(["mousemove", "--wheel", "-x", str(dx), "-y", str(dy)])
            return done(rc == 0, f"scrolled {direction} x{amount}", err)

        if action in ("key_combo", "type_text"):
            # Keystrokes go to whatever has focus — make sure that's the right window.
            if args.get("window"):
                pick, wins, ferr = _find_window(args["window"])
                if not pick:
                    return done(False, stderr=ferr + ". Open windows: " +
                                ", ".join(w["name"] for w in wins[:15]))
                _kdo(["windowactivate", pick["id"]])
                time.sleep(0.4)
            aw = _active_window()
            if _is_log_terminal(aw):
                wins, _ = _list_windows()
                return done(False, stderr=(
                    "Keyboard focus is on the 'Jarvis Agent' log terminal, not an app — nothing "
                    "was typed. Pass window=<title> or call focus_window first. Open windows: " +
                    ", ".join(w["name"] for w in wins[:15] if not _is_log_terminal(w))))
            where = f" into window '{aw.get('name', '?')}'"
            if action == "key_combo":
                combo = args.get("keys") or args.get("combo") or ""
                rc, _, err = _key_combo(combo, args.get("repeat", 1))
                return done(rc == 0, f"pressed {combo}{where}", err)
            text = str(args.get("text", ""))
            if not text:
                return done(False, stderr="missing arg text")
            ok, msg = _type_text(text, str(args.get("method", "paste")).lower(),
                                 str(args.get("enter", "")).lower() in ("1", "true", "yes"))
            return done(ok, (msg + where) if ok else "", "" if ok else msg)

        if action == "list_windows":
            wins, err = _list_windows()
            return done(bool(wins), "\n".join(f"- {w['name']}" for w in wins) or "", err,
                        windows=wins)
        if action == "active_window":
            aw = _active_window()
            return done(bool(aw), json.dumps(aw), "" if aw else "kdotool returned nothing",
                        window=aw)
        if action in ("focus_window", "close_window"):
            pick, wins, err = _find_window(args.get("name") or args.get("app") or "")
            if not pick:
                return done(False, stderr=err + ". Open windows: " +
                            ", ".join(w["name"] for w in wins[:15]))
            verb = "windowactivate" if action == "focus_window" else "windowclose"
            rc, _, kerr = _kdo([verb, pick["id"]])
            return done(rc == 0, f"{verb} '{pick['name']}'", kerr)

        if action == "ui_tree":
            items = _ui_snapshot(args.get("app", ""))
            lines = []
            for it in items:
                flags = "".join([" [focused]" if it["focused"] else "",
                                 " [disabled]" if not it["enabled"] else "",
                                 " [checked]" if it["checked"] else ""])
                lines.append(f"[{it['idx']}] {it['role']} '{it['name']}' "
                             f"@norm({it['nx']},{it['ny']}){flags}")
            head = (f"Window: {_UI_SNAP.get('window','')} (app: {_UI_SNAP.get('app','')}) — "
                    f"{len(items)} interactive elements. Use click_element index=N or "
                    f"set_text index=N text=…\n")
            return done(True, head + "\n".join(lines)[:6000],
                        elements=[_public_item(i) for i in items])

        if action == "click_element":
            it = _resolve_element(args)
            with _ATSPI_LOCK:
                how = "" if str(args.get("mouse", "")).lower() in ("1", "true", "yes") \
                      else _press_element(it)
            if how:
                return done(True, f"pressed {it['role']} '{it['name']}' via accessibility ({how})")
            _move(it["sx"], it["sy"]); time.sleep(0.05)       # fallback: real mouse click
            rc, _, err = _click(str(args.get("button", "left")), int(args.get("count", 1)))
            return done(rc == 0, f"mouse-clicked {it['role']} '{it['name']}' at "
                                 f"norm({it['nx']},{it['ny']})", err)

        if action == "set_text":
            it = _resolve_element(args)
            text = str(args.get("text", ""))
            if it["role"] == "password text":
                return done(False, stderr="Refusing to fill a password field — the user enters passwords.")
            with _ATSPI_LOCK:
                try:
                    it["_acc"].queryComponent().grabFocus()
                    it["_acc"].queryEditableText().setTextContents(text)
                    ok = True
                except Exception:
                    ok = False
            if not ok:                                   # fallback: click, select all, paste
                _move(it["sx"], it["sy"]); time.sleep(0.05)
                _click("left", 1); time.sleep(0.1)
                _key_combo("ctrl+a")
                ok, msg = _type_text(text, "paste")
                if not ok:
                    return done(False, stderr=msg)
            if str(args.get("enter", "")).lower() in ("1", "true", "yes"):
                _key_combo("enter")
            return done(True, f"set {it['role']} '{it['name']}' to {len(text)} chars")

        if action == "open_url":
            url = str(args.get("url", "")).strip()
            # browser-internal pages (brave://extensions, chrome://settings, about:...) can't be
            # opened from the command line — focus the browser and type them in the address bar
            if re.match(r"^(brave|chrome|edge|vivaldi|about|chrome-extension):\S*$", url, re.I):
                browser = str(args.get("browser", "")).strip().lower() or url.split(":")[0].lower()
                pick, wins, ferr = _find_window(browser)
                if not pick:
                    return done(False, stderr=f"open the {browser} browser first ({ferr})")
                _kdo(["windowactivate", pick["id"]]); time.sleep(0.4)
                _key_combo("ctrl+l"); time.sleep(0.25)
                ok_t, msg_t = _type_text(url, "paste", True)
                return done(ok_t, f"typed {url} into {pick['name']}'s address bar" if ok_t else "",
                            "" if ok_t else msg_t)
            if not re.match(r"^https?://\S+$", url):
                return done(False, stderr="url must start with http://, https://, or a browser page like brave://extensions")
            browser = str(args.get("browser", "")).strip().lower()
            argv = ["xdg-open", url]
            if browser:
                exe = APP_MAP.get(browser, "")
                if exe and not exe.startswith("steam "):
                    # APP_MAP command (handles flatpak run ..., /usr/bin/vivaldi-stable, etc.)
                    argv = [a for a in shlex.split(exe) if not a.startswith("%")] + [url]
                else:
                    cands = [browser, f"{browser}-stable", f"{browser}-browser"]
                    binp = next((shutil.which(c) for c in cands if shutil.which(c)), None)
                    if not binp:
                        return done(False, stderr=f"browser '{browser}' not found on this PC")
                    argv = [binp, url]
            subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            # focus the browser window so follow-up keys land there
            time.sleep(1.5)
            pick, _, _ = _find_window(browser or "")
            if pick:
                _kdo(["windowactivate", pick["id"]])
            return done(True, f"opened {url}" + (f" in {os.path.basename(argv[0])}" if browser else "")
                        + (f"; window '{pick['name']}' focused" if pick else ""))

        if action == "wait":
            secs = max(0.0, min(15.0, float(args.get("seconds", 1))))
            time.sleep(secs)
            return done(True, f"waited {secs:.1f}s")

    except ValueError as e:
        return _gui_resp(action, False, stderr=str(e))
    except RuntimeError as e:
        return _gui_resp(action, False, stderr=str(e))
    except Exception as e:
        return _gui_resp(action, False, stderr=f"{type(e).__name__}: {e}")
    return _gui_resp(action, False, stderr=f"unhandled GUI action {action}")


app = Flask("jarvis_pc_agent")

# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Visible activity terminal + privileged helpers
# -----------------------------------------------------------------------------
import shutil

_REDACT_KEYS = {"password", "pass", "token", "secret", "credential"}

def _redact(d: dict) -> dict:
    return {k: ("***" if any(s in k.lower() for s in _REDACT_KEYS) else v)
            for k, v in d.items()}

def _log_activity(text: str) -> None:
    try:
        with open(AGENT_LOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("[%H:%M:%S] ") + text.rstrip() + "\n")
    except Exception:
        pass

def _ensure_terminal() -> None:
    """Open a terminal on this PC's display tailing the activity log, so the
    user can watch the agent work. No-op if disabled or already open."""
    if not SHOW_TERMINAL:
        return
    try:
        chk = subprocess.run(["pgrep", "-f", f"tail .*-f {AGENT_LOG}"],
                             capture_output=True, text=True)
        if chk.returncode == 0:
            return                      # already showing
        candidates = [
            ("alacritty", ["alacritty", "--title", "Jarvis Agent",
                           "-e", "tail", "-n", "40", "-f", AGENT_LOG]),
            ("konsole",   ["konsole", "--title", "Jarvis Agent",
                           "-e", "tail", "-n", "40", "-f", AGENT_LOG]),
            ("kitty",     ["kitty", "--title", "Jarvis Agent",
                           "tail", "-n", "40", "-f", AGENT_LOG]),
            ("xterm",     ["xterm", "-T", "Jarvis Agent",
                           "-e", "tail", "-n", "40", "-f", AGENT_LOG]),
        ]
        for binname, argv in candidates:
            if shutil.which(binname):
                open(AGENT_LOG, "a").close()   # make sure the file exists
                prev = _active_window() if GUI_ENABLED else {}
                subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL,
                                 start_new_session=True)
                _log_activity(f"--- Jarvis Agent terminal opened ({binname}) ---")
                # the new terminal grabs keyboard focus — hand it straight back
                # so the agent's next keystrokes don't land in the log viewer
                if prev.get("id"):
                    time.sleep(1.2)
                    _kdo(["windowactivate", prev["id"]])
                return
    except Exception:
        pass                             # never let the viewer break the work

def _sudo(argv: list, input_text: str = None, timeout: int = 120):
    """Run a privileged command NON-interactively (sudo -n). Secrets go through
    stdin (input_text), never onto a command line. Returns (rc, out, err)."""
    try:
        proc = subprocess.run(["sudo", "-n"] + argv, input=input_text,
                              capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout or ""), (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "", "timed out"
    except Exception as e:
        return 1, "", str(e)


def _lan_ip() -> str:
    """Best-effort primary LAN IP of this machine (for building app URLs)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))       # no packets sent; just picks the route
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return HOSTNAME


def _read_tail(path: str, n: int = 800) -> str:
    try:
        with open(path) as lf:
            return lf.read()[-n:]
    except Exception:
        return ""


# Frameworks announce their listening address on startup; match the common ones.
_LISTEN_PATTERNS = [
    re.compile(r"Running on\s+http://([0-9.]+|\[[0-9a-fA-F:]+\]|localhost):(\d+)"),  # Flask/Werkzeug
    re.compile(r"Uvicorn running on\s+http://([0-9.]+|localhost):(\d+)"),           # uvicorn/FastAPI
    re.compile(r"Serving HTTP on\s+([0-9.]+|::)\s+port\s+(\d+)"),                     # http.server
    re.compile(r"https?://([0-9.]+|0\.0\.0\.0|localhost):(\d+)"),                     # generic fallback
]

def _detect_listen_url(log_path: str):
    """Scan the app log for 'listening on host:port' banners.
    Returns (host, port) as strings, preferring a non-loopback host (so a LAN
    link is offered when the app bound all interfaces), or (None, None)."""
    txt = _read_tail(log_path, 4000)
    if not txt:
        return None, None
    loopback = {"127.0.0.1", "localhost", "::1"}
    fallback = None
    for pat in _LISTEN_PATTERNS:
        for m in pat.finditer(txt):
            host = m.group(1).strip("[]")
            port = m.group(2)
            if host not in loopback:
                return host, port          # LAN-reachable address wins
            if fallback is None:
                fallback = (host, port)     # remember loopback in case that's all
    return fallback if fallback else (None, None)


_SUDO_HINT = ("sudo needs a password on this machine. Create "
              "/etc/sudoers.d/jarvis-agent (visudo -f) with:\n"
              f"  {os.getenv('USER') or 'youruser'} ALL=(root) NOPASSWD: /usr/bin/pacman, /usr/bin/tee, "
              "/usr/bin/mkdir, /usr/bin/chmod, /usr/bin/mount, /usr/bin/umount, "
              "/usr/bin/systemctl daemon-reload")

def do_mount_smb(args: dict) -> dict:
    """Permanently mount an SMB share: cifs-utils, credentials file (0600),
    idempotent /etc/fstab entry (nofail + systemd automount), mount now."""
    share = (args.get("share") or "").strip()
    name = (args.get("name") or "").strip()
    username = (args.get("username") or "").strip()
    password = args.get("password") or ""
    uid = str(args.get("uid") or 1000)
    gid = str(args.get("gid") or 1000)

    if not re.match(r"^//[^\s]+/[^\s]+$", share):
        return {"ok": False, "error": "share must look like //host/sharename"}
    if not re.match(r"^[A-Za-z0-9_-]{1,32}$", name):
        return {"ok": False, "error": "name must be a short folder name (letters/digits/_-)"}
    if not username:
        return {"ok": False, "error": "missing arg username"}

    mountpoint = f"/mnt/{name}"
    credfile = f"/etc/jarvis-smb-{name}.cred"
    steps = []

    def fail(step, err):
        if "password is required" in err or "a terminal is required" in err:
            err += "\n" + _SUDO_HINT
        return {"ok": False, "error": f"{step}: {err}", "steps": steps}

    # 1) cifs-utils present?
    if subprocess.run(["pacman", "-Qi", "cifs-utils"], capture_output=True).returncode != 0:
        rc, out, err = _sudo(["pacman", "-S", "--noconfirm", "--needed", "cifs-utils"],
                             timeout=300)
        if rc != 0:
            return fail("install cifs-utils", err or out)
        steps.append("installed cifs-utils")
    else:
        steps.append("cifs-utils already installed")

    # 2) credentials file via stdin -> never on a command line
    cred_content = f"username={username}\npassword={password}\n"
    rc, out, err = _sudo(["tee", credfile], input_text=cred_content)
    if rc != 0:
        return fail("write credentials", err)
    rc, out, err = _sudo(["chmod", "600", credfile])
    if rc != 0:
        return fail("chmod credentials", err)
    steps.append(f"credentials -> {credfile} (0600)")

    # 3) idempotent fstab entry
    fstab_line = (f"{share} {mountpoint} cifs "
                  f"credentials={credfile},uid={uid},gid={gid},iocharset=utf8,"
                  f"file_mode=0664,dir_mode=0775,nofail,x-systemd.automount 0 0")
    have = subprocess.run(["grep", "-qs", f" {mountpoint} ", "/etc/fstab"],
                          capture_output=True)
    if have.returncode == 0:
        steps.append("fstab entry already present - left as is")
    else:
        rc, out, err = _sudo(["tee", "-a", "/etc/fstab"], input_text=fstab_line + "\n")
        if rc != 0:
            return fail("append fstab", err)
        steps.append("fstab entry added (nofail + systemd automount)")

    # 4) mountpoint + mount now
    rc, out, err = _sudo(["mkdir", "-p", mountpoint])
    if rc != 0:
        return fail("mkdir mountpoint", err)
    _sudo(["systemctl", "daemon-reload"])
    rc, out, err = _sudo(["mount", "-a"])
    if rc != 0:
        return fail("mount -a", err)
    steps.append("mounted")

    verify = subprocess.run(["ls", mountpoint], capture_output=True, text=True, timeout=20)
    listing = (verify.stdout or "").strip()[:400]
    return {"ok": True, "mountpoint": mountpoint, "steps": steps,
            "listing": listing or "(empty share or automount pending first access)"}

def do_unmount_smb(args: dict) -> dict:
    mp = (args.get("mountpoint") or "").strip()
    if not re.match(r"^/mnt/[A-Za-z0-9_-]{1,32}$", mp):
        return {"ok": False, "error": "mountpoint must look like /mnt/<name>"}
    rc, out, err = _sudo(["umount", mp])
    if rc != 0:
        return {"ok": False, "error": f"umount: {err or out} "
                "(fstab entry is left in place; it will re-automount on access)"}
    return {"ok": True, "mountpoint": mp,
            "note": "unmounted now; fstab entry kept (nofail), so it can automount again"}


def _authorized(req) -> bool:
    auth = req.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:] == AGENT_TOKEN and bool(AGENT_TOKEN)
    # also allow ?token= for easy curl testing
    return req.args.get("token") == AGENT_TOKEN and bool(AGENT_TOKEN)

@app.before_request
def _gate():
    if request.endpoint == "health":
        return  # health check is open
    if not _authorized(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "host": HOSTNAME})

@app.get("/tools")
def list_tools():
    """Advertise what this machine can do, so the agent knows its options.
    Sends the full command dict (with {arg} placeholders) plus native actions."""
    cmds = dict(ALLOWED_COMMANDS)
    cmds["create_file"] = "(native) writes {content} to {path} safely"  # advertise native action
    cmds["mount_smb"] = ("(native) PERMANENTLY mount an SMB/Samba share. args: "
                         "share=//host/sharename, name=<short folder name>, username, "
                         "password, [uid, gid]. Installs cifs-utils if needed, writes a "
                         "0600 credentials file, adds a boot-safe /etc/fstab entry "
                         "(nofail + systemd automount), and mounts it now at /mnt/<name>.")
    cmds["unmount_smb"] = "(native) unmount a share now. args: mountpoint=/mnt/<name>"
    # --- v10 desktop control (computer use). x,y are NORMALISED 0-1000 over the
    # whole screenshot unless coords="px". Any GUI action accepts
    # screenshot_after=true to return a fresh screenshot of the result.
    gui = {
        "screen_capture": "(native) screenshot of the whole desktop -> image_b64 + geometry",
        "screen_info":    "(native) desktop size/monitors + active window",
        "gui_doctor":     "(native) check ydotool/kdotool/AT-SPI setup and print fixes",
        "ui_tree":        "(native) list clickable elements of the ACTIVE window with indexes. "
                          "args: [app=<name to target a specific app>]. PREFER this over pixel clicks.",
        "click_element":  "(native) press an element from ui_tree. args: index=N | name='Save' "
                          "[role='push button'] [mouse=true to force a real click]",
        "set_text":       "(native) fill a text field from ui_tree. args: index=N | name, text, [enter=true]",
        "click":          "(native) click. args: [x,y (0-1000)], [button=left|right|middle], [count]",
        "double_click":   "(native) double-click at x,y",
        "right_click":    "(native) right-click at x,y",
        "move_mouse":     "(native) move pointer to x,y",
        "drag":           "(native) drag. args: x1,y1,x2,y2",
        "scroll":         "(native) scroll. args: direction=up|down|left|right, [amount=3], [x,y]",
        "type_text":      "(native) type into the focused field. args: text, [enter=true], "
                          "[window=<title substring> to focus first], [method=paste|type]. "
                          "Refuses password fields.",
        "key_combo":      "(native) press keys. args: keys='ctrl+s' | 'enter' | 'alt+tab', "
                          "[window=<title substring>], [repeat]",
        "list_windows":   "(native) list open windows",
        "active_window":  "(native) name + geometry of the focused window",
        "focus_window":   "(native) bring a window to front. args: name (substring of title)",
        "close_window":   "(native) close a window gracefully. args: name",
        "open_url":       "(native) open a URL. args: url, [browser=vivaldi|firefox|chrome...] "
                          "(default browser if omitted)",
        "wait":           "(native) pause for UI to settle. args: seconds (max 15)",
        # gui_pause / gui_resume are USER controls — still callable on /run, but
        # deliberately NOT advertised, so the agent model never toggles them.
    }
    cmds.update(gui)
    return jsonify({
        "ok": True,
        "host": HOSTNAME,
        "commands": cmds,
        "apps": sorted(refresh_app_map().keys()),
    })

LAUNCH_WINDOW_WAIT = float(os.getenv("JARVIS_LAUNCH_WINDOW_WAIT", "10"))

@app.post("/run")
def run():
    """launch_app wrapper: after launching, wait for the app's window, focus it
    and report its title — so the next keystrokes go to the app (not to the log
    terminal) and the brain knows what actually appeared (e.g. a profile picker)."""
    data = request.get_json(force=True, silent=True) or {}
    if (data.get("action") or "").strip() != "launch_app" or not GUI_ENABLED:
        return _run_inner()
    app_name = str((data.get("args") or {}).get("app", "")).strip().lower()
    try:
        before = {w["id"] for w in _list_windows()[0]}
    except Exception:
        before = set()
    resp = _run_inner()
    r_obj, status = (resp if isinstance(resp, tuple) else (resp, 200))
    body = r_obj.get_json(silent=True) or {}
    if not body.get("ok"):
        return resp
    pick, deadline = None, time.time() + LAUNCH_WINDOW_WAIT
    while time.time() < deadline and not pick:
        time.sleep(0.7)
        wins = [w for w in _list_windows()[0] if not _is_log_terminal(w)]
        new = [w for w in wins if w["id"] not in before]
        named = [w for w in wins if app_name and app_name in w["name"].lower()]
        pick = (new or named or [None])[0]
    if pick:
        _kdo(["windowactivate", pick["id"]])
        msg = f"launched; window '{pick['name']}' is open and focused"
    else:
        msg = f"launched, but no window appeared within {LAUNCH_WINDOW_WAIT:.0f}s"
    body["stdout"] = ((body.get("stdout") or "").strip() + "\n" + msg).strip()
    body["window"] = pick or {}
    _log_activity(f"  -> {msg}")
    return jsonify(body), status


def _run_inner():
    """Run an allowlisted command.
    Body: {"action": "<name>", "args": {...}}
    e.g. {"action":"launch_app","args":{"app":"firefox"}}
    """
    data = request.get_json(force=True, silent=True) or {}
    action = (data.get("action") or "").strip()
    args = data.get("args") or {}

    _log_activity(f"ACTION {action}  args={json.dumps(_redact(dict(args)))[:300]}")

    # v10 — desktop-control actions. Handled BEFORE the terminal pop-up (a new
    # window would steal focus mid-task) and BEFORE path sanitising (which would
    # rewrite typed text such as "~/notes").
    if action in GUI_ACTIONS:
        return handle_gui(action, args)

    # show the work: open/refresh the on-screen terminal
    _ensure_terminal()

    # -------------------------------------------------------------------------
    # PATH SANITIZATION: fix LLM path-guessing and expand '~'
    # -------------------------------------------------------------------------
    real_home = os.path.expanduser("~")
    for k, v in list(args.items()):
        if isinstance(v, str):
            # Catch common fake home directories the LLM invents
            v = re.sub(r"^/home/(user|your_username|username|test|guest)(/|$)", f"{real_home}\\2", v)
            # Expand ~ to the real home so shlex.quote doesn't break it
            if v.startswith("~/"):
                v = v.replace("~", real_home, 1)
            args[k] = v

    # -------------------------------------------------------------------------
    # NATIVE FILE CREATION — write code safely (bypasses shell quoting issues)
    # -------------------------------------------------------------------------
    if action == "create_file":
        path = os.path.expanduser(args.get("path", ""))
        content = args.get("content", "")
        if not path:
            return jsonify({"ok": False, "error": "missing arg path"}), 400
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return jsonify({
                "ok": True, "host": HOSTNAME, "action": action, "returncode": 0,
                "stdout": f"Successfully wrote {len(content)} bytes to {path}", "stderr": "",
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e), "action": action}), 500

    if action == "mount_smb":
        result = do_mount_smb(args)
        _log_activity(f"  -> mount_smb ok={result.get('ok')} "
                      f"{result.get('mountpoint','')} {result.get('error','')}"[:300])
        status = 200 if result.get("ok") else 500
        result.update({"host": HOSTNAME, "action": action,
                       "returncode": 0 if result.get("ok") else 1,
                       "stdout": json.dumps({k: v for k, v in result.items()
                                             if k not in ("ok",)}, indent=2)[:2000],
                       "stderr": "" if result.get("ok") else result.get("error", "")})
        return jsonify(result), status

    if action == "unmount_smb":
        result = do_unmount_smb(args)
        _log_activity(f"  -> unmount_smb ok={result.get('ok')} {result.get('error','')}"[:300])
        status = 200 if result.get("ok") else 500
        result.update({"host": HOSTNAME, "action": action,
                       "returncode": 0 if result.get("ok") else 1,
                       "stdout": json.dumps(result, indent=2)[:1000],
                       "stderr": "" if result.get("ok") else result.get("error", "")})
        return jsonify(result), status

    # -------------------------------------------------------------------------
    # NATIVE: list_apps / stop_python_app — driven by the registry so we stop the
    # app that was ACTUALLY launched, even if the model forgot or guessed the dir.
    # -------------------------------------------------------------------------
    # (screen_capture is now handled by handle_gui() above — see _capture_screen)

    if action == "list_apps":
        live = _reg_prune()
        if not live:
            return jsonify({"ok": True, "host": HOSTNAME, "action": action, "returncode": 0,
                            "stdout": "No apps launched by Jarvis are currently running.",
                            "stderr": "", "apps": []})
        lines = [f"- {e['dir']} (pid {e['pid']}, {e.get('url') or 'port '+str(e.get('port'))})"
                 for e in live]
        return jsonify({"ok": True, "host": HOSTNAME, "action": action, "returncode": 0,
                        "stdout": "Running apps:\n" + "\n".join(lines), "stderr": "", "apps": live})

    if action == "stop_python_app":
        live = _reg_prune()
        want_dir = os.path.expanduser(str(args.get("dir", "")).strip()) if args.get("dir") else ""
        stop_all = str(args.get("all", "")).lower() in ("1", "true", "yes")

        # Pick targets: exact dir match -> that app; else if only one is running or
        # 'all' -> everything; else the most recently started (what it just made).
        targets = []
        if want_dir:
            targets = [e for e in live if os.path.normpath(e["dir"]) == os.path.normpath(want_dir)]
        if not targets:
            if stop_all or len(live) == 1:
                targets = live
            elif live:
                targets = [max(live, key=lambda e: e.get("started", 0))]

        if not targets:
            return jsonify({"ok": False, "host": HOSTNAME, "action": action, "returncode": 1,
                            "stdout": "", "stderr": "No Jarvis-launched app is running to stop."}), 200

        stopped = []
        for e in targets:
            pid = e.get("pid")
            killed = False
            try:
                if pid and _pid_alive(pid):
                    os.kill(int(pid), signal.SIGTERM)
                    for _ in range(10):
                        if not _pid_alive(pid):
                            break
                        time.sleep(0.3)
                    if _pid_alive(pid):
                        os.kill(int(pid), signal.SIGKILL)
                    killed = True
            except Exception as ex:
                print(f"[stop] SIGTERM pid {pid} failed: {ex}")
            # backstop: also pkill anything still bound to that venv
            try:
                subprocess.run(f"pkill -f {shlex.quote(e['dir'] + '/venv/bin/python')}",
                               shell=True, timeout=10)
            except Exception:
                pass
            stopped.append({"dir": e["dir"], "pid": pid, "killed": killed})

        # remove stopped apps from the registry
        remaining = [e for e in live if e not in targets]
        _reg_save(remaining)
        names = ", ".join(s["dir"] for s in stopped)
        return jsonify({"ok": True, "host": HOSTNAME, "action": action, "returncode": 0,
                        "stdout": f"Stopped {len(stopped)} app(s): {names}", "stderr": "",
                        "stopped": stopped})

    if action not in ALLOWED_COMMANDS:
        return jsonify({"ok": False, "error": f"action '{action}' is not allowed",
                        "allowed": sorted(ALLOWED_COMMANDS.keys()) + ["create_file"]}), 400

    template = ALLOWED_COMMANDS[action]

    # Special handling for launch_app and close_app: translate names using APP_MAP
    if action in ("launch_app", "close_app"):
        app_name = (args.get("app") or "").strip().lower()
        exec_cmd = refresh_app_map().get(app_name)
        
        if action == "launch_app":
            if not exec_cmd:
                return jsonify({"ok": False, "error": f"app '{app_name}' not in APP_MAP",
                                "apps": sorted(APP_MAP.keys())}), 400
            cmd = template.format(app=shlex.quote(f"exec {exec_cmd}"))
            
        elif action == "close_app":
            if exec_cmd:
                # Extract the real binary name (e.g., "/usr/bin/code" -> "code")
                # Group steam games under "steam" to avoid breaking pkill
                if exec_cmd.startswith("steam "):
                    real_proc = "steam"
                else:
                    real_proc = os.path.basename(exec_cmd.split()[0])
                cmd = template.format(app=shlex.quote(real_proc))
            else:
                # Fallback: if it's not in APP_MAP, just try killing what the LLM guessed
                cmd = template.format(app=shlex.quote(app_name))
    else:
        # Generic placeholder substitution with shell-quoting for safety.
        try:
            safe_args = {k: shlex.quote(str(v)) for k, v in args.items()}
            cmd = template.format(**safe_args) if safe_args else template
        except KeyError as e:
            return jsonify({"ok": False, "error": f"missing arg {e}"}), 400

    # -------------------------------------------------------------------------
    # DETACHED LAUNCH — start long-running servers without blocking/timing out.
    # The process is fully detached (start_new_session) with stdin closed and
    # output redirected to the project's app.log, so it keeps running after we
    # return. We then watch the log for the server's "Running on http://…:PORT"
    # banner (or an early crash) and hand back a real URL + stop instructions.
    # -------------------------------------------------------------------------
    if action in DETACHED_ACTIONS:
        try:
            target_dir = os.path.expanduser(args.get("dir", "")) or real_home
            log_path = os.path.join(target_dir, "app.log")
            try:
                logf = open(log_path, "w")
            except Exception:
                logf = subprocess.DEVNULL
            p = subprocess.Popen(
                cmd, shell=True,
                stdin=subprocess.DEVNULL,
                stdout=logf, stderr=logf,
                start_new_session=True,   # fully detach: survives this request
            )

            # Watch for up to ~10s: either the framework prints its listening
            # banner (Flask/Werkzeug, uvicorn, http.server, etc.) or it crashes.
            port, host_bound = None, None
            deadline = time.time() + 10
            while time.time() < deadline:
                if p.poll() is not None:
                    break                 # crashed — handled below
                host_bound, port = _detect_listen_url(log_path)
                if port:
                    break
                time.sleep(0.4)

            if p.poll() is not None:       # exited during the watch -> crashed
                tail = _read_tail(log_path, 800)
                return jsonify({
                    "ok": False, "host": HOSTNAME, "action": action,
                    "returncode": p.returncode, "stdout": "",
                    "stderr": f"App exited on startup. Log tail:\n{tail}",
                })

            stop_hint = (f"To stop it, use the stop_python_app action with "
                         f"dir={target_dir} (or run: pkill -f {target_dir}/venv/bin/python).")
            if port:
                loopback = {"127.0.0.1", "localhost", "::1"}
                local_url = f"http://localhost:{port}"
                if host_bound and host_bound not in loopback and host_bound not in ("0.0.0.0", "::"):
                    # Werkzeug already printed the real LAN address — use it.
                    lan_url = f"http://{host_bound}:{port}"
                elif host_bound in ("0.0.0.0", "::", None):
                    lan_url = f"http://{_lan_ip()}:{port}"   # bound to all interfaces
                else:
                    lan_url = None                            # loopback-only
                primary = lan_url or local_url
                links = primary + (f" (or {local_url} on this machine)" if lan_url else "")
                stdout = (f"App started (pid {p.pid}) and is listening. "
                          f"Open it at {links}. Logs: {log_path}. {stop_hint}")
                _reg_add({"dir": target_dir, "pid": p.pid, "port": port,
                          "url": primary, "started": time.time()})
                return jsonify({
                    "ok": True, "host": HOSTNAME, "action": action, "returncode": 0,
                    "stdout": stdout, "stderr": "",
                    "url": primary, "local_url": local_url, "port": port, "pid": p.pid,
                })
            # Alive but no banner yet — report honestly, don't claim a URL.
            return jsonify({
                "ok": True, "host": HOSTNAME, "action": action, "returncode": 0,
                "stdout": (f"Started (pid {p.pid}) but I couldn't detect the port within 10s. "
                           f"Check {log_path}, or use check_port to verify. {stop_hint}"),
                "stderr": "", "pid": p.pid,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e), "action": action}), 500

    # -------------------------------------------------------------------------
    # NORMAL (blocking) commands — capped at 30s.
    # -------------------------------------------------------------------------
    try:
        timeout_s = ACTION_TIMEOUTS.get(action, 30)
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_s)
        _log_activity(f"  -> rc={proc.returncode} "
                      f"{(proc.stdout or proc.stderr or '').strip()[:200]}")
        return jsonify({
            "ok": proc.returncode == 0,
            "host": HOSTNAME,
            "action": action,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[:4000],
            "stderr": (proc.stderr or "")[:1000],
        })
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "command timed out", "action": action}), 504
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "action": action}), 500

if __name__ == "__main__":
    print(f"[jarvis_pc_agent] {HOSTNAME} listening on {AGENT_BIND}:{AGENT_PORT}")
    print(f"[jarvis_pc_agent] {len(ALLOWED_COMMANDS)} allowed commands "
          f"(+1 native create_file), {len(APP_MAP)} apps")
    print(f"[jarvis_pc_agent] desktop control: {'ON' if GUI_ENABLED else 'OFF'} "
          f"({len(GUI_ACTIONS)} GUI actions) | kill switch file: {GUI_STOP_FLAG} | "
          f"ydotool socket: {os.environ.get('YDOTOOL_SOCKET', '(not found yet)')}")
    if GUI_ENABLED:
        # Ask apps to expose their accessibility tree (ui_tree / click_element).
        # Firefox & GTK apps honour this after they're (re)started; Chromium
        # browsers (Vivaldi/Chrome/Brave) also need --force-renderer-accessibility.
        try:
            _session_env_defaults()
            r = subprocess.run(["busctl", "--user", "set-property", "org.a11y.Bus", "/org/a11y/bus",
                                "org.a11y.Status", "IsEnabled", "b", "true"],
                               capture_output=True, text=True, timeout=5)
            print(f"[jarvis_pc_agent] accessibility bus: "
                  f"{'enabled' if r.returncode == 0 else 'could not enable: ' + (r.stderr or '').strip()[:120]}")
        except Exception as e:
            print(f"[jarvis_pc_agent] accessibility bus: {e}")
    app.run(host=AGENT_BIND, port=AGENT_PORT, debug=False)
#!/usr/bin/env python3
"""
Jarvis setup wizard — writes/updates .env for you.

    python setup.py              # full guided setup
    python setup.py devices      # only add / change Ollama devices
    python setup.py check        # test every configured service

It never deletes anything you already have in .env: existing values are shown
as defaults (press Enter to keep them), and keys it doesn't ask about are kept.
Only the Python standard library is used, so it runs before `pip install`.
"""
import json
import os
import re
import secrets
import socket
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(HERE, ".env")
EXAMPLE_PATH = os.path.join(HERE, ".env.example")

SLOTS = [  # (prefix, ui slot, description)
    ("DEVICE1", "heavy", "your main GPU PC"),
    ("DEVICE2", "light", "a second GPU PC"),
    ("MAC", "mac", "an Apple Silicon Mac"),
]


# ----------------------------------------------------------------------- .env io
def load_env(path):
    vals = {}
    if not os.path.exists(path):
        return vals
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\s*([A-Z0-9_]+)\s*=(.*)$", line)
        if m:
            v = re.sub(r"\s+#.*$", "", m.group(2)).strip().strip('"').strip("'")
            vals[m.group(1)] = v
    return vals


def save_env(updates):
    """Merge `updates` into .env, keeping layout/comments from the existing file
    (or .env.example the first time)."""
    src = ENV_PATH if os.path.exists(ENV_PATH) else EXAMPLE_PATH
    lines = open(src, encoding="utf-8").read().splitlines() if os.path.exists(src) else []
    seen = set()
    out = []
    for line in lines:
        m = re.match(r"^(\s*)([A-Z0-9_]+)(\s*=)(.*)$", line)
        if m and m.group(2) in updates:
            key = m.group(2)
            comment = re.search(r"(\s+#.*)$", m.group(4))
            val = _quote(updates[key])
            out.append(f"{m.group(1)}{key}{m.group(3)}{val}{comment.group(1) if comment else ''}")
            seen.add(key)
        else:
            out.append(line)
    extra = [k for k in updates if k not in seen]
    if extra:
        out.append("")
        out.append("# --- added by setup.py ---")
        out.extend(f"{k}={_quote(updates[k])}" for k in extra)
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    try:
        os.chmod(ENV_PATH, 0o600)          # secrets: owner read/write only
    except OSError:
        pass


def _quote(v):
    v = "" if v is None else str(v)
    return f'"{v}"' if (" #" in v or v.startswith("#")) else v


# ----------------------------------------------------------------------- prompts
def ask(prompt, default="", secret=False, required=False):
    shown = ("*" * 6 if secret and default else default)
    while True:
        try:
            raw = input(f"  {prompt}" + (f" [{shown}]" if shown else "") + ": ").strip()
        except EOFError:
            raw = ""
        val = raw or default
        if val or not required:
            return val
        print("    (required)")


def yes(prompt, default=True):
    d = "Y/n" if default else "y/N"
    try:
        raw = input(f"  {prompt} [{d}]: ").strip().lower()
    except EOFError:
        raw = ""
    return default if not raw else raw.startswith("y")


def pick(prompt, options, default=""):
    """Choose from a numbered list, or type a value."""
    if not options:
        return ask(prompt, default)
    for i, o in enumerate(options, 1):
        print(f"      {i:>2}. {o}")
    val = ask(prompt + " (number or name)", default)
    if val.isdigit() and 1 <= int(val) <= len(options):
        return options[int(val) - 1]
    return val


def section(title):
    print("\n" + "=" * 70 + f"\n {title}\n" + "=" * 70)


# ----------------------------------------------------------------------- probes
def http_json(url, timeout=4, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def ollama_models(base):
    try:
        data = http_json(base.rstrip("/") + "/api/tags")
        return sorted(m.get("name") for m in data.get("models", []) if m.get("name"))
    except Exception as e:
        print(f"    ! couldn't reach Ollama at {base}: {e}")
        print("      (on that machine set OLLAMA_HOST=0.0.0.0 and restart Ollama — docs/DEVICES.md)")
        return None


def port_open(host, port, timeout=2):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ----------------------------------------------------------------------- steps
def step_basics(env, up):
    section("1/5  Assistant + server")
    up["JARVIS_ASSISTANT_NAME"] = ask("Assistant name", env.get("JARVIS_ASSISTANT_NAME", "Jarvis"))
    up["JARVIS_WAKE_WORD"] = ask("Wake word", env.get("JARVIS_WAKE_WORD", "jarvis")).lower()
    up["JARVIS_TIMEZONE"] = ask("Timezone (e.g. Europe/London)", env.get("JARVIS_TIMEZONE", "UTC"))
    up["JARVIS_DEFAULT_LOCATION"] = ask("Your city, country", env.get("JARVIS_DEFAULT_LOCATION", ""))
    up["JARVIS_HOST_IP"] = ask("LAN IP of THIS machine", env.get("JARVIS_HOST_IP") or lan_ip())
    up["JARVIS_HTTP_PORT"] = ask("Web UI port", env.get("JARVIS_HTTP_PORT", "5006"))
    for key in ("JARVIS_SECRET_KEY", "JARVIS_API_TOKEN", "AGENT_TOKEN"):
        cur = env.get(key, "")
        if not cur or cur.startswith("change-me"):
            up[key] = secrets.token_urlsafe(32)
    print("    generated JARVIS_SECRET_KEY / JARVIS_API_TOKEN / AGENT_TOKEN where missing")


def step_users(env, up):
    section("2/5  Users")
    print("  Each user logs into the web UI with a username + PIN.")
    users = dict(p.split(":", 1) for p in env.get("JARVIS_USERS", "").split(",") if ":" in p)
    names = dict(p.split(":", 1) for p in env.get("JARVIS_USER_NAMES", "").split(",") if ":" in p)
    users = {k: v for k, v in users.items() if not v.startswith("change")}
    if users:
        print("  Existing users: " + ", ".join(users))
    while True:
        if users and not yes("Add another user?", default=False):
            break
        u = ask("Username (letters/numbers)", required=True).lower()
        if not re.fullmatch(r"[a-z0-9_.-]+", u):
            print("    use letters, numbers, . _ - only")
            continue
        pin = ask("PIN / password (6+ chars)", secret=True, required=True)
        if len(pin) < 6:
            print("    too short")
            continue
        users[u] = pin
        names[u] = ask("Name the assistant should call them", u.capitalize())
    up["JARVIS_USERS"] = ",".join(f"{k}:{v}" for k, v in users.items())
    up["JARVIS_USER_NAMES"] = ",".join(f"{k}:{names.get(k, k.capitalize())}" for k in users)
    first = next(iter(users))
    up["JARVIS_DEFAULT_USER"] = ask("Owner (voice/phone chats belong to)", env.get("JARVIS_DEFAULT_USER") or first)
    up["JARVIS_OWNER_PROFILE"] = ask("One line about the owner (optional)", env.get("JARVIS_OWNER_PROFILE", ""))


def step_devices(env, up):
    section("3/5  Local LLM devices (Ollama)")
    print("  Up to three slots: two GPU PCs and one Apple Silicon Mac.")
    print("  Each configured device shows up in the web UI automatically.\n")
    for prefix, slot, desc in SLOTS:
        cur_ip = env.get(f"{prefix}_IP", "")
        label = env.get(f"{prefix}_NAME") or desc
        if not yes(f"Configure slot '{slot}' ({desc})" + (f" — currently {label} @ {cur_ip}" if cur_ip else "") + "?",
                   default=bool(cur_ip) or prefix == "DEVICE1"):
            if cur_ip and yes(f"  Remove {label} from Jarvis?", default=False):
                up[f"{prefix}_IP"] = ""
                up[f"{prefix}_OLLAMA_URL"] = ""
            continue
        print()
        up[f"{prefix}_NAME"] = ask("Display name", env.get(f"{prefix}_NAME") or
                                   {"DEVICE1": "Gaming PC", "DEVICE2": "Second PC", "MAC": "Mac"}[prefix])
        up[f"{prefix}_SPECS"] = ask("Specs shown in UI (e.g. RTX 4070 / 32GB)", env.get(f"{prefix}_SPECS", ""))
        ip = ask("IP address", cur_ip, required=True)
        up[f"{prefix}_IP"] = ip
        url = ask("Ollama URL", env.get(f"{prefix}_OLLAMA_URL") or f"http://{ip}:11434")
        up[f"{prefix}_OLLAMA_URL"] = url
        models = ollama_models(url) or []
        if models:
            print(f"    found {len(models)} models on {up[f'{prefix}_NAME']}")
        if prefix == "MAC":
            keys = [("MAC_FAST_MODEL", "Fast text model (kept loaded)"),
                    ("MAC_VOICE_MODEL", "Voice model (small, kept loaded)"),
                    ("MAC_FAST_CODE_MODEL", "Code model"),
                    ("MAC_BETTER_MODEL", "Better model (big, loaded alone)"),
                    ("MAC_AGENT_MODEL", "Agent model (vision; empty = code model)")]
            up["MAC_SSH_USER"] = ask("Mac username for SSH (purge / GPU limit; optional)", env.get("MAC_SSH_USER", ""))
        else:
            keys = [(f"{prefix}_MODEL", "'Better' text+vision model"),
                    (f"{prefix}_FAST_MODEL", "'Fast' text+vision model"),
                    (f"{prefix}_CODE_MODEL", "'Better' code model"),
                    (f"{prefix}_FAST_CODE_MODEL", "'Fast' code model")]
        for key, what in keys:
            up[key] = pick(what, models, env.get(key, ""))
        up[f"{prefix}_ALIASES"] = ask("Words that mean this device (comma-separated)",
                                      env.get(f"{prefix}_ALIASES") or f"{slot},{up[f'{prefix}_NAME'].lower()}")
        if prefix != "MAC":
            up[f"{prefix}_SSH_USER"] = ask("SSH user on this PC (optional, for copying files)",
                                           env.get(f"{prefix}_SSH_USER", ""))
            print("    Wake-up when it's off (both optional):")
            up[f"{prefix}_WAKE_HA_BUTTON"] = ask("  Home Assistant button entity", env.get(f"{prefix}_WAKE_HA_BUTTON", ""))
            up[f"{prefix}_WOL_MAC"] = ask("  Wake-on-LAN MAC address", env.get(f"{prefix}_WOL_MAC", ""))
        sd = env.get(f"{prefix}_SD_URL") or f"http://{ip}:7860"
        if port_open(ip, 7860) or env.get(f"{prefix}_SD_URL"):
            up[f"{prefix}_SD_URL"] = ask("Stable Diffusion API URL", sd)
        vid = env.get(f"{prefix}_VIDEO_URL") or f"http://{ip}:8188"
        if port_open(ip, 8188) or env.get(f"{prefix}_VIDEO_URL"):
            up[f"{prefix}_VIDEO_URL"] = ask("ComfyUI (video) URL", vid)
            up[f"{prefix}_VIDEO_ENGINE"] = pick("Video engine", ["freevideo", "wan"],
                                                env.get(f"{prefix}_VIDEO_ENGINE", "freevideo"))
        print()


def step_cloud(env, up):
    section("4/5  Cloud models (all optional — Enter to skip)")
    for key, where in [("NVIDIA_API_KEY", "build.nvidia.com — cloud 'Better' + cloud agent"),
                       ("GEMINI_API_KEY", "aistudio.google.com — cloud 'Fast', images"),
                       ("GROQ_API_KEY", "console.groq.com — fast intent router"),
                       ("OPENAI_API_KEY", "platform.openai.com"),
                       ("ELEVENLABS_API_KEY", "elevenlabs.io — voice (or use local Kokoro)")]:
        up[key] = ask(f"{key}  ({where})", env.get(key, ""), secret=True)
    if up.get("ELEVENLABS_API_KEY"):
        up["ELEVENLABS_VOICE_ID"] = ask("ElevenLabs voice id", env.get("ELEVENLABS_VOICE_ID", ""))
    up["KOKORO_TTS_URL"] = ask("Local Kokoro TTS URL (optional, e.g. http://192.168.1.20:8880)",
                               env.get("KOKORO_TTS_URL", ""))


def step_integrations(env, up):
    section("5/5  Integrations (Enter to skip any)")
    if yes("Home Assistant?", default=bool(env.get("HOME_ASSISTANT_URL"))):
        up["HOME_ASSISTANT_URL"] = ask("HA URL", env.get("HOME_ASSISTANT_URL", "http://homeassistant.local:8123"))
        up["HOME_ASSISTANT_TOKEN"] = ask("HA long-lived token", env.get("HOME_ASSISTANT_TOKEN", ""), secret=True)
        up["HA_NOTIFY_SERVICE"] = ask("Phone notify service (e.g. mobile_app_my_phone)", env.get("HA_NOTIFY_SERVICE", ""))
    if yes("Nextcloud (notes/contacts/calendar/mail)?", default=bool(env.get("NC_BASE_URL"))):
        up["NC_BASE_URL"] = ask("Nextcloud LAN URL", env.get("NC_BASE_URL", ""))
        up["NC_PUBLIC_URL"] = ask("Nextcloud public URL (links in chat)", env.get("NC_PUBLIC_URL") or up["NC_BASE_URL"])
        up["NC_USER"] = ask("Nextcloud user", env.get("NC_USER", ""))
        up["NC_APP_PASSWORD"] = ask("Nextcloud app password", env.get("NC_APP_PASSWORD", ""), secret=True)
    if yes("Immich (photo search)?", default=bool(env.get("IMMICH_BASE_URL"))):
        up["IMMICH_BASE_URL"] = ask("Immich URL", env.get("IMMICH_BASE_URL", ""))
        up["IMMICH_API_KEY"] = ask("Immich API key", env.get("IMMICH_API_KEY", ""), secret=True)
    up["FILE_SEARCH_ROOTS"] = ask("Folders to search for files (colon-separated)", env.get("FILE_SEARCH_ROOTS", ""))


# ----------------------------------------------------------------------- check
def check(env):
    section("Checking configured services")
    def line(ok, what, detail=""):
        print(f"  {'OK  ' if ok else 'FAIL'}  {what}" + (f"  — {detail}" if detail else ""))
    for prefix, slot, _ in SLOTS:
        url = env.get(f"{prefix}_OLLAMA_URL") or (f"http://{env[prefix + '_IP']}:11434" if env.get(prefix + "_IP") else "")
        if not url:
            continue
        models = ollama_models(url)
        line(models is not None, f"{env.get(prefix + '_NAME') or slot}: Ollama {url}",
             f"{len(models)} models" if models is not None else "unreachable")
        if models is not None:
            keys = (["MAC_FAST_MODEL", "MAC_VOICE_MODEL", "MAC_BETTER_MODEL"] if prefix == "MAC"
                    else [f"{prefix}_MODEL", f"{prefix}_FAST_MODEL"])
            for k in keys:
                m = env.get(k)
                if m:
                    have = m in models or (m + ":latest") in models
                    line(have, f"    model {m}", "" if have else f"not pulled — run: ollama pull {m}")
        agent_ip = env.get(prefix + "_IP")
        if prefix != "MAC" and agent_ip:
            line(port_open(agent_ip, env.get("AGENT_PORT", "8765")), f"    PC agent {agent_ip}:{env.get('AGENT_PORT', '8765')}",
                 "optional — needed for Agent mode / launching apps")
    if env.get("HOME_ASSISTANT_URL"):
        try:
            http_json(env["HOME_ASSISTANT_URL"].rstrip("/") + "/api/",
                      headers={"Authorization": f"Bearer {env.get('HOME_ASSISTANT_TOKEN', '')}"})
            line(True, "Home Assistant")
        except urllib.error.HTTPError as e:
            line(False, "Home Assistant", f"HTTP {e.code} (check the token)")
        except Exception as e:
            line(False, "Home Assistant", str(e))
    users = [p for p in env.get("JARVIS_USERS", "").split(",") if ":" in p]
    line(bool(users) and not any(p.split(":", 1)[1].startswith("change") for p in users),
         "Users", f"{len(users)} configured" if users else "none — run: python setup.py")
    for key in ("JARVIS_SECRET_KEY", "JARVIS_API_TOKEN", "AGENT_TOKEN"):
        line(bool(env.get(key)) and not env[key].startswith("change"), key)
    print()


# ----------------------------------------------------------------------- main
def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    env = load_env(ENV_PATH) or load_env(EXAMPLE_PATH)
    if mode == "check":
        return check(load_env(ENV_PATH))
    up = {}
    if mode in ("all", "setup"):
        step_basics(env, up)
        step_users(env, up)
        step_devices(env, up)
        step_cloud(env, up)
        step_integrations(env, up)
    elif mode == "devices":
        step_devices(env, up)
    else:
        print(__doc__)
        return
    save_env(up)
    print(f"\nSaved {len(up)} settings to {ENV_PATH} (permissions 600).")
    check(load_env(ENV_PATH))
    print("Start Jarvis with:  python jarvis_server.py   (or: docker compose up -d)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled — nothing was saved.")

# Jarvis — a self-hosted, multi-device AI assistant

Jarvis runs on any always-on Linux box (a Raspberry Pi 5 works well) and
orchestrates the computers you already own: your GPU PCs and your Mac run the
models through **Ollama**, and cloud models step in when you want them.

**What it does**

- **Chat** in a web UI (installable as a phone app) with local or cloud models, file and
  image attachments, web search, memory, and code-aware answers
- **Pick the device per message**: every Ollama machine you configure shows up in the UI with
  Fast/Better modes. Offline PCs are woken through Home Assistant or Wake-on-LAN.
- **Agent mode**: multi-step tasks on your PCs, such as creating projects, running commands,
  launching apps, and (on KDE Wayland) **seeing the screen and using the mouse and keyboard**
- **Images** with Stable Diffusion (A1111/Forge) on any device, or cloud image models
- **Video** with ComfyUI (FreeVideo / Wan 2.2), including long clips stitched from chunks
- **Voice**: wake word on the server, a phone endpoint for shortcuts, and local Kokoro or ElevenLabs TTS
- **Home Assistant**: lights, switches, device state and power, reminders, to-dos, a security alarm
- **Nextcloud** notes, contacts, calendar and mail · **Immich** photo search (including text in photos)
- **WhatsApp** through Twilio · **screen watching** (alerts/logs to chat or your phone) · k3s control

Each feature turns on when you fill in its settings in `.env`; anything left empty stays off.

```
             ┌──────────── your LAN ────────────┐
 phone/PC ─▶│  Jarvis server (Pi / NAS / box)  │──▶ cloud APIs (optional)
 browser     │   ├─ Ollama on PC 1  (slot heavy)│
             │   ├─ Ollama on PC 2  (slot light)│
             │   ├─ Ollama on a Mac (slot mac)  │
             │   ├─ PC agent on each PC         │
             │   └─ Home Assistant / Nextcloud… │
             └──────────────────────────────────┘
```

## Quick start

**1. Get the code and configure it**

```bash
git clone https://github.com/nareshconda/jarvis.git && cd jarvis
python3 setup.py            # guided: users, devices (scans each Ollama for models), keys, integrations
```

The wizard writes `.env` (readable only by you) and then tests every service it configured.
You can also copy `.env.example` to `.env` and edit it by hand. Every setting is
documented in that file.

**2. Run it**

With Docker:

```bash
docker compose up -d
docker compose logs -f
```

Or directly on Debian, Ubuntu or Raspberry Pi OS:

```bash
sudo apt install -y python3-venv portaudio19-dev ffmpeg iputils-ping openssh-client plocate
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python jarvis_server.py
```

Open `http://<server-ip>:5006` and sign in with a user from `JARVIS_USERS`.

**3. Add your devices**

On each machine that should run models:

- Install [Ollama](https://ollama.com) and make it listen on the LAN (`OLLAMA_HOST=0.0.0.0`).
- Pull the models you want, for example `ollama pull qwen3.5:9b`.
- Run `python setup.py devices` on the server.

Reload the web UI and the new device appears. See **[docs/DEVICES.md](docs/DEVICES.md)**.

**4. Optional extras**

| Feature | How |
|---|---|
| Agent mode / launching apps on a PC | [pc_agent/README.md](pc_agent/README.md) |
| Stable Diffusion images | [docs/INTEGRATIONS.md#stable-diffusion](docs/INTEGRATIONS.md#stable-diffusion) |
| Video (ComfyUI) | [comfyui/README.md](comfyui/README.md) |
| Home Assistant, Nextcloud, Immich, WhatsApp, voice | [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) |
| Mac wake-word voice client | [mac/](mac/) |
| Kubernetes (k3s) deployment | [k8s/](k8s/) |

## Configuration files

| File | Purpose |
|---|---|
| `.env` | everything: devices, models, keys, URLs, users (see `.env.example`) |
| `config/home_assistant.json` | spoken names, alarm lights and status devices (copy the `.example`) |
| `config/k3s.json` | deployments Jarvis may restart or scale |
| `config/app_aliases.json` | extra spoken names for apps |

## Security

- Keep the server on your LAN. If you expose it, put it behind HTTPS (`SSL_CERT_FILE`/`SSL_KEY_FILE`
  or a reverse proxy) and use long PINs.
- `.env`, the chat database and memory are in `.gitignore`. Never commit them.
- The PC agent only runs allow-listed actions and needs `AGENT_TOKEN`. Never expose it to the internet.
- Text from web pages, files and screens is treated as data, never as instructions.
- Agent mode will not type into password fields and won't check out or pay. You keep a kill switch.

See [SECURITY.md](SECURITY.md).

## Credits

By Naresh Conda. Started as a fork of [Infatoshi/llama-va](https://github.com/Infatoshi/llama-va) and grew from there.
Started off as a learning Hobby to understand how local AI and homelab works, now publishing this out to the wild as sharing is caring <3
Uses Ollama, Flask, ComfyUI, AUTOMATIC1111/Forge, Kokoro, Home Assistant, Nextcloud and Immich.
"Jarvis" is just the default name. Set `JARVIS_ASSISTANT_NAME` to anything you like.

## License

[MIT](LICENSE)

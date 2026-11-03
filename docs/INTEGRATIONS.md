# Integrations

Every integration is optional. Set its `.env` values to turn it on; restart the server.

## Home Assistant

1. HA → your profile → **Security** → *Long-lived access tokens* → create one.
2. In `.env` set `HOME_ASSISTANT_URL` and `HOME_ASSISTANT_TOKEN`.

   - **Phone push:** set `HA_NOTIFY_SERVICE` to your phone's notify service, e.g.
     `mobile_app_my_phone` (find it under Developer tools → Actions → `notify.`).
   - **Header shortcut:** set `HA_URL_PUBLIC` if you reach HA remotely.

3. Optional: copy `config/home_assistant.example.json` to `config/home_assistant.json` to add
   spoken names, the lights that flash on the security alarm, and the devices in "status report".

Jarvis already fuzzy-matches your HA entities by their friendly names, so "turn on the
desk lamp" works without any aliases. Aliases are for nicknames.

**Jarvis inside HA (Ingress / iframe):** set `JARVIS_TRUSTED_PROXY_IPS` to your Home Assistant
host's IP. Requests that come through HA from that IP are treated as the owner. Everything
else still needs a login.

Reminders, to-dos and the calendar use `HA_REMINDER_TODO`, `HA_TODO_ENTITY` and
`HA_CALENDAR_ENTITY`. Create those lists and calendars in HA first, or change the ids.

## Stable Diffusion

Jarvis uses the AUTOMATIC1111-compatible API, which Forge also provides. Start the web UI with
**`--api --listen`**:

```bash
SD_DIR=~/stable-diffusion-webui-forge ./sd/start_sd.sh
```

The script adds `--api --listen --port 7860` plus VRAM-appropriate flags (`--xformers`,
`--medvram` for 8 GB, Apple Silicon flags on a Mac). Pass extra flags with
`SD_EXTRA_ARGS="--ckpt-dir /data/models"`.

In `.env`:

```
DEVICE1_SD_URL=http://192.168.1.21:7860   # default: http://DEVICE1_IP:7860
SD_HEAVY_MODEL=juggernautXL_v9            # checkpoint name as shown in the SD dropdown
SD_STEPS=28
SD_WIDTH=1024
SD_HEIGHT=1024
```

`SD_LIGHT_MODEL` / `DEVICE2_SD_URL` and `SD_MAC_MODEL` / `MAC_*` do the same for the other slots.

## Video (ComfyUI)

See [../comfyui/README.md](../comfyui/README.md).

## Voice

- **Local text-to-speech:** run [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)
  (`docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu`), then set
  `KOKORO_TTS_URL=http://HOST:8880` and `KOKORO_VOICE`.
- **ElevenLabs:** leave `KOKORO_TTS_URL` empty and set `ELEVENLABS_API_KEY` and
  `ELEVENLABS_VOICE_ID`.
- **Server microphone:** set `MIC_DEVICE_INDEX`. List the devices with
  `python -c "import pyaudio;p=pyaudio.PyAudio();[print(i,p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"`.
- **Phone shortcuts:** POST `{"text": "...", "llm_mode": "mac"}` to `/api/voice/phone` with
  the header `X-Jarvis-Token: <JARVIS_API_TOKEN>`. The reply includes text and an audio URL.
- **Mac wake word:** [../mac/](../mac/) (openWakeWord, then a macOS Shortcut).

## Nextcloud

Create an app password under Settings → Security, then set `NC_BASE_URL` (LAN address),
`NC_PUBLIC_URL` (used for links), `NC_USER` and `NC_APP_PASSWORD`. Mail uses the
`NC_IMAP_*` settings. Notes, contacts and calendar need the Notes app and the built-in
CardDAV/CalDAV.

## Immich (photo search)

Account settings → API keys → create one. Set `IMMICH_BASE_URL` and `IMMICH_API_KEY`,
plus `IMMICH_EXTERNAL_URL` if links should use a public address. Text-in-photo search
needs Immich's OCR feature enabled.

## Files

`FILE_SEARCH_ROOTS=/mnt/data:/mnt/media` searches folders mounted on the server. Use
`plocate` (`sudo updatedb`) for speed. Alternatively, set `FILE_SEARCH_SSH_HOST=user@nas` to
search with plocate on another machine over SSH.

## WhatsApp (Twilio)

1. In Twilio, set up the WhatsApp sandbox or a sender.
2. Set `TWILIO_AUTH_TOKEN`, and `WHATSAPP_ALLOWED_FROM=whatsapp:+44…` (only that number gets replies).
3. Set `NGROK_AUTHTOKEN`. Jarvis starts an ngrok tunnel for port `JARVIS_WHATSAPP_PORT` and
   prints the URL to paste into Twilio's webhook.

## Screen watching

In chat: *"monitor the alerts window on the gaming pc every 2 minutes and notify my phone
about new critical alerts"*. This needs the PC agent and a local vision model
(`SCREEN_WATCH_BRAIN=heavy|light|mac`). Screenshots never leave your network.

## Kubernetes (k3s)

To let Jarvis restart or scale deployments ("restart grafana"), copy
`config/k3s.example.json` to `config/k3s.json` and mount a kubeconfig. Example manifests
to run Jarvis itself on k3s are in [../k8s/](../k8s/).

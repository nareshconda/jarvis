# Adding Ollama devices

Jarvis has three local device slots, plus Cloud. Each slot you configure appears in the
web UI's model picker, in the image and video pickers, and in the Agent brain choices.
A slot you leave empty is hidden.

| Slot | `.env` prefix | Meant for | Special handling |
|---|---|---|---|
| `heavy` | `DEVICE1_` | your strongest GPU PC | wake-up, PC agent, SD, video |
| `light` | `DEVICE2_` | a second (smaller) GPU PC | same as above |
| `mac` | `MAC_` | an Apple Silicon Mac | unified-memory juggling: small models stay pinned; a big model gets the whole Mac, then the small ones are restored |

The slot names are internal. In the UI each device shows the `*_NAME` you give it
(for example "Gaming PC"), and you can refer to it in chat by any of its `*_ALIASES`
("open spotify on the gaming pc").

## 1. On the device: install Ollama and open it to the LAN

**Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl edit ollama
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0"
sudo systemctl restart ollama
```

**macOS**

Install the app from ollama.com, then run:

```bash
launchctl setenv OLLAMA_HOST 0.0.0.0     # then quit + reopen Ollama
```

If you run Ollama with Homebrew services, add `OLLAMA_HOST` to its plist instead.

**Windows**

Install Ollama, then set the system environment variable `OLLAMA_HOST=0.0.0.0`
and restart Ollama.

Allow TCP **11434** through the device's firewall from your server's IP only, then pull models:

```bash
ollama pull qwen3.5:9b          # good all-rounder with vision (~7 GB)
ollama pull llama3.1:8b         # fast
ollama pull qwen2.5-coder:7b    # code
```

From the Jarvis server, check it's reachable with `curl http://DEVICE_IP:11434/api/tags`.

### Which models fit?

| GPU memory | "Better" | "Fast" | Code |
|---|---|---|---|
| 8 GB | `qwen3.5:4b` | `llama3.2:3b` | `qwen2.5-coder:7b` |
| 12 GB | `qwen3.5:9b` | `llama3.1:8b` | `qwen2.5-coder:14b` |
| 16–24 GB | `qwen3.5:27b` (Q4) | `qwen3.5:9b` | `qwen2.5-coder:14b` |
| Mac 24 GB+ | `qwen3.5:27b`, loaded alone | `qwen3.5:9b`, pinned | `qwen2.5-coder:7b` |

## 2. On the server: register the device

```bash
python setup.py devices
```

For each slot, the wizard:

- asks for a display name, specs and the device's IP
- lists the models installed on that device so you can pick Better, Fast and Code
- optionally sets a wake-up method (Home Assistant button and/or Wake-on-LAN MAC)
- detects Stable Diffusion on port 7860 and ComfyUI on port 8188 if they're running

Restart the server and reload the UI.

## Waking a device that's off

When a request needs a PC that doesn't answer pings, Jarvis tries, for up to about 2 minutes:

- `DEVICEn_WAKE_HA_BUTTON`: presses a Home Assistant button, for example a smart plug or an
  "Wake on LAN" button entity
- `DEVICEn_WOL_MAC`: sends a Wake-on-LAN magic packet. Enable WoL in the BIOS. The server
  must be on the same LAN; for a container use `network_mode: host`.

## Mac extras (optional)

These let Jarvis free memory before loading a big model:

1. **Remote Login.** On the Mac: System Settings → General → Sharing → Remote Login.
   Then copy the server's SSH key: `ssh-copy-id you@MAC_IP`.
2. **Passwordless purge and GPU limit.** On the Mac, run `sudo visudo -f /etc/sudoers.d/jarvis`
   and add:

   ```
   you ALL=(root) NOPASSWD: /usr/sbin/purge, /usr/sbin/sysctl iogpu.wired_limit_mb=*
   ```

3. **Settings in `.env`:**
   - Set `MAC_SSH_USER=you`.
   - Optionally set `MAC_GPU_WIRED_MB`, for example `20480` on a 24 GB Mac, so a 27B model
     runs 100% on the GPU.

With these set, a Mac agent task or Better reply does the following:

1. Flushes the pinned models and purges the file cache.
2. Loads the big model alone.
3. When it's done, re-pins the Fast and Voice models.

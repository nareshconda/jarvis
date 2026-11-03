# Video generation (ComfyUI)

Jarvis sends a workflow to a ComfyUI server's `/prompt` API, follows its progress over the
`/ws` socket, then downloads the clip. Two engines are supported per device:

| Engine | Needs | Workflow |
|---|---|---|
| `freevideo` (MiniMax H3 via FreeVideo) | BF16: Apple Silicon or NVIDIA RTX 30xx+ | `workflows/freevideo_workflow_api.json` |
| `wan` (Wan 2.2 14B GGUF) | any NVIDIA 8 GB+ (RTX 20xx works) | `workflows/wan22_workflow_api.json` |

## Wan 2.2 on a Linux PC

1. Install ComfyUI in `~/ComfyUI` with its own venv, plus the custom nodes
   `ComfyUI-GGUF` and `ComfyUI-VideoHelperSuite`.
2. Put the models in ComfyUI's `models/` folders. The file names must match `.env`:
   `WAN_HIGH_GGUF`, `WAN_LOW_GGUF`, `WAN_CLIP`, `WAN_VAE`.
3. Start ComfyUI on the LAN, either directly or as a service:

   ```bash
   ./start_comfyui_wan.sh
   # or as a service:
   cp start_comfyui_wan.sh ~/
   cp comfyui-wan.service ~/.config/systemd/user/
   sudo loginctl enable-linger $USER
   systemctl --user enable --now comfyui-wan
   ```

4. In Jarvis's `.env`:

   ```
   DEVICE2_VIDEO_URL=http://<pc-ip>:8188
   DEVICE2_VIDEO_ENGINE=wan
   ```

## FreeVideo on a Mac or an RTX 30xx+ PC

Install FreeVideo, then expose its raw ComfyUI on the LAN with `start_freevideo.sh` (Mac)
or `start_freevideo_lan.sh` / `freevideo-lan.service` (Linux). Then set `MAC_VIDEO_URL`
(or `DEVICE1_VIDEO_URL`) and `*_VIDEO_ENGINE=freevideo`.

Long videos are made from `VIDEO_CHUNK_SECONDS` clips: the last frame of each clip seeds
the next, and the clips are stitched with ffmpeg.

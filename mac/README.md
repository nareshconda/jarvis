# Mac voice client (optional)

Hands-free voice on a Mac: [openWakeWord](https://github.com/dscripka/openWakeWord) listens
for the wake word locally, then runs a macOS Shortcut that dictates your request, sends it
to Jarvis and plays the reply.

```bash
pip3 install openwakeword onnxruntime sounddevice numpy
export JARVIS_SHORTCUT="Jarvis"                # the name of your Shortcut
export JARVIS_OWW_MODEL="hey_jarvis"           # bundled model; add a custom .onnx path after a comma
python3 jarvis_wake_shortcut.py --test         # runs the Shortcut once
python3 jarvis_wake_shortcut.py                # listen
```

Your Shortcut should:

1. Dictate text.
2. Get Contents of URL: `POST http://JARVIS_SERVER_IP:5006/api/voice/phone`
   - JSON body: `{"text": <dictated>, "llm_mode": "mac"}`
   - header: `X-Jarvis-Token: <JARVIS_API_TOKEN>`
3. Speak or play the `text_for_tts` / audio from the reply.

`jarvis_voice_mac.py` + `run_jarvis_voice.sh` are an all-Python alternative (no Shortcut). To
start one of them at login, edit `com.jarvis.voice.plist` (replace `YOUR_USER`), copy it to
`~/Library/LaunchAgents/`, and run `launchctl load` on it.

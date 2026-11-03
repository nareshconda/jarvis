#!/usr/bin/env python3
"""
jarvis_wake_shortcut.py — openWakeWord wake detection that triggers a macOS
Shortcut.

openWakeWord detects the wake word locally (free, no account/key), then hands
off to your iPhone-quality Shortcut, which does the dictation, transcription,
POST to Jarvis and playback. openWakeWord releases the microphone the instant it
fires, so the Shortcut's "Dictate text" gets a free mic.

v9.1 — supports MULTIPLE wake models at once, so you can run the bundled
"hey_jarvis" together with a CUSTOM "jarvis" model and have EITHER phrase fire.
openWakeWord has no built-in bare-"jarvis" model, so train one (free, synthetic
TTS audio — see the note below) and point JARVIS_OWW_MODEL at it.

Setup on the Mac:
    pip3 install openwakeword onnxruntime sounddevice numpy
    shortcuts list                 # confirm your shortcut's exact name
    # Terminal/python needs Microphone permission (System Settings > Privacy &
    # Security > Microphone). Run once from a GUI session to accept the prompt.

Use ONE or BOTH models (comma-separated — built-in names AND/OR .onnx paths):
    export JARVIS_OWW_MODEL="hey_jarvis"                                  # default
    export JARVIS_OWW_MODEL="hey_jarvis,$HOME/jarvis_assistant/models/jarvis.onnx"
    export JARVIS_OWW_MODEL="$HOME/jarvis_assistant/models/jarvis.onnx"   # bare word only

Train a custom "jarvis" model (free, ~20 min, no recordings):
    openWakeWord's automatic trainer generates synthetic "jarvis" clips with TTS
    and trains an .onnx. Open the repo's automatic_model_training.ipynb in Google
    Colab (github.com/dscripka/openWakeWord), set the target word to "jarvis",
    run all cells, download jarvis.onnx, copy it to ~/jarvis_assistant/models/.

Run:
    python3 jarvis_wake_shortcut.py                 # listen
    python3 jarvis_wake_shortcut.py --list          # list available shortcuts
    python3 jarvis_wake_shortcut.py --test          # run the shortcut once, no mic
"""

import os
import sys
import time
import subprocess
import argparse
import datetime

# ----------------------------------------------------------------- config
SHORTCUT_NAME  = os.getenv("JARVIS_SHORTCUT", "Jarvis")   # exact name from `shortcuts list`
# One or more models, comma-separated: bundled names ("hey_jarvis") and/or paths
# to custom .onnx files. ANY of them firing triggers the Shortcut.
OWW_MODELS     = [m.strip() for m in os.getenv("JARVIS_OWW_MODEL", "hey_jarvis").split(",") if m.strip()]
OWW_THRESHOLD  = float(os.getenv("JARVIS_OWW_THRESHOLD", "0.5"))
# Optional PER-MODEL thresholds, e.g. "hey_jarvis=0.5,jarvis=0.65". A bare
# "jarvis" false-triggers more, so you'll usually want it higher. The key is the
# model's name as openWakeWord reports it (bundled name, or the .onnx filename
# without extension).
OWW_THRESHOLDS = {}
for _pair in os.getenv("JARVIS_OWW_THRESHOLDS", "").split(","):
    if "=" in _pair:
        _k, _v = _pair.split("=", 1)
        try:
            OWW_THRESHOLDS[_k.strip()] = float(_v)
        except ValueError:
            pass
OWW_DEBOUNCE   = int(os.getenv("JARVIS_OWW_DEBOUNCE", "3"))     # consecutive frames to confirm
OWW_COOLDOWN   = float(os.getenv("JARVIS_OWW_COOLDOWN", "2.0")) # seconds before another trigger
SAMPLE_RATE    = 16000
CHUNK          = 1280                                            # 80ms @ 16kHz (openWakeWord frame)
SHORTCUT_BUSY_SECS = float(os.getenv("JARVIS_SHORTCUT_BUSY", "20"))
CHIME          = os.getenv("JARVIS_CHIME", "0") == "1"

LOG_FILE = os.getenv("JARVIS_LOG_FILE",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_wake.log"))
_LOG_FH = None


def _log_fh():
    global _LOG_FH
    if _LOG_FH is None:
        try:
            _LOG_FH = open(LOG_FILE, "a", buffering=1)   # line-buffered for live tail -f
        except Exception:
            _LOG_FH = False
    return _LOG_FH


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    fh = _log_fh()
    if fh:
        try:
            fh.write(line + "\n")
        except Exception:
            pass


def chime():
    if CHIME:
        subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"])


def list_shortcuts():
    r = subprocess.run(["shortcuts", "list"], capture_output=True, text=True)
    print(r.stdout or r.stderr)


def run_shortcut():
    """Launch the Shortcut. `shortcuts run` returns quickly; the Shortcut then
    does its own dictation/transcription/POST/playback."""
    try:
        r = subprocess.run(["shortcuts", "run", SHORTCUT_NAME],
                           capture_output=True, text=True, timeout=SHORTCUT_BUSY_SECS + 30)
        if r.returncode != 0:
            log(f"shortcut '{SHORTCUT_NAME}' returned {r.returncode}: "
                f"{(r.stderr or r.stdout).strip()[:200]}")
        else:
            log(f"shortcut '{SHORTCUT_NAME}' launched")
    except FileNotFoundError:
        log("`shortcuts` CLI not found — is this macOS 12+? (it ships with Shortcuts)")
    except subprocess.TimeoutExpired:
        log(f"shortcut '{SHORTCUT_NAME}' still running after timeout (that's usually fine)")
    except Exception as e:
        log(f"failed to run shortcut '{SHORTCUT_NAME}': {e}")


def _threshold_for(model_key):
    return OWW_THRESHOLDS.get(model_key, OWW_THRESHOLD)


def run_loop():
    import numpy as np
    import sounddevice as sd
    from openwakeword.model import Model

    log(f"loading openWakeWord model(s): {OWW_MODELS}")
    try:
        oww = Model(wakeword_models=OWW_MODELS, inference_framework="onnx")
    except Exception as e:
        log(f"could not load {OWW_MODELS} ({e}); loading all bundled models instead")
        oww = Model(inference_framework="onnx")
    log(f"ready. models: {list(oww.models.keys())} | shortcut='{SHORTCUT_NAME}' "
        f"| threshold={OWW_THRESHOLD} per-model={OWW_THRESHOLDS or '-'} | Ctrl+C to stop")

    consec = 0
    cooldown_until = 0.0
    # Mic-health: every ~5s log peak audio level + best wake score. level=0.0000
    # under launchd but real from Terminal => mic is fed SILENCE (a macOS
    # Microphone-permission/TCC problem, not the wake word).
    _hb_last, _hb_peak, _hb_best = 0.0, 0.0, 0.0

    def open_stream():
        return sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                              dtype="float32", blocksize=CHUNK)

    stream = open_stream()
    stream.start()
    try:
        while True:
            data, _ = stream.read(CHUNK)
            chunk = data[:, 0] if data.ndim > 1 else data

            if time.time() < cooldown_until:
                consec = 0
                continue

            pcm = (chunk * 32767).astype(np.int16)
            preds = oww.predict(pcm)
            # The model whose score most exceeds ITS OWN threshold this frame.
            best_model, best_margin, best_score = None, -1.0, 0.0
            for name, val in preds.items():
                margin = val - _threshold_for(name)
                if margin > best_margin:
                    best_model, best_margin, best_score = name, margin, val

            _hb_peak = max(_hb_peak, float(np.max(np.abs(chunk))) if chunk.size else 0.0)
            _hb_best = max(_hb_best, max(preds.values(), default=0.0))
            _now = time.time()
            if _now - _hb_last >= 5.0:
                log(f"mic-health: peak_level={_hb_peak:.4f} best_score={_hb_best:.2f}"
                    + ("  <-- SILENT MIC (check Microphone permission for this python)"
                       if _hb_peak < 1e-4 else ""))
                _hb_last, _hb_peak, _hb_best = _now, 0.0, 0.0

            # fired this frame if the best model is at/over its own threshold
            fired = best_model is not None and best_margin >= 0
            consec = consec + 1 if fired else 0
            if consec < OWW_DEBOUNCE:
                continue

            consec = 0
            log(f"wake word detected via '{best_model}' (score={best_score:.2f}) "
                f"— handing mic to the Shortcut")
            oww.reset()
            chime()

            # CRITICAL: release the mic so the Shortcut's Dictate Text can use it.
            stream.stop()
            stream.close()

            run_shortcut()

            # Let the Shortcut finish its dictation + reply before we reopen.
            time.sleep(SHORTCUT_BUSY_SECS)

            stream = open_stream()
            stream.start()
            oww.reset()
            cooldown_until = time.time() + OWW_COOLDOWN
    finally:
        try:
            stream.stop(); stream.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="openWakeWord -> macOS Shortcut trigger")
    ap.add_argument("--list", action="store_true", help="list available shortcuts and exit")
    ap.add_argument("--test", action="store_true", help="run the shortcut once (no mic) and exit")
    args = ap.parse_args()

    if args.list:
        return list_shortcuts()
    if args.test:
        log(f"test-running shortcut '{SHORTCUT_NAME}'")
        return run_shortcut()

    log(f"listening for wake word(s) {OWW_MODELS} -> shortcut '{SHORTCUT_NAME}'")
    try:
        run_loop()
    except KeyboardInterrupt:
        log("bye")


if __name__ == "__main__":
    main()

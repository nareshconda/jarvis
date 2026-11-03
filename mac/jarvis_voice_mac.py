#!/usr/bin/env python3
"""
jarvis_voice_mac.py — always-on CLI voice assistant for the Mac mini.

Replaces the Shortcuts + Voice Control route, which cannot work: Voice Control
holds the microphone to hear the trigger phrase, so the shortcut's "Dictate
text" action can never acquire it. This owns the mic itself, so there's no
contention, and it needs no GUI session (runs headless over SSH).

Flow:
    listen -> wake word ("jarvis") -> record until you stop talking
    -> transcribe locally (faster-whisper) -> POST /api/voice/phone
    -> play the reply (server audio, else local Kokoro, else macOS `say`)

Setup on the Mac:
    pip3 install faster-whisper sounddevice numpy
    # Terminal needs mic permission: System Settings > Privacy & Security >
    # Microphone > enable for Terminal. Run once from a GUI session to accept
    # the prompt; afterwards SSH works too.

Run:
    python3 jarvis_voice_mac.py --check          # test connectivity first
    python3 jarvis_voice_mac.py --once "hello"   # one-shot, no mic
    python3 jarvis_voice_mac.py --push-to-talk   # press Enter to talk
    python3 jarvis_voice_mac.py                  # wake-word mode
"""

import os
import sys
import json
import time
import base64
import tempfile
import subprocess
import argparse
import datetime
import urllib.request
import urllib.error
from dotenv import load_dotenv

# ----------------------------------------------------------------- config
load_dotenv()
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")
# Talk to the Pi directly over the LAN — no tunnel, no TLS, lowest latency.
JARVIS_URL   = os.getenv("JARVIS_URL", "http://JARVIS_SERVER_IP:5006/api/voice/phone")
LLM_MODE     = os.getenv("JARVIS_LLM_MODE", "mac")       # mac = always-warm 9B
CHAT_ID      = os.getenv("JARVIS_CHAT_ID", "voice-mac")
WAKE_WORDS   = [w.strip().lower() for w in
                os.getenv("JARVIS_WAKE", "jarvis,hey jarvis, jaavis, jervis, jawvis, jorvus, jargis, jarvish, jovis, jorvis").split(",") if w.strip()]

# Kokoro runs on the Mac. 127.0.0.1 works when this script runs there; the LAN
# address is the fallback so it also works from another machine.
KOKORO_URL     = os.getenv("KOKORO_URL", "http://127.0.0.1:8880/v1/audio/speech")
KOKORO_URL_ALT = os.getenv("KOKORO_URL_ALT", "")
KOKORO_VOICE   = os.getenv("KOKORO_VOICE", "bm_daniel(8)+bm_george(2)")
KOKORO_SPEED   = float(os.getenv("KOKORO_SPEED", "1.2"))

STT_MODEL    = os.getenv("JARVIS_STT_MODEL", "base.en")  # tiny.en / base.en / small.en
SAMPLE_RATE  = 16000
SILENCE_RMS  = float(os.getenv("JARVIS_SILENCE_RMS", "0.012"))  # raise if it cuts you off
SILENCE_SECS = float(os.getenv("JARVIS_SILENCE_SECS", "1.5"))  # pause that ends a command
MAX_CMD_SECS = float(os.getenv("JARVIS_MAX_CMD_SECS", "20"))
CMD_WAIT_SECS = float(os.getenv("JARVIS_CMD_WAIT", "8"))    # time to start talking after the chime
PREROLL_SECS  = float(os.getenv("JARVIS_PREROLL", "0.6"))   # audio kept from before onset (raised to not clip first syllable)
CALIBRATE     = os.getenv("JARVIS_CALIBRATE", "1") == "1"
REQ_TIMEOUT  = int(os.getenv("JARVIS_TIMEOUT", "180"))

_model = None

# Live log file — tail it from another machine while headless:
#   ssh mac 'tail -f ~/jarvis_assistant/jarvis_voice.log'
LOG_FILE = os.getenv("JARVIS_LOG_FILE",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_voice.log"))
_LOG_FH = None


def _log_fh():
    """Append-mode, line-buffered so `tail -f` sees each line immediately."""
    global _LOG_FH
    if _LOG_FH is None:
        try:
            _LOG_FH = open(LOG_FILE, "a", buffering=1)   # buffering=1 -> line-buffered
        except Exception as e:
            print(f"[jarvis-voice] could not open log file {LOG_FILE}: {e}", flush=True)
            _LOG_FH = False
    return _LOG_FH


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)          # stdout (and launchd's StandardOutPath)
    fh = _log_fh()
    if fh:
        try:
            fh.write(line + "\n")   # already line-buffered; visible to tail -f at once
        except Exception:
            pass


# ----------------------------------------------------------------- audio out
def _play_bytes(data, why=""):
    if not data:
        return False
    path = os.path.join(tempfile.gettempdir(), "jarvis_reply.mp3")
    with open(path, "wb") as f:
        f.write(data)
    r = subprocess.run(["afplay", path])
    if r.returncode != 0:
        log(f"afplay failed on {path} (rc={r.returncode})")
        return False
    if why:
        log(f"audio: {why}")
    return True


def _kokoro_bytes(text, url):
    """Ask Kokoro for mp3 bytes. Raises on failure."""
    body = json.dumps({
        "model": "kokoro",
        "input": text,
        "voice": KOKORO_VOICE,
        "response_format": "mp3",
        "speed": KOKORO_SPEED,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _kokoro_say(text):
    """Synthesize locally so the voice matches the server's even when the reply
    came back without audio. Tries localhost, then the Mac's LAN address."""
    if not text:
        return False
    for url in [u for u in (KOKORO_URL, KOKORO_URL_ALT) if u]:
        try:
            data = _kokoro_bytes(text, url)
            if data and _play_bytes(data, f"local Kokoro via {url}"):
                return True
        except Exception as e:
            log(f"Kokoro {url} failed: {e}")
    return False


def speak(text, audio_url=None, audio_b64=None):
    """Server audio first (inline base64 needs no extra request), then its URL,
    then local Kokoro, then macOS `say`."""
    if audio_b64:
        try:
            if _play_bytes(base64.b64decode(audio_b64), "server audio (inline)"):
                return
        except Exception as e:
            log(f"inline audio failed: {e}")

    if audio_url and str(audio_url).startswith("http"):
        try:
            with urllib.request.urlopen(audio_url, timeout=30) as r:
                if _play_bytes(r.read(), f"server audio ({audio_url})"):
                    return
        except Exception as e:
            log(f"audio_url failed: {e}")

    if _kokoro_say(text):
        return

    if text:
        log("audio: macOS say (Kokoro unavailable)")
        subprocess.run(["say", "-v", os.getenv("JARVIS_SAY_VOICE", "Daniel"), text])


def chime():
    subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"])


# ----------------------------------------------------------------- audio in
def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        log(f"loading STT model '{STT_MODEL}' (first run downloads it)…")
        _model = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
        log("STT ready.")
    return _model


def transcribe(samples):
    import numpy as np
    audio = np.asarray(samples, dtype="float32")
    if audio.size == 0:
        return ""
    # beam_size=5 considers more alternatives — significantly better accuracy on
    # short or quiet utterances than the greedy beam_size=1.
    # no_speech_threshold lowered so near-silent frames aren't silently dropped.
    segments, _ = get_model().transcribe(audio, language="en", beam_size=5,
                                         vad_filter=True,
                                         vad_parameters={"min_silence_duration_ms": 300},
                                         no_speech_threshold=0.4,
                                         log_prob_threshold=-1.5)
    return " ".join(s.text for s in segments).strip()


def _rms(chunk):
    import numpy as np
    return float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0


def calibrate(stream, secs=1.0):
    """Measure room noise so onset detection suits this mic, not a guess."""
    global SILENCE_RMS
    import numpy as np
    block = int(SAMPLE_RATE * 0.1)
    levels = []
    for _ in range(int(secs / 0.1)):
        data, _ = stream.read(block)
        levels.append(_rms(data[:, 0] if data.ndim > 1 else data))
    ambient = float(np.median(levels)) if levels else 0.0
    SILENCE_RMS = max(0.006, ambient * 3.0)
    log(f"calibrated: ambient={ambient:.4f} -> speech threshold={SILENCE_RMS:.4f}")


def capture_utterance(stream, max_wait=None, max_secs=MAX_CMD_SECS):
    """Wait for speech to START, then record until the speaker pauses.

    The old fixed-window approach split the wake word across window boundaries
    and, after the chime, returned empty if you didn't speak within 1.2s. This
    waits for onset (up to max_wait, or forever if None), keeps a short
    pre-roll so the first syllable isn't clipped, then records the whole
    utterance.
    """
    import numpy as np
    block = int(SAMPLE_RATE * 0.1)
    pre_n = max(1, int(PREROLL_SECS / 0.1))
    pre, waited = [], 0.0

    # phase 1 — wait for someone to start talking
    while True:
        data, _ = stream.read(block)
        chunk = (data[:, 0] if data.ndim > 1 else data).copy()
        if _rms(chunk) >= SILENCE_RMS:
            pre.append(chunk)
            break
        pre.append(chunk)
        if len(pre) > pre_n:
            pre.pop(0)
        waited += 0.1
        if max_wait is not None and waited >= max_wait:
            return np.zeros(0, dtype="float32")      # nobody spoke

    # phase 2 — record until they stop
    frames, quiet_for, started = list(pre), 0.0, time.time()
    while True:
        data, _ = stream.read(block)
        chunk = (data[:, 0] if data.ndim > 1 else data).copy()
        frames.append(chunk)
        quiet_for = quiet_for + 0.1 if _rms(chunk) < SILENCE_RMS else 0.0
        if quiet_for >= SILENCE_SECS:
            break
        if time.time() - started > max_secs:
            break
    return np.concatenate(frames)


# ----------------------------------------------------------------- jarvis
REQ_RETRIES = int(os.getenv("JARVIS_RETRIES", "3"))
REQ_RETRY_DELAY = float(os.getenv("JARVIS_RETRY_DELAY", "2"))

def ask_jarvis(text):
    payload = json.dumps({"text": text, "llm_mode": LLM_MODE, "chat_id": CHAT_ID}).encode()
    last_err = None
    for attempt in range(1, REQ_RETRIES + 1):
        req = urllib.request.Request(JARVIS_URL, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            log(f"server HTTP {e.code} (attempt {attempt}/{REQ_RETRIES}): {e.read()[:120]!r}")
            return {"reply": f"Server error {e.code}, sir."}  # don't retry 4xx/5xx
        except Exception as e:
            last_err = e
            if attempt < REQ_RETRIES:
                log(f"network error (attempt {attempt}/{REQ_RETRIES}): {e} — retrying in {REQ_RETRY_DELAY}s")
                time.sleep(REQ_RETRY_DELAY)
    log(f"request to {JARVIS_URL} failed after {REQ_RETRIES} attempts: {last_err}")
    return {"reply": "Sorry sir, I couldn't reach the server."}


def handle(text):
    if not text:
        return
    log(f"you: {text}")
    data = ask_jarvis(text)
    reply = (data.get("text_for_tts") or data.get("reply")
             or data.get("plain_reply") or "").strip()
    log(f"jarvis: {reply[:300]}")
    speak(reply, data.get("audio_url"), data.get("audio_base64"))


def strip_wake(text):
    low = text.lower()
    for w in WAKE_WORDS:
        i = low.find(w)
        if i != -1:
            return text[i + len(w):].lstrip(" ,.!?-").strip()
    return ""


def heard_wake(text):
    low = (text or "").lower()
    return any(w in low for w in WAKE_WORDS)


# ----------------------------------------------------------------- self-test
def run_check():
    """Prove each piece works, one at a time, so a failure is obvious."""
    ok = True

    log(f"1/4 Jarvis  -> {JARVIS_URL}")
    data = ask_jarvis("say hello in five words")
    reply = (data.get("text_for_tts") or data.get("reply") or "").strip()
    if reply and "couldn't reach" not in reply:
        log(f"     OK  reply: {reply[:80]!r}")
        log(f"     inline audio: {'yes' if data.get('audio_base64') else 'no'}"
            f" | audio_url: {data.get('audio_url') or 'none'}")
    else:
        ok = False
        log("     FAIL — is jarvisServer running and reachable on this network?")

    for i, url in enumerate((KOKORO_URL, KOKORO_URL_ALT), start=2):
        log(f"{i}/4 Kokoro  -> {url}")
        try:
            b = _kokoro_bytes("Kokoro check.", url)
            if b:
                log(f"     OK  {len(b)} bytes of mp3")
            else:
                ok = False
                log("     FAIL — empty response")
        except Exception as e:
            ok = False
            log(f"     FAIL — {e}")

    log("4/4 playback -> afplay + say")
    if _kokoro_say("Kokoro playback check."):
        log("     OK  heard Kokoro")
    else:
        log("     Kokoro playback unavailable; testing macOS say")
        subprocess.run(["say", "macOS say check"])
    log("check complete — " + ("all good" if ok else "see FAIL lines above"))


# ----------------------------------------------------------------- modes
def run_wake_loop_old():
    import sounddevice as sd
    get_model()
    log(f"listening for {WAKE_WORDS} — Ctrl+C to stop")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        if CALIBRATE:
            calibrate(stream)
        while True:
            audio = capture_utterance(stream)          # waits for you to speak
            if audio.size < SAMPLE_RATE * 0.25:
                continue
            heard = transcribe(audio)
            if not heard:
                continue
            if not heard_wake(heard):
                log(f"(ignored: {heard[:60]!r})")       # shows what it mishears
                continue

            log(f"wake: {heard!r}")
            rest = strip_wake(heard)
            if rest:                       # "jarvis turn on the lights" in one breath
                handle(rest)
                continue

            chime()                        # bare "jarvis" -> now wait for the command
            cmd = capture_utterance(stream, max_wait=CMD_WAIT_SECS)
            if cmd.size < SAMPLE_RATE * 0.25:
                log("no command heard — say the whole thing in one go next time, "
                    "e.g. 'jarvis, what's the weather'")
                continue
            handle(transcribe(cmd))

def run_wake_loop():
    import sounddevice as sd
    import numpy as np
    from openwakeword.model import Model

    log("Initializing openWakeWord engine...")
    try:
        # "hey_jarvis" is a built-in pre-trained model downloaded automatically on first run
        oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    except Exception as e:
        log(f"Failed to initialize openWakeWord: {e}")
        return

    get_model()
    log("Listening for 'hey jarvis' via openWakeWord — Ctrl+C to stop")

    # openWakeWord processes 1280 PCM audio samples (~80ms at 16kHz) per prediction frame
    chunk_size = 1280

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        if CALIBRATE:
            calibrate(stream)

        cooldown_until = 0.0          # don't re-trigger until this timestamp passes
        consec = 0                        # consecutive frames above threshold (de-bounce)
        DEBOUNCE = int(os.getenv("JARVIS_OWW_DEBOUNCE", "3"))  # frames needed to confirm

        while True:
            # Phase 1: Continuous wake-word detection
            data, _ = stream.read(chunk_size)
            chunk = (data[:, 0] if data.ndim > 1 else data)
            pcm = (chunk * 32767).astype(np.int16)
            prediction = oww_model.predict(pcm)
            best_score = max((v for v in prediction.values()), default=0.0)

            # Cooldown: ignore frames that follow a recent trigger
            if time.time() < cooldown_until:
                consec = 0
                continue

            if best_score >= 0.35:
                consec += 1
            else:
                consec = 0

            # De-bounce: require DEBOUNCE consecutive high-scoring frames
            if consec < DEBOUNCE:
                continue

            # Confirmed wake word
            consec = 0
            cooldown_until = time.time() + OWW_COOLDOWN
            oww_model.reset()
            log(f"Wake word detected! (score={best_score:.2f})")
            chime()

            # Phase 2: drain the mic briefly so we don't pick up the chime sound
            drain_secs = 0.25
            for _ in range(int(drain_secs / (chunk_size / SAMPLE_RATE))):
                stream.read(chunk_size)

            # Phase 3: Capture and transcribe the command
            cmd = capture_utterance(stream, max_wait=CMD_WAIT_SECS)
            if cmd.size < SAMPLE_RATE * 0.25:
                log("No command heard — say your command after the chime.")
                continue
            handle(transcribe(cmd))   # speak() inside handle() blocks until audio finishes

            # Phase 4: drain the mic after Jarvis finishes speaking. afplay is
            # synchronous so by the time we reach here the reply audio is done —
            # but room reverb lingers for ~0.5s and would trigger another capture.
            post_drain = float(os.getenv("JARVIS_POST_DRAIN", "0.6"))
            for _ in range(int(post_drain * SAMPLE_RATE / chunk_size)):
                try:
                    stream.read(chunk_size)
                except Exception:
                    break
            oww_model.reset()     # prevent the reverb from scoring as a new wake word

def run_push_to_talk():
    import sounddevice as sd
    get_model()
    log("push-to-talk: press Enter, speak, then pause. Ctrl+C to quit.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        if CALIBRATE:
            calibrate(stream)
        while True:
            try:
                input("\n[Enter to talk] ")
            except EOFError:
                return
            chime()
            text = transcribe(capture_utterance(stream, max_wait=CMD_WAIT_SECS))
            if heard_wake(text):
                text = strip_wake(text) or text
            handle(text)


def main():
    ap = argparse.ArgumentParser(description="Jarvis CLI voice assistant (macOS)")
    ap.add_argument("--check", action="store_true", help="test Jarvis + Kokoro + playback")
    ap.add_argument("--push-to-talk", action="store_true", help="press Enter instead of a wake word")
    ap.add_argument("--once", metavar="TEXT", help="send one message and exit (no mic)")
    args = ap.parse_args()

    log(f"endpoint {JARVIS_URL} | llm_mode={LLM_MODE}")
    if args.check:
        return run_check()
    if args.once:
        return handle(args.once)
    try:
        run_push_to_talk() if args.push_to_talk else run_wake_loop()
    except KeyboardInterrupt:
        log("bye")


if __name__ == "__main__":
    main()

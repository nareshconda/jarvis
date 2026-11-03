"""
surveillance.py  —  JARVIS home surveillance / security mode  (v8.2)
====================================================================
A single-camera worker for the Anker USB webcam that:

  1. Serves a live MJPEG stream + JPEG snapshot (view it on LAN/VPN in any
     browser, or embed in the Jarvis app).
  2. When "security mode" is armed, watches for motion and, on detection:
       - records ~60s of video to a folder,
       - flickers the configured lights red/blue for ~30s (Home Assistant),
       - has Jarvis speak the intruder warning for ~30s (ElevenLabs, Pi speakers),
       - sends an "intruder alert" notification via Home Assistant.

ONE thread owns the camera (USB webcams allow only one reader). That same loop
feeds the stream, runs motion detection, and writes recordings — so nothing
ever fights over /dev/video0.

It reuses Jarvis's own speech + HA functions via init_surveillance(), so the
voice and Home Assistant calls are identical to the rest of your server.

Runtime deps (install on the Pi / in the pod image):
    pip install opencv-python-headless
    # ffmpeg + ALSA already present for Jarvis TTS

Env (all optional — sensible defaults):
    CAM_DEVICE=0                 # or /dev/video0
    CAM_WIDTH=640  CAM_HEIGHT=480  CAM_FPS=20  STREAM_FPS=12
    SECURITY_LIGHTS=light.living_room,light.hallway
    HA_NOTIFY_SERVICE=notify     # e.g. mobile_app_my_phone
    ALARM_SECONDS=30  RECORD_SECONDS=60  SECURITY_COOLDOWN=120
    MOTION_MIN_AREA=1500  MOTION_STREAK=3  MOTION_WARMUP=30
    SURVEILLANCE_DIR=/data/recordings
    ALARM_MESSAGE="Intruder alert, sending footage to owner and alarming is enabled"
"""

import os
import time
import threading
import datetime

try:
    import cv2
    _CV2_OK = True
except Exception as _e:                       # keep Jarvis importable without cv2
    cv2 = None
    _CV2_OK = False
    print(f"[surveillance] OpenCV unavailable ({_e}); camera features disabled. "
          f"Install with: pip install opencv-python-headless")

from flask import Blueprint, Response, jsonify

# --- config -----------------------------------------------------------------
CAM_DEVICE   = os.getenv("CAM_DEVICE", "0")
CAM_WIDTH    = int(os.getenv("CAM_WIDTH", "640"))
CAM_HEIGHT   = int(os.getenv("CAM_HEIGHT", "480"))
CAM_FPS      = int(os.getenv("CAM_FPS", "20"))
STREAM_FPS   = int(os.getenv("STREAM_FPS", "12"))

SECURITY_LIGHTS = [s.strip() for s in os.getenv(
    "SECURITY_LIGHTS", ""
).split(",") if s.strip()]

HA_NOTIFY_SERVICE = os.getenv("HA_NOTIFY_SERVICE", "notify")
ALARM_MESSAGE     = os.getenv(
    "ALARM_MESSAGE", "Intruder alert, sending footage to owner and alarming is enabled")
ALARM_SECONDS    = int(os.getenv("ALARM_SECONDS", "30"))
RECORD_SECONDS   = int(os.getenv("RECORD_SECONDS", "60"))
COOLDOWN_SECONDS = int(os.getenv("SECURITY_COOLDOWN", "120"))

MOTION_MIN_AREA = int(os.getenv("MOTION_MIN_AREA", "1500"))   # changed-pixel count
MOTION_STREAK   = int(os.getenv("MOTION_STREAK", "3"))        # consecutive frames
WARMUP_FRAMES   = int(os.getenv("MOTION_WARMUP", "30"))

REC_DIR = os.getenv("SURVEILLANCE_DIR", "/data/recordings")
try:
    os.makedirs(REC_DIR, exist_ok=True)
except Exception as e:
    print(f"[surveillance] could not create {REC_DIR} ({e}); falling back to ./recordings")
    REC_DIR = os.path.join(os.getcwd(), "recordings")
    os.makedirs(REC_DIR, exist_ok=True)


def _cam_source():
    try:
        return int(CAM_DEVICE)
    except ValueError:
        return CAM_DEVICE


class Surveillance:
    def __init__(self):
        self.speak = None            # injected: play_tts_response(text)  (blocks)
        self.ha_data = None          # injected: ha_service_with_data(domain, service, data)
        self.armed = False
        self.running = False
        self.in_alarm = False
        self._last_alarm = 0.0
        self.latest_jpeg = None
        self.cap = None
        self._prev_gray = None
        self._motion_streak = 0
        self._frame_count = 0
        self._frame_size = (CAM_WIDTH, CAM_HEIGHT)
        self.recording = False
        self._writer = None
        self._rec_path = None
        self._lock = threading.Lock()

    # --- lifecycle ----------------------------------------------------------
    def start(self):
        if not _CV2_OK:
            print("[surveillance] cannot start camera worker: OpenCV missing.")
            return
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        print(f"[surveillance] camera worker started "
              f"(device={CAM_DEVICE}, {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}).")

    def _open_camera(self):
        cap = cv2.VideoCapture(_cam_source())
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        return cap

    def _capture_loop(self):
        fail = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open_camera()
                time.sleep(1.0)
                if not self.cap.isOpened():
                    fail += 1
                    if fail % 10 == 1:
                        print(f"[surveillance] cannot open camera {CAM_DEVICE}; retrying...")
                    time.sleep(2.0)
                    continue
                fail = 0

            ok, frame = self.cap.read()
            if not ok or frame is None:
                fail += 1
                if fail > 5:
                    try: self.cap.release()
                    except Exception: pass
                    self.cap = None
                time.sleep(0.05)
                continue
            fail = 0
            self._frame_count += 1
            self._frame_size = (frame.shape[1], frame.shape[0])

            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok2:
                self.latest_jpeg = buf.tobytes()

            if self.recording and self._writer is not None:
                try: self._writer.write(frame)
                except Exception: pass

            if (self.armed and not self.in_alarm
                    and (time.time() - self._last_alarm) > COOLDOWN_SECONDS):
                if self._detect_motion(frame):
                    self._trigger_alarm()

    # --- motion -------------------------------------------------------------
    def _detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._frame_count < WARMUP_FRAMES or self._prev_gray is None:
            self._prev_gray = gray
            return False
        delta = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        if cv2.countNonZero(thresh) > MOTION_MIN_AREA:
            self._motion_streak += 1
        else:
            self._motion_streak = 0
        return self._motion_streak >= MOTION_STREAK

    # --- alarm --------------------------------------------------------------
    def _trigger_alarm(self):
        with self._lock:
            if self.in_alarm:
                return
            self.in_alarm = True
            self._last_alarm = time.time()
            self._motion_streak = 0
        print("[surveillance] !!! MOTION DETECTED — triggering alarm !!!")
        threading.Thread(target=self._run_alarm, daemon=True).start()

    def _run_alarm(self):
        try:
            self._start_recording(RECORD_SECONDS)               # ~60s footage
            if self.ha_data:                                    # notify owner once
                self.ha_data("notify", HA_NOTIFY_SERVICE, {
                    "title": "JARVIS Security",
                    "message": "Intruder alert - motion detected at home.",
                })
            # lights + speech run together for ALARM_SECONDS
            t_lights = threading.Thread(target=self._flicker_lights,
                                        args=(ALARM_SECONDS,), daemon=True)
            t_voice  = threading.Thread(target=self._speak_loop,
                                        args=(ALARM_SECONDS,), daemon=True)
            t_lights.start(); t_voice.start()
            t_lights.join();  t_voice.join()
        except Exception as e:
            print(f"[surveillance] alarm error: {e}")
        finally:
            self.in_alarm = False
            print("[surveillance] alarm finished; cooldown started "
                  f"({COOLDOWN_SECONDS}s). Recording continues if still active.")

    def _flicker_lights(self, seconds):
        if not self.ha_data or not SECURITY_LIGHTS:
            return
        red, blue = [255, 0, 0], [0, 0, 255]
        end, i = time.time() + seconds, 0
        while time.time() < end:
            self.ha_data("light", "turn_on", {
                "entity_id": SECURITY_LIGHTS,
                "rgb_color": red if i % 2 == 0 else blue,
                "brightness": 255,
            })
            i += 1
            time.sleep(0.5)
        # turn the alarm lights off afterwards (swap for your own restore if preferred)
        self.ha_data("light", "turn_off", {"entity_id": SECURITY_LIGHTS})

    def _speak_loop(self, seconds):
        if not self.speak:
            return
        end = time.time() + seconds
        while time.time() < end:
            try:
                self.speak(ALARM_MESSAGE)        # blocks for ~one utterance
            except Exception as e:
                print(f"[surveillance] TTS error: {e}")
                time.sleep(1.0)

    # --- recording ----------------------------------------------------------
    def _start_recording(self, seconds):
        if not _CV2_OK or self.recording:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._rec_path = os.path.join(REC_DIR, f"intrusion_{ts}.mp4")
        size = self._frame_size or (CAM_WIDTH, CAM_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._rec_path, fourcc, float(CAM_FPS), size)
        self.recording = True
        print(f"[surveillance] recording -> {self._rec_path} ({seconds}s)")
        threading.Timer(seconds, self._stop_recording).start()

    def _stop_recording(self):
        self.recording = False
        w, self._writer = self._writer, None
        if w is not None:
            try: w.release()
            except Exception: pass
        print(f"[surveillance] recording saved: {self._rec_path}")

    # --- stream / status ----------------------------------------------------
    def mjpeg(self):
        delay = 1.0 / max(1, STREAM_FPS)
        while True:
            jpg = self.latest_jpeg
            if jpg is None:
                time.sleep(0.1); continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            time.sleep(delay)

    def status(self):
        cooldown = 0
        if self._last_alarm:
            cooldown = max(0, int(COOLDOWN_SECONDS - (time.time() - self._last_alarm)))
        return {
            "armed": self.armed,
            "running": self.running,
            "in_alarm": self.in_alarm,
            "recording": self.recording,
            "camera_ok": bool(_CV2_OK and self.cap is not None and self.cap.isOpened()),
            "lights": SECURITY_LIGHTS,
            "cooldown_remaining": cooldown,
        }


SURV = Surveillance()


# --- public hooks for jarvisServer ------------------------------------------
def init_surveillance(speak_fn=None, ha_data_fn=None):
    """Wire in Jarvis's own speech + Home Assistant functions."""
    SURV.speak = speak_fn
    SURV.ha_data = ha_data_fn
    print("[surveillance] initialised (speech + Home Assistant wired in).")


def start_surveillance():
    """Start the camera capture / motion worker (call once, on server start)."""
    SURV.start()


# --- Flask blueprint --------------------------------------------------------
surveillance_bp = Blueprint("surveillance", __name__)


@surveillance_bp.route("/api/camera/stream")
def camera_stream():
    if not _CV2_OK:
        return jsonify({"error": "OpenCV not installed on the server"}), 503
    return Response(SURV.mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@surveillance_bp.route("/api/camera/snapshot")
def camera_snapshot():
    if SURV.latest_jpeg is None:
        return jsonify({"error": "no frame yet"}), 503
    return Response(SURV.latest_jpeg, mimetype="image/jpeg")


@surveillance_bp.route("/api/security/arm", methods=["POST"])
def security_arm():
    SURV.armed = True
    print("[surveillance] security mode ARMED.")
    return jsonify({"ok": True, **SURV.status()})


@surveillance_bp.route("/api/security/disarm", methods=["POST"])
def security_disarm():
    SURV.armed = False
    print("[surveillance] security mode DISARMED.")
    return jsonify({"ok": True, **SURV.status()})


@surveillance_bp.route("/api/security/status")
def security_status():
    return jsonify(SURV.status())


@surveillance_bp.route("/api/security/test", methods=["POST"])
def security_test():
    """Fire the full alarm sequence manually (no motion needed) for testing."""
    SURV._trigger_alarm()
    return jsonify({"ok": True, "msg": "alarm test fired"})


@surveillance_bp.route("/camera")
def camera_view():
    return ("<!doctype html><meta name='viewport' "
            "content='width=device-width,initial-scale=1'>"
            "<body style='margin:0;background:#050a0f;text-align:center'>"
            "<img src='/api/camera/stream' "
            "style='max-width:100%;height:auto;display:block;margin:auto'>"
            "</body>")

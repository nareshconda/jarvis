"""
location_service.py — automatic user-location resolution for Jarvis.

Resolves location in four layers, best-available-wins:

    1. Live device location     (most accurate; GPS pushed from your phone/browser
                                 to /api/location)
    2. Forwarded CLIENT IP      (the device hitting the Flask app, when reached
                                 remotely via a tunnel and the IP is a real public
                                 address — rough, city-level, but follows you)
    3. Server (Pi) IP           (zero-setup default; reflects where the SERVER is,
                                 i.e. home — also the result when you're on the LAN)
    4. Hard-coded default        (safety net so nothing ever breaks)

resolve_user_location() returns a "City, Country" string in the same format the
rest of the server already expects, so it is a drop-in replacement for the old
USER_LOCATION constant.
"""

import os
import time
import threading
import ipaddress
import requests
from flask import Blueprint, request, jsonify, has_request_context

# --- config -----------------------------------------------------------------
# Last-resort fallback when GPS push and IP lookup both fail ("City, Country")
DEFAULT_LOCATION = os.getenv("JARVIS_DEFAULT_LOCATION", "London, United Kingdom")
IP_GEO_TTL_SECONDS    = 6 * 60 * 60    # re-check the server's public IP every 6h
LIVE_MAX_AGE_SECONDS  = 60 * 60        # trust a pushed GPS fix for up to 1h
CLIENT_IP_TTL_SECONDS = 30 * 60        # cache a client-IP lookup for 30 min
CLIENT_IP_NEG_TTL     = 2 * 60         # ...but only 2 min if the lookup failed

# Turn the client-IP fallback off entirely with JARVIS_CLIENT_IP_GEO=0
ENABLE_CLIENT_IP_GEO = os.getenv("JARVIS_CLIENT_IP_GEO", "1") != "0"

# Print which layer answered each resolution. Enable with JARVIS_LOCATION_DEBUG=1
LOCATION_DEBUG = os.getenv("JARVIS_LOCATION_DEBUG", "0") == "1"

# Optional shared secret. If set (via env), POST /api/location must send a
# matching  X-Location-Token  header. Leave unset to accept unauthenticated pushes.
LOCATION_TOKEN = os.getenv("JARVIS_LOCATION_TOKEN", "")

# ip-api.com: free, no key. We pull hosting/proxy flags too, so datacenter IPs
# (e.g. Twilio's WhatsApp webhooks, VPNs) can be rejected rather than mistaken
# for the user. (HTTP only on the free tier.)
_IP_FIELDS = "status,city,regionName,country,lat,lon,timezone,proxy,hosting"
def _ip_api_url(ip=""):
    return f"http://ip-api.com/json/{ip}?fields={_IP_FIELDS}"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
HTTP_HEADERS  = {"User-Agent": "JarvisAssistant/1.0 (personal use)"}

# --- state ------------------------------------------------------------------
_lock            = threading.Lock()
_ip_cache        = {"location": None, "lat": None, "lon": None, "tz": None, "ts": 0.0}
_live_fix        = {"location": None, "lat": None, "lon": None, "tz": None, "ts": 0.0}
_client_ip_cache = {}   # ip -> {"data": {...}|None, "ts": float}


# --- shared parser ----------------------------------------------------------
def _parse_ip_api(d):
    """Turn an ip-api.com JSON response into our standard dict, or None.
    Rejects datacenter/proxy IPs so webhook/VPN addresses aren't trusted."""
    if not isinstance(d, dict) or d.get("status") != "success":
        return None
    if d.get("hosting") or d.get("proxy"):
        return None
    country = (d.get("country") or "").strip()
    if not country:
        return None
    city = (d.get("city") or "").strip()
    return {"location": f"{city}, {country}" if city else country,
            "lat": d.get("lat"), "lon": d.get("lon"), "tz": d.get("timezone")}


# --- layer 3: server's own public IP ----------------------------------------
def _fetch_server_ip_location():
    try:
        r = requests.get(_ip_api_url(), headers=HTTP_HEADERS, timeout=5)
        return _parse_ip_api(r.json())
    except Exception as e:
        print(f"[location_service] server IP geolocation failed: {e}")
        return None


def _get_server_ip_location():
    """Cached server-IP geolocation. Refreshes at most every IP_GEO_TTL_SECONDS."""
    now = time.time()
    with _lock:
        if _ip_cache["location"] and (now - _ip_cache["ts"]) < IP_GEO_TTL_SECONDS:
            return dict(_ip_cache)
    fetched = _fetch_server_ip_location()       # network call, outside the lock
    if fetched:
        with _lock:
            _ip_cache.update(fetched)
            _ip_cache["ts"] = now
        return fetched
    with _lock:                                  # on failure, reuse last good value
        return dict(_ip_cache) if _ip_cache["location"] else None


# --- layer 2: forwarded client IP -------------------------------------------
def _is_global_ip(ip):
    try:
        return ipaddress.ip_address(ip).is_global
    except ValueError:
        return False


def _get_client_ip():
    """Best-effort real client IP of the current request."""
    # Cloudflare / some CDNs expose the true client IP directly (not spoofable
    # while you are genuinely behind them):
    for h in ("CF-Connecting-IP", "True-Client-IP"):
        v = (request.headers.get(h) or "").strip()
        if v:
            return v
    # ProxyFix has already de-proxied the socket address by one hop (ngrok case):
    if request.remote_addr:
        return request.remote_addr
    # Last resort, the original client per X-Forwarded-For (best-effort; the
    # left-most entry can be spoofed by a determined client):
    xff = (request.headers.get("X-Forwarded-For") or "").split(",")
    return xff[0].strip() if xff and xff[0].strip() else None


def _prune_client_cache_locked(now):
    if len(_client_ip_cache) > 64:
        keep = sorted(_client_ip_cache, key=lambda k: _client_ip_cache[k]["ts"])[-32:]
        for k in list(_client_ip_cache):
            if k not in keep:
                _client_ip_cache.pop(k, None)
    else:
        for k in [k for k, v in _client_ip_cache.items()
                  if now - v["ts"] > CLIENT_IP_TTL_SECONDS]:
            _client_ip_cache.pop(k, None)


def _geolocate_specific_ip(ip):
    """Cached geolocation of a specific (client) IP. None if it fails or the IP
    is a datacenter/proxy address."""
    now = time.time()
    with _lock:
        ent = _client_ip_cache.get(ip)
        if ent:
            ttl = CLIENT_IP_TTL_SECONDS if ent["data"] else CLIENT_IP_NEG_TTL
            if (now - ent["ts"]) < ttl:
                return dict(ent["data"]) if ent["data"] else None
    data = None
    try:
        r = requests.get(_ip_api_url(ip), headers=HTTP_HEADERS, timeout=5)
        data = _parse_ip_api(r.json())
    except Exception as e:
        print(f"[location_service] client IP geolocation failed: {e}")
        data = None
    with _lock:
        _client_ip_cache[ip] = {"data": data, "ts": now}
        _prune_client_cache_locked(now)
    return dict(data) if data else None


def _client_ip_location():
    """Geolocate the device currently hitting the Flask app, if usable."""
    if not ENABLE_CLIENT_IP_GEO or not has_request_context():
        return None
    ip = _get_client_ip()
    if not ip or not _is_global_ip(ip):     # LAN/loopback/etc. -> not usable
        return None
    return _geolocate_specific_ip(ip)


# --- layer 1: live device fix (pushed to /api/location) ---------------------
def _reverse_geocode(lat, lon):
    """Turn lat/lon into a 'City, Country' string via OpenStreetMap. None on failure."""
    try:
        r = requests.get(NOMINATIM_URL,
                         params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 10},
                         headers=HTTP_HEADERS, timeout=6)
        a = (r.json() or {}).get("address", {})
        city = (a.get("city") or a.get("town") or a.get("village")
                or a.get("municipality") or a.get("county") or "").strip()
        country = (a.get("country") or "").strip()
        if not country:
            return None
        return f"{city}, {country}" if city else country
    except Exception as e:
        print(f"[location_service] reverse geocode failed: {e}")
        return None


def set_live_location(lat, lon, name=None, tz=None):
    """
    Record the user's current device location (called from /api/location).
    If `name` isn't supplied we reverse-geocode the coordinates to a label.
    Returns the stored fix dict, or None if no usable label could be resolved.
    """
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None

    label = (name or "").strip() or _reverse_geocode(lat, lon)
    if not label:
        return None     # don't store an unlabelled fix — we'd rather fall back

    fix = {"location": label, "lat": lat, "lon": lon, "tz": tz, "ts": time.time()}
    with _lock:
        _live_fix.update(fix)
    print(f"[location_service] live location set -> {label} ({lat:.4f}, {lon:.4f})")
    return fix


def _live_if_fresh(max_age=LIVE_MAX_AGE_SECONDS):
    with _lock:
        if _live_fix["location"] and (time.time() - _live_fix["ts"]) < max_age:
            return dict(_live_fix)
    return None


# --- resolution -------------------------------------------------------------
def _best_source(prefer_live=True, live_max_age=LIVE_MAX_AGE_SECONDS):
    """Return the best available source dict, tagged with which layer won:
    {location, lat, lon, tz, source}. `source` is one of
    live_gps / client_ip / server_ip / default."""
    src = None
    if prefer_live:
        live = _live_if_fresh(live_max_age)
        if live:
            live["source"] = "live_gps"
            src = live
    if src is None:
        client = _client_ip_location()          # the device accessing the app
        if client:
            client["source"] = "client_ip"
            src = client
    if src is None:
        server = _get_server_ip_location()       # the Pi itself
        if server:
            server["source"] = "server_ip"
            src = server
    if src is None:
        src = {"location": DEFAULT_LOCATION, "lat": None, "lon": None,
               "tz": None, "source": "default"}
    if LOCATION_DEBUG:
        print(f"[location_service] resolved via {src['source']}: {src['location']}")
    return src


def resolve_user_location(prefer_live=True, live_max_age=LIVE_MAX_AGE_SECONDS):
    """
    Drop-in replacement for the old USER_LOCATION constant.
    Returns a geocodable 'City, Country' string.
    Order: GPS fix -> client IP -> server IP -> hard-coded default.
    """
    src = _best_source(prefer_live, live_max_age)
    return src["location"] if src and src.get("location") else DEFAULT_LOCATION


def resolve_user_coords(prefer_live=True, live_max_age=LIVE_MAX_AGE_SECONDS):
    """
    Same precedence as resolve_user_location() but returns
    (lat, lon, timezone_string) — handy if you later want to feed coordinates
    straight into the weather API or set the clock's timezone dynamically.
    Any field may be None.
    """
    src = _best_source(prefer_live, live_max_age)
    if src and src.get("lat") is not None:
        return src["lat"], src["lon"], src.get("tz")
    return None, None, None


def location_prompt_line():
    """A ready-to-inject system-prompt sentence describing the user's location
    WITH a confidence cue based on how it was determined, so the assistant
    phrases it honestly (confident for GPS, hedged for IP, unknown otherwise)."""
    src = _best_source()
    place, source = src["location"], src["source"]
    if source == "live_gps":
        return (f"The user's current location, from their device GPS, is {place}. "
                "This is accurate — you may state it directly if asked.")
    if source in ("client_ip", "server_ip"):
        return (f"The user's approximate location, inferred from their network IP "
                f"address, is {place}. IP-based location is rough and can be wrong "
                "by a long way (especially on mobile data). If asked where they are, "
                "hedge: say they appear to be near there based on their connection, "
                "not as a definite fact. Don't mention location unless asked.")
    return (f"The user's actual location is unknown — there is no GPS fix or usable "
            f"network location, and {place} is only a configured fallback. If asked "
            "where they are, say you don't currently have their location rather than "
            "guessing.")


# --- Flask endpoint ---------------------------------------------------------
location_bp = Blueprint("location", __name__)


@location_bp.route("/api/location", methods=["GET", "POST"])
def api_set_location():
    """
    Receive a live location fix from your phone (Tasker / iOS Shortcuts) or the
    browser. Accepts JSON body, form data, or query string:
        lat, lon   (required)
        name       (optional 'City, Country' string to skip reverse geocoding)
    If JARVIS_LOCATION_TOKEN is set, an X-Location-Token header must match.
    """
    if LOCATION_TOKEN and request.headers.get("X-Location-Token", "") != LOCATION_TOKEN:
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    src  = request.get_json(silent=True) or request.form or request.args
    lat  = src.get("lat")
    lon  = src.get("lon")
    name = src.get("name")

    fix = set_live_location(lat, lon, name=name)
    if not fix:
        return jsonify({"ok": False, "error": "could not resolve location"}), 400
    return jsonify({"ok": True, "location": fix["location"],
                    "lat": fix["lat"], "lon": fix["lon"]})


def location_diagnostics():
    """Everything needed to see WHY a given location was chosen. Most useful
    when fetched within a request (so the client-IP fields are populated)."""
    src = _best_source()
    diag = {
        "resolved_location": src["location"],
        "source": src["source"],     # live_gps | client_ip | server_ip | default
        "coords": {"lat": src.get("lat"), "lon": src.get("lon"), "tz": src.get("tz")},
        "client_ip_geo_enabled": ENABLE_CLIENT_IP_GEO,
        "default_fallback": DEFAULT_LOCATION,
    }

    with _lock:
        lf = dict(_live_fix)
    if lf["location"]:
        age = time.time() - lf["ts"]
        diag["live_fix"] = {"location": lf["location"], "lat": lf["lat"],
                            "lon": lf["lon"], "age_seconds": round(age),
                            "fresh": age < LIVE_MAX_AGE_SECONDS}
    else:
        diag["live_fix"] = None      # no GPS has ever been pushed this run

    if has_request_context():
        cip = _get_client_ip()
        is_pub = _is_global_ip(cip) if cip else False
        diag["request"] = {
            "client_ip": cip,
            "client_ip_is_public": is_pub,
            "client_ip_geo": _geolocate_specific_ip(cip) if (cip and is_pub) else None,
            "raw_remote_addr": request.remote_addr,
            "X-Forwarded-For": request.headers.get("X-Forwarded-For"),
            "CF-Connecting-IP": request.headers.get("CF-Connecting-IP"),
            "served_over_https": request.is_secure,   # browser GPS needs this True
        }

    diag["server_ip_geo"] = _get_server_ip_location()
    return diag


@location_bp.route("/api/location/debug", methods=["GET"])
def api_location_debug():
    return jsonify(location_diagnostics())


if __name__ == "__main__":
    # quick smoke test:  python location_service.py
    print("resolve_user_location() ->", resolve_user_location())
    print("resolve_user_coords()   ->", resolve_user_coords())
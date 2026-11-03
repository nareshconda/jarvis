#!/usr/bin/env python3
"""
nextcloud_service.py — JARVIS <-> Nextcloud bridge (v9.0)

Talks to Nextcloud over its STANDARD protocols so nothing has to be hacked on
the Nextcloud side:
    Contacts  -> CardDAV   (/remote.php/dav/addressbooks/...)
    Calendar  -> CalDAV    (/remote.php/dav/calendars/...)   read + write
    Notes     -> Notes REST API (/index.php/apps/notes/api/v1/notes)
    Mail      -> IMAP directly to the underlying mail account

Everything is read-mostly and reaches personal data, so callers should keep it
on a LOCAL model tier. Passwords are intentionally NOT implemented.

Config (env; put in the JARVIS .env):
    NC_BASE_URL        e.g. http://192.168.1.20:8080  (LAN address of your Nextcloud)
    NC_USER            your Nextcloud username
    NC_APP_PASSWORD    Settings -> Security -> Create app password (NOT your login pw)

    # Mail (IMAP) — optional, only needed for the mail features
    NC_IMAP_HOST       e.g. imap.gmail.com
    NC_IMAP_PORT       993
    NC_IMAP_USER       full email address
    NC_IMAP_PASSWORD   an app password for that mailbox
"""

import os
import re
import json
import html
import uuid
import imaplib
import email
import datetime
from email.header import decode_header, make_header
import requests
from urllib.parse import quote
from requests.auth import HTTPBasicAuth

# ----------------------------------------------------------------- config
# NC_BASE_URL  -> used for API CALLS (JARVIS -> Nextcloud, server-side). Point this
#                 at the LAN address so calls skip any reverse proxy / tunnel and keep
#                 working if the internet is down.
# NC_PUBLIC_URL-> used for LINKS shown in chat (your browser -> Nextcloud). Must stay
#                 the public domain, or links break when you're away from home.
# Config is read from the environment. IMPORTANT: the server imports this module
# near the top of the file, which can happen BEFORE load_dotenv() runs — in that
# case every value would be empty and nc_configured() would wrongly report "not
# configured". The server therefore calls reload_config() right after loading
# .env, which re-reads everything below.
NC_BASE_URL     = ""
NC_PUBLIC_URL   = ""
NC_USER         = ""
NC_APP_PASSWORD = ""
NC_VERIFY_SSL   = True
NC_IMAP_HOST     = ""
NC_IMAP_PORT     = 993
NC_IMAP_USER     = ""
NC_IMAP_PASSWORD = ""
NC_TIMEOUT        = 45
NC_NOTES_TIMEOUT  = 90     # Notes list can be slow on big libraries
NC_NOTES_TITLES_ONLY = False
NC_REMINDER_CALENDAR = ""  # slug/displayname of the task list iPhone Reminders syncs


_MAX = 8   # max results returned to the chat


def reload_config():
    """(Re-)read every setting from os.environ. Safe to call repeatedly; the
    server calls it after load_dotenv() so import order can't break config."""
    global NC_BASE_URL, NC_PUBLIC_URL, NC_USER, NC_APP_PASSWORD, NC_VERIFY_SSL
    global NC_IMAP_HOST, NC_IMAP_PORT, NC_IMAP_USER, NC_IMAP_PASSWORD, NC_TIMEOUT
    global NC_NOTES_TIMEOUT, NC_NOTES_TITLES_ONLY, NC_REMINDER_CALENDAR

    def _s(name, default=""):
        return (os.getenv(name) or default).strip()

    NC_BASE_URL     = _s("NC_BASE_URL", "").rstrip("/")
    NC_PUBLIC_URL   = _s("NC_PUBLIC_URL").rstrip("/") or NC_BASE_URL
    NC_USER         = _s("NC_USER")
    NC_APP_PASSWORD = _s("NC_APP_PASSWORD")
    NC_VERIFY_SSL   = _s("NC_VERIFY_SSL", "1").lower() not in ("0", "false", "no", "off")
    if not NC_VERIFY_SSL:
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

    NC_IMAP_HOST     = _s("NC_IMAP_HOST")
    NC_IMAP_PORT     = int(_s("NC_IMAP_PORT", "993") or 993)
    NC_IMAP_USER     = _s("NC_IMAP_USER")
    NC_IMAP_PASSWORD = _s("NC_IMAP_PASSWORD")
    NC_TIMEOUT       = int(_s("NC_TIMEOUT", "45") or 45)
    NC_NOTES_TIMEOUT = int(_s("NC_NOTES_TIMEOUT", str(max(90, NC_TIMEOUT * 2))) or 90)
    NC_NOTES_TITLES_ONLY = _s("NC_NOTES_TITLES_ONLY", "0").lower() in ("1", "true", "yes", "on")
    NC_REMINDER_CALENDAR = _s("NC_REMINDER_CALENDAR")
    return {"base": NC_BASE_URL, "public": NC_PUBLIC_URL,
            "user_set": bool(NC_USER and NC_APP_PASSWORD),
            "imap_set": bool(NC_IMAP_HOST and NC_IMAP_USER and NC_IMAP_PASSWORD)}


# read once at import (correct when .env is already loaded); the server calls
# reload_config() again afterwards to be certain.
reload_config()


def nc_configured() -> bool:
    return bool(NC_BASE_URL and NC_USER and NC_APP_PASSWORD)

def _auth():
    return HTTPBasicAuth(NC_USER, NC_APP_PASSWORD)

def _dav(path):
    return f"{NC_BASE_URL}/remote.php/dav/{path.lstrip('/')}"

def _score(query, text):
    """Simple relevance: word-overlap + substring, 0..1."""
    q = (query or "").lower(); t = (text or "").lower()
    if not q:
        return 0.0
    if q in t:
        return 1.0
    qw = {w for w in re.findall(r"[a-z0-9@.]+", q) if len(w) > 1}
    tw = {w for w in re.findall(r"[a-z0-9@.]+", t)}
    return (len(qw & tw) / len(qw)) if qw else 0.0


# ============================================================ CONTACTS (CardDAV)
def _carddav_report():
    """Pull all vCards from the user's default address book via a CardDAV REPORT."""
    url = _dav(f"addressbooks/users/{NC_USER}/contacts/")
    body = ('<?xml version="1.0" encoding="utf-8" ?>'
            '<c:addressbook-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:carddav">'
            '<d:prop><d:getetag/><c:address-data/></d:prop></c:addressbook-query>')
    r = requests.request("REPORT", url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                         headers={"Depth": "1", "Content-Type": "application/xml"},
                         data=body)
    r.raise_for_status()
    # vCards are embedded in the multistatus XML; pull each BEGIN:VCARD..END:VCARD
    return re.findall(r"BEGIN:VCARD.*?END:VCARD", r.text, re.S)

def _clean(val):
    """vCards come out of the CardDAV XML with CRLF encoded as the literal entity
    &#13; (and &#10;), which otherwise shows verbatim in the UI and breaks deep
    links. Decode XML/HTML entities first, then strip the CR/LF and stray escapes."""
    val = html.unescape(val or "")        # &#13; -> \r, &amp; -> &, etc.
    return val.replace("\r", "").replace("\n", "").replace("\\,", ",").strip()

def _ics_escape(val):
    """Escape a TEXT value for iCalendar (VEVENT/VTODO). Unescaped commas,
    semicolons or newlines in an address/summary can make a CalDAV server
    mis-parse the component, so escape them per RFC 5545."""
    return (str(val or "")
            .replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
            .replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", ""))

def _parse_vcard(v):
    def field(tag):
        m = re.search(rf"^{tag}[^:]*:(.+)$", v, re.M | re.I)
        return _clean(m.group(1)) if m else ""
    fn = field("FN")
    tels = [_clean(t) for t in re.findall(r"^TEL[^:]*:(.+)$", v, re.M | re.I)]
    emails = [_clean(e) for e in re.findall(r"^EMAIL[^:]*:(.+)$", v, re.M | re.I)]
    uid = field("UID")
    return {"name": fn, "tels": [t for t in tels if t],
            "emails": [e for e in emails if e], "uid": uid}

def search_contacts(query, limit=3):
    """Return the best-matching contacts, MERGED by person. Nextcloud often holds
    several vCards for one person (phone-only, email-only, synced copies); we
    merge them into one entry so the answer is 'the' contact, not five fragments."""
    cards = [_parse_vcard(v) for v in _carddav_report()]
    scored = []
    for c in cards:
        blob = " ".join([c["name"]] + c["tels"] + c["emails"])
        sc = _score(query, blob)
        # a direct name hit should outrank an incidental email/number match
        if _score(query, c["name"]) >= 0.9:
            sc += 0.5
        if sc > 0.1:
            scored.append((sc, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    # If the query is a FULL name (>=2 words) and some contact's name contains
    # ALL of those words, restrict to those exact-person matches — don't also
    # dump every other 'John'. A single-word query ('john') is unrestricted.
    qwords = {w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) > 1}
    if len(qwords) >= 2:
        def _covers_all(name):
            nw = {w for w in re.findall(r"[a-z0-9]+", (name or "").lower())}
            return qwords.issubset(nw)
        exact = [(sc, c) for sc, c in scored if _covers_all(c["name"])]
        if exact:
            scored = exact

    merged = {}      # name(lower) -> entry
    order = []
    for sc, c in scored:
        key = (c["name"] or "").lower().strip()
        if not key:
            continue
        if key not in merged:
            merged[key] = {"name": c["name"], "tels": [], "emails": [],
                           "uid": c["uid"], "score": sc}
            order.append(key)
        e = merged[key]
        for t in c["tels"]:
            if t not in e["tels"]:
                e["tels"].append(t)
        for m in c["emails"]:
            if m not in e["emails"]:
                e["emails"].append(m)
        if not e["uid"] and c["uid"]:
            e["uid"] = c["uid"]

    out = [merged[k] for k in order][:limit]
    for c in out:
        # confirmed-working format: /apps/contacts/All%20contacts[/<uid>]
        if c["uid"]:
            c["url"] = f"{NC_PUBLIC_URL}/apps/contacts/All%20contacts/{quote(c['uid'], safe='')}"
        else:
            c["url"] = f"{NC_PUBLIC_URL}/apps/contacts/All%20contacts"
    return out


def _carddav_cards():
    """Return [(href, vcard_text), ...] for every card — the href is needed to
    PUT an edited vCard back. (search_contacts drops hrefs; edits need them.)"""
    url = _dav(f"addressbooks/users/{NC_USER}/contacts/")
    body = ('<?xml version="1.0" encoding="utf-8" ?>'
            '<c:addressbook-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:carddav">'
            '<d:prop><d:getetag/><c:address-data/></d:prop></c:addressbook-query>')
    r = requests.request("REPORT", url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                         headers={"Depth": "1", "Content-Type": "application/xml"}, data=body)
    r.raise_for_status()
    out = []
    # Split on <response> (namespace-agnostic); each block has one href + one vCard.
    for block in re.split(r"(?i)<(?:\w+:)?response[\s>]", r.text)[1:]:
        href_m = re.search(r"(?i)<(?:\w+:)?href>([^<]+)</(?:\w+:)?href>", block)
        vc_m = re.search(r"(BEGIN:VCARD.*?END:VCARD)", block, re.S)
        if href_m and vc_m:
            out.append((html.unescape(href_m.group(1).strip()),
                        html.unescape(vc_m.group(1)).replace("\r", "")))
    return out


def update_contact_add(name, email=None, tel=None):
    """Add an EMAIL and/or TEL to the best-matching contact's vCard and PUT it
    back via CardDAV. Returns (ok, message, contact_name)."""
    if not (email or tel):
        return False, "nothing to add", None
    cards = _carddav_cards()
    best, best_sc = None, 0.0
    for href, vc in cards:
        m = re.search(r"^FN[^:]*:(.+)$", vc, re.M | re.I)
        fn = _clean(m.group(1)) if m else ""
        if not fn:
            continue
        sc = _score(name, fn) + (0.5 if _score(name, fn) >= 0.9 else 0.0)
        if sc > best_sc:
            best_sc, best = sc, (href, vc, fn)
    if not best or best_sc < 0.5:
        return False, f"no contact matching '{name}'", None
    href, vc, fn = best

    new_lines, body_low = [], vc.lower()
    if email and email.lower() not in body_low:
        new_lines.append(f"EMAIL;TYPE=INTERNET:{email}")
    if tel and re.sub(r'[^0-9+]', '', tel) not in re.sub(r'[^0-9+]', '', body_low):
        new_lines.append(f"TEL;TYPE=CELL:{tel}")
    if not new_lines:
        return True, f"{fn} already has that detail", fn

    # Insert the new property line(s) just before END:VCARD, keep everything else
    # (UID/REV/etc.) so the server treats it as an edit of the same contact.
    out_lines = []
    for ln in vc.splitlines():
        if ln.strip().upper() == "END:VCARD":
            out_lines.extend(new_lines)
        out_lines.append(ln)
    new_vc = "\r\n".join(out_lines) + "\r\n"

    put_url = (NC_BASE_URL.rstrip("/") + href) if href.startswith("/") else href
    r = requests.put(put_url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                     headers={"Content-Type": "text/vcard; charset=utf-8"},
                     data=new_vc.encode("utf-8"))
    r.raise_for_status()
    added = []
    if email and email.lower() not in body_low: added.append(f"email {email}")
    if tel: added.append(f"number {tel}")
    return True, f"Added {' and '.join(added)} to {fn}", fn


# ============================================================ NOTES (REST API)
def _notes_all(titles_only=None):
    # Notes is a plain REST API (NOT an OCS app), so it does not need the
    # OCS-APIRequest header. Uses a longer timeout than the rest of Nextcloud
    # because listing a large note library can be slow. Pass titles_only=True to
    # omit note bodies (exclude=content) for a MUCH faster list; None uses the
    # NC_NOTES_TITLES_ONLY env default.
    if titles_only is None:
        titles_only = NC_NOTES_TITLES_ONLY
    url = f"{NC_BASE_URL}/index.php/apps/notes/api/v1/notes"
    params = {"exclude": "content"} if titles_only else None
    r = requests.get(url, auth=_auth(), timeout=NC_NOTES_TIMEOUT, verify=NC_VERIFY_SSL,
                     params=params, headers={"Accept": "application/json"})
    r.raise_for_status()
    return r.json()

def get_note(note_id):
    """Fetch a single note (with its full content) — used to lazy-load a note's
    body when it's opened, so the list feed can stay titles-only and fast."""
    url = f"{NC_BASE_URL}{_NOTES_API}/{note_id}"
    r = requests.get(url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                     headers={"Accept": "application/json"})
    r.raise_for_status()
    n = r.json()
    return {"id": n.get("id"), "title": _clean(n.get("title") or ""),
            "content": _clean(n.get("content") or ""),
            "favorite": bool(n.get("favorite")),
            "url": f"{NC_PUBLIC_URL}/apps/notes/note/{n.get('id')}"}

def search_notes(query):
    notes = _notes_all()
    scored = []
    for n in notes:
        s = max(_score(query, n.get("title", "")),
                _score(query, (n.get("content", "") or "")[:2000]),
                _score(query, n.get("category", "") or ""))
        if s > 0.1:
            scored.append((s, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _, n in scored[:_MAX]:
        content = (n.get("content") or "").strip()
        out.append({
            "title": n.get("title") or "(untitled)",
            "snippet": content[:220] + ("…" if len(content) > 220 else ""),
            "category": n.get("category") or "",
            "url": f"{NC_PUBLIC_URL}/apps/notes/note/{n.get('id')}",
        })
    return out

# ---- Notes CRUD for the Notes & Reminders UI --------------------------------
_NOTES_API = "/index.php/apps/notes/api/v1/notes"

def list_notes(limit=50, with_content=False):
    """Notes shaped for the UI: id/title/content/category/favorite/url.
    Favorites (Nextcloud 'starred' notes) come first, then newest-modified.
    Defaults to TITLES-ONLY (content=""), which is far faster for the polled
    list; the full body is lazy-loaded per note via get_note() when opened."""
    notes = _notes_all(titles_only=(not with_content))
    # (not favorite -> 1) sorts favorites first; then newest modified first.
    notes.sort(key=lambda n: (0 if n.get("favorite") else 1, -(n.get("modified") or 0)))
    out = []
    for n in notes[:limit]:
        out.append({
            "id": n.get("id"),
            "title": _clean(n.get("title") or "(untitled)"),
            "content": _clean(n.get("content") or "") if with_content else "",
            "category": n.get("category") or "",
            "favorite": bool(n.get("favorite")),
            "url": f"{NC_PUBLIC_URL}/apps/notes/note/{n.get('id')}",
        })
    return out

def add_note(title, content="", category=""):
    url = f"{NC_BASE_URL}{_NOTES_API}"
    body = {"title": (title or "New note").strip(), "content": content or "", "category": category or ""}
    r = requests.post(url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                      headers={"Accept": "application/json", "Content-Type": "application/json"},
                      data=json.dumps(body))
    r.raise_for_status()
    n = r.json()
    return {"id": n.get("id"), "title": n.get("title"), "content": n.get("content"),
            "url": f"{NC_PUBLIC_URL}/apps/notes/note/{n.get('id')}"}

def find_note_id(title):
    """Resolve a note by its (visible) title. Exact match first, then contains."""
    t = (title or "").strip().lower()
    notes = _notes_all()
    for n in notes:
        if (n.get("title") or "").strip().lower() == t:
            return n.get("id")
    for n in notes:
        if t and t in (n.get("title") or "").strip().lower():
            return n.get("id")
    return None

def update_note(note_id, title=None, content=None, category=None, favorite=None):
    url = f"{NC_BASE_URL}{_NOTES_API}/{note_id}"
    body = {}
    if title is not None:    body["title"] = title
    if content is not None:  body["content"] = content
    if category is not None: body["category"] = category
    if favorite is not None: body["favorite"] = bool(favorite)   # NC 'star'
    r = requests.put(url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                     headers={"Accept": "application/json", "Content-Type": "application/json"},
                     data=json.dumps(body))
    r.raise_for_status()
    return True

def delete_note(note_id):
    url = f"{NC_BASE_URL}{_NOTES_API}/{note_id}"
    r = requests.delete(url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                        headers={"Accept": "application/json"})
    r.raise_for_status()
    return True


# ============================================================ CALENDAR (CalDAV)
def _caldav_calendars():
    """Find the user's calendars (hrefs) via a PROPFIND."""
    url = _dav(f"calendars/{NC_USER}/")
    body = ('<?xml version="1.0"?><d:propfind xmlns:d="DAV:" '
            'xmlns:cs="http://calendarserver.org/ns/" xmlns:c="urn:ietf:params:xml:ns:caldav">'
            '<d:prop><d:resourcetype/><d:displayname/></d:prop></d:propfind>')
    r = requests.request("PROPFIND", url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                         headers={"Depth": "1", "Content-Type": "application/xml"}, data=body)
    r.raise_for_status()
    hrefs = re.findall(r"<d:href>(.*?)</d:href>", r.text, re.I)
    # keep collections that look like calendars (skip the parent and non-calendar ones)
    cals = [h for h in hrefs if h.rstrip("/").split("/")[-1] not in ("", NC_USER)]
    return cals

def _caldav_events(start, end):
    """REPORT VEVENTs in [start,end] across the user's calendars."""
    fmt = "%Y%m%dT%H%M%SZ"
    body = ('<?xml version="1.0"?><c:calendar-query xmlns:d="DAV:" '
            'xmlns:c="urn:ietf:params:xml:ns:caldav"><d:prop><c:calendar-data/></d:prop>'
            '<c:filter><c:comp-filter name="VCALENDAR"><c:comp-filter name="VEVENT">'
            f'<c:time-range start="{start.strftime(fmt)}" end="{end.strftime(fmt)}"/>'
            '</c:comp-filter></c:comp-filter></c:filter></c:calendar-query>')
    events = []
    for cal in _caldav_calendars():
        url = NC_BASE_URL + cal if cal.startswith("/") else cal
        try:
            r = requests.request("REPORT", url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                                 headers={"Depth": "1", "Content-Type": "application/xml"}, data=body)
            if r.status_code >= 400:
                continue
            for block in re.findall(r"BEGIN:VEVENT.*?END:VEVENT", r.text, re.S):
                events.append(block)
        except Exception:
            continue
    return events

def _ev_field(block, tag):
    m = re.search(rf"^{tag}[^:]*:(.+)$", block, re.M | re.I)
    return m.group(1).strip() if m else ""

def _parse_ev_dt(val):
    val = val.strip()
    for f in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S", "%Y%m%d"):
        try:
            return datetime.datetime.strptime(val, f)
        except ValueError:
            continue
    return None

def list_events(days_ahead=1):
    now = datetime.datetime.utcnow()
    end = now + datetime.timedelta(days=days_ahead)
    out = []
    for b in _caldav_events(now - datetime.timedelta(hours=1), end):
        dt = _parse_ev_dt(_ev_field(b, "DTSTART"))
        out.append({"summary": _ev_field(b, "SUMMARY") or "(no title)",
                    "start": dt, "location": _ev_field(b, "LOCATION")})
    out = [e for e in out if e["start"]]
    out.sort(key=lambda e: e["start"])
    return out[:_MAX]

def list_events_ui(days_ahead=90, limit=100):
    """Events shaped like the HA calendar feed the Notes UI expects:
    {summary, start(ISO), end(ISO), description}. start/end are ISO strings.
    The CalDAV time-range query is formatted with a 'Z' (UTC) suffix, so the
    bounds must be UTC — matching list_events()."""
    now = datetime.datetime.utcnow()
    end = now + datetime.timedelta(days=days_ahead)
    out = []
    for b in _caldav_events(now - datetime.timedelta(hours=12), end):
        sdt = _parse_ev_dt(_ev_field(b, "DTSTART"))
        if not sdt:
            continue
        edt = _parse_ev_dt(_ev_field(b, "DTEND")) or (sdt + datetime.timedelta(hours=1))
        out.append({
            "summary": _clean(_ev_field(b, "SUMMARY")) or "(no title)",
            "start": sdt.isoformat(),
            "end": edt.isoformat(),
            "description": _clean(_ev_field(b, "DESCRIPTION")),
        })
    out.sort(key=lambda e: e["start"])
    return out[:limit]

def add_event(summary, start_dt, duration_min=60, location=""):
    """Create a VEVENT in the user's first/default calendar."""
    cals = _caldav_calendars()
    if not cals:
        raise RuntimeError("no calendar found")
    cal = cals[0]
    base = NC_BASE_URL + cal if cal.startswith("/") else cal
    # Unique UID: second-resolution collided when several events were added in
    # the same second (same UID -> same URL -> each PUT overwrote the last).
    uid = f"jarvis-{int(datetime.datetime.utcnow().timestamp())}-{uuid.uuid4().hex[:8]}@jarvis"
    end_dt = start_dt + datetime.timedelta(minutes=duration_min)
    fmt = "%Y%m%dT%H%M%S"
    ics = ("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//JARVIS//EN\r\nBEGIN:VEVENT\r\n"
           f"UID:{uid}\r\nSUMMARY:{_ics_escape(summary)}\r\n"
           f"DTSTART:{start_dt.strftime(fmt)}\r\nDTEND:{end_dt.strftime(fmt)}\r\n"
           + (f"LOCATION:{_ics_escape(location)}\r\n" if location else "")
           + f"DTSTAMP:{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}\r\n"
           "END:VEVENT\r\nEND:VCALENDAR\r\n")
    url = base.rstrip("/") + f"/{uid}.ics"
    r = requests.put(url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                     headers={"Content-Type": "text/calendar; charset=utf-8"}, data=ics)
    r.raise_for_status()
    return {"summary": summary, "start": start_dt,
            "url": f"{NC_PUBLIC_URL}/apps/calendar"}


# -------- REMINDERS as CalDAV tasks (VTODO) -> sync to iPhone Reminders -------
def _calendars_with_components():
    """PROPFIND each calendar's supported-calendar-component-set, so we can find
    one that accepts VTODO (tasks). Returns [(href, displayname, comps_set)]."""
    url = _dav(f"calendars/{NC_USER}/")
    body = ('<?xml version="1.0"?><d:propfind xmlns:d="DAV:" '
            'xmlns:c="urn:ietf:params:xml:ns:caldav">'
            '<d:prop><d:displayname/><c:supported-calendar-component-set/></d:prop>'
            '</d:propfind>')
    r = requests.request("PROPFIND", url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                         headers={"Depth": "1", "Content-Type": "application/xml"}, data=body)
    r.raise_for_status()
    out = []
    for resp in re.findall(r"<d:response>.*?</d:response>", r.text, re.S | re.I):
        hm = re.search(r"<d:href>(.*?)</d:href>", resp, re.I)
        if not hm:
            continue
        href = hm.group(1)
        if href.rstrip("/").split("/")[-1] in ("", NC_USER):
            continue
        dn = re.search(r"<d:displayname>(.*?)</d:displayname>", resp, re.I)
        name = (dn.group(1).strip() if dn else "")
        comps = set(c.lower() for c in re.findall(r'name="(VEVENT|VTODO)"', resp, re.I))
        out.append((href, name, comps))
    return out

def _pick_task_calendar():
    """Choose the calendar/list to put VTODO reminders in. Prefers (1) the list
    named by NC_REMINDER_CALENDAR, (2) any VTODO-capable list, (3) first list."""
    cals = _calendars_with_components()
    if not cals:
        raise RuntimeError("no calendar found")
    want = (NC_REMINDER_CALENDAR or "").strip().lower()
    if want:
        for href, name, comps in cals:
            slug = href.rstrip("/").split("/")[-1].lower()
            if want in (slug, name.lower()):
                return href
    for href, name, comps in cals:          # first that accepts tasks
        if "vtodo" in comps:
            return href
    return cals[0][0]                        # last resort

def add_reminder(summary, due_dt, priority=0):
    """Create a VTODO (task) with a DUE time + alarm in a Nextcloud task list.
    iOS Reminders syncs VTODO over CalDAV, so this lands in the iPhone Reminders
    app on the next sync. due_dt is a naive LOCAL datetime (floating time)."""
    cal = _pick_task_calendar()
    base = NC_BASE_URL + cal if cal.startswith("/") else cal
    uid = f"jarvis-rem-{int(datetime.datetime.utcnow().timestamp())}-{uuid.uuid4().hex[:8]}@jarvis"
    due = due_dt.strftime("%Y%m%dT%H%M%S")                 # floating local time
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    summary = _ics_escape(_clean(summary).replace("\n", " "))
    ics = ("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//JARVIS//EN\r\n"
           "BEGIN:VTODO\r\n"
           f"UID:{uid}\r\nDTSTAMP:{stamp}\r\n"
           f"SUMMARY:{summary}\r\n"
           f"DUE:{due}\r\n"
           "STATUS:NEEDS-ACTION\r\n"
           + (f"PRIORITY:{priority}\r\n" if priority else "")
           + "BEGIN:VALARM\r\nACTION:DISPLAY\r\nTRIGGER;VALUE=DURATION:PT0S\r\n"
           f"DESCRIPTION:{summary}\r\nEND:VALARM\r\n"
           "END:VTODO\r\nEND:VCALENDAR\r\n")
    url = base.rstrip("/") + f"/{uid}.ics"
    r = requests.put(url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                     headers={"Content-Type": "text/calendar; charset=utf-8"}, data=ics)
    r.raise_for_status()
    return {"summary": summary, "due": due_dt,
            "url": f"{NC_PUBLIC_URL}/apps/tasks"}


# ============================================================ FILES (OCS search)
def search_files(query, limit=10):
    """Search Nextcloud files using the built-in unified search provider.
    This is Nextcloud's OWN index, so it finds files by name (and, where the
    full-text search app is installed, by content) — the same index the web UI
    search box uses. Returns [{name, path, url}]."""
    url = f"{NC_BASE_URL}/ocs/v2.php/search/providers/files/search"
    r = requests.get(url, auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                     headers={"OCS-APIRequest": "true", "Accept": "application/json"},
                     params={"term": query, "limit": limit})
    r.raise_for_status()
    data = r.json()
    entries = (((data or {}).get("ocs") or {}).get("data") or {}).get("entries") or []
    out = []
    for e in entries[:limit]:
        name = _clean(e.get("title") or "")
        sub = _clean(e.get("subline") or "")          # usually the folder path
        link = e.get("resourceUrl") or ""
        if link.startswith("/"):
            link = NC_PUBLIC_URL + link
        out.append({"name": name, "path": sub, "url": link or f"{NC_PUBLIC_URL}/apps/files"})
    return out


def files_configured() -> bool:
    return nc_configured()


# ====================================================== AI BRIDGE (TaskProcessing)
# Nextcloud 30+ exposes an OCS TaskProcessing API. Scheduling a task here runs it
# INSIDE Nextcloud with Nextcloud's own AI providers and its own file index — so
# JARVIS can borrow Eva/Assistant/Context-Chat capabilities (RAG over your files,
# the context agent that can act on Nextcloud) instead of duplicating them.
#   GET  /ocs/v2.php/taskprocessing/tasktypes   -> what this server can do
#   POST /ocs/v2.php/taskprocessing/schedule    -> start a task
#   GET  /ocs/v2.php/taskprocessing/task/{id}   -> poll until finished
_TP = "/ocs/v2.php/taskprocessing"
_OCS_HEADERS = {"OCS-APIRequest": "true", "Accept": "application/json"}

# Preference order: an agent that can ACT on Nextcloud, then RAG over files,
# then plain chat, then generic text-to-text.
_TASK_PREFERENCE = [
    "core:contextagent:interaction",
    "core:multimodal:contextagent:interaction",
    "context_chat:context_chat",
    "core:text2text:chat",
    "core:text2text",
]

def task_types():
    """Map of task-type id -> name, as this Nextcloud actually supports."""
    r = requests.get(f"{NC_BASE_URL}{_TP}/tasktypes", auth=_auth(),
                     timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL, headers=_OCS_HEADERS)
    r.raise_for_status()
    types = (((r.json() or {}).get("ocs") or {}).get("data") or {}).get("types") or {}
    out = {}
    for k, v in (types.items() if isinstance(types, dict) else []):
        out[k] = (v or {}).get("name") if isinstance(v, dict) else str(v)
    return out

def pick_task_type(available=None, prefer=None):
    """Choose the most capable available task type."""
    avail = available if available is not None else task_types()
    if prefer and prefer in avail:
        return prefer
    for t in _TASK_PREFERENCE:
        if t in avail:
            return t
    # fall back to anything that looks like text-to-text
    for k in avail:
        if "text2text" in k:
            return k
    return None

def _input_shape(task_type, prompt):
    """Different task types take different input shapes."""
    if "contextagent" in task_type:
        return {"input": prompt, "confirmation": 0, "conversation_token": ""}
    if task_type.endswith(":chat") or "context_chat" in task_type:
        return {"input": prompt, "system_prompt": "", "history": []}
    return {"input": prompt}

def _extract_output(task):
    out = (task or {}).get("output") or {}
    if isinstance(out, dict):
        for key in ("output", "message", "text", "answer"):
            v = out.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # some agents return a list of messages
        for v in out.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""

def ask(prompt, prefer=None, timeout=180, poll=2.0):
    """Ask Nextcloud's AI (Eva / Assistant / Context Chat) and wait for the answer.
    Returns (answer:str, task_type:str). Raises on hard failure."""
    import time as _t
    avail = task_types()
    if not avail:
        raise RuntimeError("Nextcloud reports no AI task types (is an AI provider configured?)")
    ttype = pick_task_type(avail, prefer)
    if not ttype:
        raise RuntimeError(f"no usable task type; available: {list(avail)[:8]}")

    payload = {"input": _input_shape(ttype, prompt), "type": ttype,
               "appId": "jarvis", "customId": f"jarvis-{int(_t.time())}"}
    r = requests.post(f"{NC_BASE_URL}{_TP}/schedule", auth=_auth(), timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL,
                      headers={**_OCS_HEADERS, "Content-Type": "application/json"},
                      data=json.dumps(payload))
    if r.status_code >= 400:
        raise RuntimeError(f"schedule failed HTTP {r.status_code}: {r.text[:200]}")
    task = (((r.json() or {}).get("ocs") or {}).get("data") or {}).get("task") or {}
    tid = task.get("id")
    if not tid:
        raise RuntimeError(f"no task id returned: {str(task)[:200]}")

    # poll — status: 1 scheduled, 2 running, 3 successful, 4 failed, 5 cancelled
    deadline = _t.time() + timeout
    while _t.time() < deadline:
        _t.sleep(poll)
        g = requests.get(f"{NC_BASE_URL}{_TP}/task/{tid}", auth=_auth(),
                         timeout=NC_TIMEOUT, verify=NC_VERIFY_SSL, headers=_OCS_HEADERS)
        if g.status_code >= 400:
            continue
        t = (((g.json() or {}).get("ocs") or {}).get("data") or {}).get("task") or {}
        status = t.get("status")
        ans = _extract_output(t)
        if ans:
            return ans, ttype
        if status in (4, 5, "STATUS_FAILED", "STATUS_CANCELLED"):
            raise RuntimeError(f"task {tid} ended with status {status}: {str(t.get('errorMessage'))[:160]}")
    raise TimeoutError(f"Nextcloud AI task {tid} did not finish within {timeout}s")


# ============================================================ MAIL (IMAP)
def mail_configured() -> bool:
    return bool(NC_IMAP_HOST and NC_IMAP_USER and NC_IMAP_PASSWORD)

def _imap():
    m = imaplib.IMAP4_SSL(NC_IMAP_HOST, NC_IMAP_PORT)
    m.login(NC_IMAP_USER, NC_IMAP_PASSWORD)
    return m

def _decode(s):
    try:
        return str(make_header(decode_header(s or "")))
    except Exception:
        return s or ""

def search_mail(query, limit=5):
    """Search INBOX by sender/subject/body for the query; newest first."""
    m = _imap()
    try:
        m.select("INBOX")
        # try FROM, then SUBJECT, then TEXT, union the results
        ids = set()
        for crit in ("FROM", "SUBJECT", "TEXT"):
            try:
                typ, data = m.search(None, crit, f'"{query}"')
                if typ == "OK":
                    ids.update(data[0].split())
            except Exception:
                continue
        ids = sorted(ids, key=lambda b: int(b), reverse=True)[:limit]
        out = []
        for i in ids:
            typ, msg_data = m.fetch(i, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
            if typ != "OK":
                continue
            hdr = email.message_from_bytes(msg_data[0][1])
            out.append({"from": _decode(hdr.get("From")),
                        "subject": _decode(hdr.get("Subject")) or "(no subject)",
                        "date": _decode(hdr.get("Date"))})
        return out
    finally:
        try: m.logout()
        except Exception: pass
# Security

- **Secrets live only in `.env`** (created with permissions 600 by `setup.py`). It is git-ignored.
  If you ever commit one, rotate every key in it. Deleting the file later doesn't remove it from git history.
- **Logins:** `JARVIS_USERS` PINs are compared in constant time. Use long ones. For access
  from outside your home, use HTTPS and a VPN (WireGuard/Tailscale) or a reverse proxy with
  its own authentication.
- **`JARVIS_API_TOKEN`** gives owner-level API access (phone shortcuts). Treat it like a password.
- **PC agent:** needs `AGENT_TOKEN` and only runs allow-listed actions. Never expose port 8765
  beyond your LAN. Desktop control pauses with `touch ~/.jarvis_agent_stop`.
- **Agent safety rules:** content from web pages, files and screens is handled as data, not
  instructions. The agent won't type into password fields, and won't place orders or pay.
- **Cloud vision:** with the Cloud agent brain and `AGENT_CLOUD_VISION=1`, screenshots are sent
  to your NVIDIA model. Set it to `0` to keep screenshots on your network.

Please report vulnerabilities privately through GitHub's "Report a vulnerability" (Security tab).

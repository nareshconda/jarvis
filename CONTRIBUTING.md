# Contributing to Jarvis

Thanks for helping! Contributions go through pull requests, and every PR is reviewed by
the maintainer before it's merged.

## Ground rules

- **Never commit secrets or personal data.** That means no `.env`, API keys, tokens, `*.db`
  files, certificates, real IPs, home paths or Home Assistant entity names. Use placeholders
  and add new settings to `.env.example` with a comment.
- **One feature or fix per PR.** Small PRs get reviewed and merged faster.
- **Everything must be optional.** A new integration must be switched off when its `.env`
  values are empty, and must not crash the server when it's missing.
- **No hard-coded hardware.** Refer to devices through `DEVICES[...]` / `_dev_name()`,
  never by a specific GPU, PC or Mac.
- **Keep the agent safe.** Content from web pages, files and screens stays *data*. Don't
  loosen the PC agent's command allow-list, kill switch, password-field and no-checkout rules.

## Getting started

```bash
git clone https://github.com/<you>/jarvis.git && cd jarvis
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python setup.py                 # writes your local .env (git-ignored)
python jarvis_server.py
```

## Before you open a PR

```bash
python -m py_compile $(git ls-files '*.py')     # everything compiles
python setup.py check                           # your services still connect
git diff --cached | grep -iE "api[_-]?key|token|password|192\.168\." && echo "check these lines!"
```

- [ ] Tested on your own setup (say which devices and integrations)
- [ ] New settings documented in `.env.example` (and `docs/` if needed)
- [ ] No secrets, personal data or unrelated changes

## Pull request flow

1. Fork the repo and create a branch: `git checkout -b feature/short-name`.
2. Commit with clear messages, push to your fork, then open a PR against `main`.
3. Describe **what** changed, **why**, and **how you tested it**. Add screenshots for UI changes.
4. The maintainer reviews it. Please respond to comments, and keep the PR up to date with `main`.

## Reporting bugs and ideas

Open an issue with your setup (OS, which devices and integrations, Ollama version), what you
expected, what happened, and the relevant log lines with secrets removed.

Security problems: **don't open a public issue**. Use the repo's *Security → Report a vulnerability*
(see `SECURITY.md`).

## License

By contributing, you agree that your contributions are licensed under the project's
[MIT License](LICENSE).

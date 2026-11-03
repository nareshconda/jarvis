# Publishing this repo to GitHub

Start from this cleaned folder with **fresh git history**. Don't reuse the `.git` folder from
your working copy: its history is the upstream fork's and must not get mixed in.

## 1. Last check before anything goes online

From the repo folder:

```bash
# nothing personal or secret should match:
grep -rnIiE "192\.168\.[0-9]+\.[0-9]+|api[_-]?key\s*=\s*['\"][A-Za-z0-9]|BEGIN (RSA|OPENSSH) PRIVATE" \
     --exclude-dir=.git . | grep -v "example\|e\.g\.\|192\.168\.1\." || echo "clean"
ls .env *.db 2>/dev/null && echo "^^ these must NOT be committed" || echo "no .env / db files"
```

## 2. Create the repository on GitHub

Use github.com → **New repository** → name it (e.g. `jarvis`) → **Public**. Leave
"Add a README / .gitignore / license" **unticked**, because this repo already has them.

Or with the GitHub CLI: `gh repo create jarvis --public --source . --remote origin`.

## 3. First commit and push

```bash
cd jarvis
git init -b main
git add .
git status                       # read the list: no .env, no *.db, no certs
git commit -m "Jarvis: self-hosted multi-device AI assistant"
git remote add origin https://github.com/<your-username>/jarvis.git
git push -u origin main
```

GitHub asks for a password when you push over HTTPS. Use a **personal access token**
(Settings → Developer settings → Fine-grained tokens, with "Contents: read and write" on this
repo), or set up an SSH key and use `git@github.com:<you>/jarvis.git`.

## 4. Turn on GitHub's safety nets

In the repo, go to **Settings → Code security** and enable:

- **Secret scanning** and **Push protection**, which block pushes that contain API keys
- **Dependabot alerts**

Then, under Settings → General:

- add a description and topics (`self-hosted`, `ollama`, `home-assistant`, `llm`, `raspberry-pi`)
- under Features, enable **Issues**

## 5. Keep your own setup separate

Clone the public repo wherever you run Jarvis, copy your private `.env` and `config/*.json`
in, and pull updates with `git pull`. Those files are git-ignored, so they never get committed.

If a secret ever lands in a commit, **rotate it straight away** at the provider. History
rewriting with `git filter-repo` comes second: forks and clones may already have a copy.

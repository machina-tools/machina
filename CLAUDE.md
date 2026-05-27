# Machina — Claude Code workspace

An open-source suite of AI developer tools that closes the gap between "I see the bug" and "the AI fixes it."  
Target: developers using AI agents (Copilot, Claude) for debugging.

- **Site**: machina.chat  
- **Repo**: github.com/machina-tools/machina (org: machina-tools)  
- **Email**: contact@machina.chat

## Tech stack

| Layer | Tech |
|-------|------|
| Landing | `index.html` static — dark theme, teal accent (#14b8a6 / #38bdf8), Inter |
| Blog | Astro 4 in `blog/` → `machina.chat/blog/` |
| Deploy | GitHub Actions `.github/workflows/deploy.yml` → GitHub Pages artifact |
| Newsletter | Brevo — form embed (no API key in frontend) |
| Domain | Hostinger → 4 A records GitHub Pages + CNAME www |

## Public tools

| Tool | Port | Description |
|------|------|-------------|
| BugCapture | 4327 | Screen+audio recording → .md with screenshot+Whisper for AI |
| ContextForge | 4328 | Git diff + SSH log + BugCapture → AI pre-session briefing |
| LearnBoard | 4331 | UI for LEARNING.md — persistent AI memory with statistics |
| PromptBoard | — | Single-file drag-and-drop canvas for building structured AI prompts |

Each tool: `server.js` / `server.mjs`, `index.html`, `package.json`, `.env.example`.  
Shared config: `~/.config/machina/servers.json`  
Setup: `bash setup.sh` from the repo root.

## Launch status

- [x] Landing live at machina.chat
- [x] Blog live at machina.chat/blog/
- [x] First article: "How I Built BugCapture"
- [x] Brevo newsletter integrated
- [x] 3 server-based tools with English HTML frontends
- [x] Demo videos — BugCapture, LearnBoard, PromptBoard (YouTube)
- [x] Email contact@machina.chat — active on Hostinger Mail
- [x] Blog post: "How I Built PromptBoard"
- [ ] Launch on Product Hunt + Hacker News

## Conventions

- **Language**: everything in English — code, UI, articles, commit messages
- **Design**: dark `#0f1117`, surface `#1a1d27`, teal `#14b8a6`, accent `#38bdf8`
- **No hardcoded paths** — env vars with sensible defaults
- **No credentials** in public code
- **Blog posts**: `.md` in `blog/src/pages/posts/`, frontmatter with `layout`, `title`, `date`, `description`, `tag`

## Working in this repo

1. **Public product** — every change is visible to external users; think installability and first-run UX
2. **Test before declaring done** — after every change, verify with `curl` on the live site for HTML, file diff for local code, Actions log for deploys. Never declare something complete without seeing concrete confirmation. Wait for deploys to finish before verifying.
3. **Update launch status** — after significant sessions, update the checklist above
4. **Blog-first** — every new feature or interesting story deserves a post
5. **Tools are self-contained** — each tool in `tools/` works independently

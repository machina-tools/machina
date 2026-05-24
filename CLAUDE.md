# Machina — Workspace AI

> Machina è la versione pubblica e open source della suite di tool sviluppata in `~/Documenti/Progetti/Tools`. Questo workspace è separato: niente path personali, niente configurazioni private, tutto in inglese.

## Progetto

Suite di tool open source che chiude il gap tra "vedo il bug" e "l'AI lo risolve".  
Target: developer che usano AI agents (Copilot, Claude) per debugging.

- **Sito**: machina.chat  
- **Repo**: github.com/machina-tools/machina (org: machina-tools)  
- **Email**: contact@machina.chat (Hostinger, da attivare)

## Stack tecnico

| Layer | Tech |
|-------|------|
| Landing | `index.html` statico — dark theme, teal accent (#14b8a6 / #38bdf8), Inter |
| Blog | Astro 4 in `blog/` → `machina.chat/blog/` |
| Deploy | GitHub Actions `.github/workflows/deploy.yml` → GitHub Pages artifact |
| Newsletter | Brevo — form embed (no API key nel frontend) |
| Dominio | Hostinger → 4 A record GitHub Pages + CNAME www |

## I 3 tool pubblici

| Tool | Port | Descrizione |
|------|------|-------------|
| BugCapture | 4327 | Record schermo+audio → .md con screenshot+Whisper per AI |
| ContextForge | 4328 | Git diff + SSH log + BugCapture → briefing AI pre-sessione |
| LearnBoard | 4331 | UI per LEARNING.md — memoria AI persistente con statistiche |

Ogni tool: `server.js` / `server.mjs`, `index.html`, `package.json`, `.env.example`.  
Config condivisa: `~/.config/machina/servers.json`  
Setup: `bash setup.sh` dalla root del repo.

## Stato lancio

- [x] Landing live su machina.chat
- [x] Blog live su machina.chat/blog/
- [x] Primo articolo: "How I Built BugCapture"
- [x] Newsletter Brevo integrata
- [x] 3 tool pubblici con HTML frontend in inglese
- [x] Testi lancio PH + HN + Twitter + LinkedIn (`LAUNCH.md`)
- [ ] Demo video — da registrare con BugCapture (47s, workflow reale)
- [ ] Email contact@machina.chat — da attivare in Hostinger → Email
- [ ] Lancio su Product Hunt + Hacker News (dopo il video)

## Convenzioni

- **Lingua**: tutto in inglese — codice, UI, articoli, commit message
- **Design**: dark `#0f1117`, surface `#1a1d27`, teal `#14b8a6`, accent `#38bdf8`
- **Nessun path hardcoded** — env var con default ragionevoli
- **Nessuna credenziale** nel codice pubblico
- **Articoli blog**: `.md` in `blog/src/pages/posts/`, frontmatter con `layout`, `title`, `date`, `description`, `tag`

## Come comportarsi in questo workspace

1. **Prodotto pubblico** — ogni modifica è visibile a utenti esterni; pensare a installabilità e UX da zero
2. **Nessun riferimento privato** — non citare tool, path o config da `~/Documenti/Progetti/Tools/`
3. **Aggiorna stato lancio** — dopo ogni sessione significativa aggiorna la checklist sopra
4. **Blog-first** — ogni feature nuova o storia interessante merita un articolo
5. **Mantieni la separazione** — le versioni in `Machina/tools/` sono indipendenti dagli originali in `Tools/`

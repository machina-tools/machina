# Machina

> AI tools that learn with you.

**[machina.chat](https://machina.chat)** · Open source · Free forever · Runs locally

---

Machina is a suite of developer tools powered by AI — that actually improve over time by tracking your workflow patterns, errors, and decisions session by session.

## The core idea

Most AI tools forget everything when you close the tab. Machina uses a persistent **learning layer** that records lessons learned, workflow patterns, and preferences — and makes them available to the AI on the next session, automatically.

**LearnBoard** is the interface to this layer: a web dashboard where you can see, edit, and manage everything your AI has learned.

## Tools (v1)

| Tool | What it does |
|------|-------------|
| [BugCapture](tools/bugcapture) | Record screen + audio → AI-ready .md bug report with Whisper transcription |
| [ContextForge](tools/contextforge) | Auto-collect git diff, SSH logs, last BugCapture → one AI briefing prompt |
| [LearnBoard](tools/learnboard) | Web UI for the AI learning layer — view, edit, track lessons and patterns |

## How it works

1. **You work with the tools** — record bugs, generate briefings, analyze schemas
2. **The AI records what it learns** — lessons, patterns, workflow observations
3. **Next session starts smarter** — the AI reads the learning layer at session start

## Getting started

```bash
git clone https://github.com/machina-tools/machina
cd machina
# Requirements: Node.js 18+, ffmpeg
node tools/bugcapture/server.mjs   # port 4327
node tools/contextforge/server.js  # port 4328
node tools/learnboard/server.js    # port 4331
```

## Philosophy

- **No cloud** — everything runs locally on your machine
- **No subscriptions** — MIT licensed, free forever
- **No lock-in** — plain files, plain markdown, no proprietary formats
- **Offline AI** — Whisper ONNX, Ollama — no API keys required

## Consulting

Need AI integrated into your development workflow? [Get in touch →](https://machina.chat/#consulting)

---

MIT License · Built by [@alexwmbi](https://github.com/alexwmbi)

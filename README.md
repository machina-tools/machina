# Machina — AI tools that learn with you

**[machina.chat](https://machina.chat)** · MIT License · Runs locally · No cloud required

Machina is an open-source suite of developer tools that closes the gap between "I see the bug" and "the AI fixes it." Each tool generates structured context your AI agent can act on immediately — and a shared learning layer makes every session smarter than the last.

---

## Tools

| Tool | What it does | How to run |
|------|-------------|------------|
| [**BugCapture**](tools/bugcapture) | Record screen + audio → `.md` bug report with Whisper transcription + screenshots | `node server.mjs` on port 4327 |
| [**ContextForge**](tools/contextforge) | Git diff + SSH logs + last BugCapture → one AI briefing prompt | `node server.js` on port 4328 |
| [**LearnBoard**](tools/learnboard) | Web UI for your AI's persistent memory — view, edit, and manage lessons | `node server.js` on port 4331 |
| [**PromptBoard**](tools/promptboard) | Drag-and-drop canvas for building structured AI prompts — no server needed | Open `index.html` directly |

---

## Quick start

**Requirements:** Node.js 18+, ffmpeg

```bash
git clone https://github.com/machina-tools/machina
cd machina
bash setup.sh
```

Then start each tool in a separate terminal:

```bash
cd tools/bugcapture   && node server.mjs   # → http://localhost:4327
cd tools/contextforge && node server.js    # → http://localhost:4328
cd tools/learnboard   && node server.js    # → http://localhost:4331
```

Open the tool's URL in your browser. PromptBoard needs no server — just open `tools/promptboard/index.html`.

---

## The workflow

```
PromptBoard        →   build your prompt visually
BugCapture         →   record the bug with voice + screenshots
ContextForge       →   collect git diff + server logs
LearnBoard         →   AI reads what it learned last session
                            ↓
                   Drop everything into Claude or Copilot
                   The AI has full context. It fixes the bug.
```

---

## BugCapture

Records your screen and microphone while you reproduce a bug. The server extracts one screenshot every 3 seconds, transcribes the audio with offline Whisper (no API key), and exports a single `.md` file your AI can act on immediately.

**Demo:** [youtube.com/watch?v=Qb9yuSKKojU](https://www.youtube.com/watch?v=Qb9yuSKKojU)

```bash
cd tools/bugcapture
cp .env.example .env   # edit port and Whisper language if needed
npm install
node server.mjs        # open http://localhost:4327
```

---

## ContextForge

Collects context automatically before an AI session: git diff of the last commit, error logs from SSH servers, and the last BugCapture report. Exports a single briefing file you drop into any AI agent.

```bash
cd tools/contextforge
cp .env.example .env
npm install
node server.js         # open http://localhost:4328
```

Configure SSH servers in `~/.config/machina/servers.json` — see [`tools/README.md`](tools/README.md) for the format.

---

## LearnBoard

The interface to Machina's learning layer. Every lesson the AI records, every workflow pattern it notices — stored in `LEARNING.md` and surfaced here as a searchable, editable dashboard.

**Demo:** [youtube.com/watch?v=IZajNU90-Fg](https://www.youtube.com/watch?v=IZajNU90-Fg)

```bash
cd tools/learnboard
cp .env.example .env
npm install
node server.js         # open http://localhost:4331
```

To connect your AI to the learning file, add to your `CLAUDE.md` or system prompt:
```
Read LEARNING.md at session start.
```

---

## PromptBoard

A drag-and-drop canvas for building structured AI prompts. Add text blocks, drop screenshots, draw flow diagrams, connect them with labeled arrows. Export as a `.md` file any AI can parse immediately. No install, no server — open the file and go.

**Demo:** [youtube.com/watch?v=GcjuYV4cH04](https://www.youtube.com/watch?v=GcjuYV4cH04)

```bash
open tools/promptboard/index.html   # or double-click it
```

---

## Philosophy

- **No cloud** — everything runs on your machine
- **No subscriptions** — MIT licensed, free forever
- **No lock-in** — plain markdown files, no proprietary formats
- **Offline AI** — Whisper ONNX for transcription, no API keys required

---

## Contributing

Issues and PRs welcome. Each tool is self-contained in `tools/<name>/` with its own `package.json` and `.env.example`.

---

## Consulting

Need AI integrated into your development workflow? [Get in touch →](https://machina.chat/#consulting)

---

MIT License · [machina.chat](https://machina.chat) · [Blog](https://machina.chat/blog/)

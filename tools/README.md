# Machina Tools

Four tools, one workflow: **build the prompt → record the bug → forge the context → AI fixes it**.

## Quick start

```bash
# From the repo root:
bash setup.sh

# Start each server-based tool in a separate terminal:
cd tools/bugcapture   && node server.mjs   # http://localhost:4327
cd tools/contextforge && node server.js    # http://localhost:4328
cd tools/learnboard   && node server.js    # http://localhost:4331
```

Then open each tool's `index.html` in your browser.

> **PromptBoard** needs no server — just open `tools/promptboard/index.html` directly.

---

## BugCapture

**Record screen + audio → AI-ready `.md` bug report**

| Requirement | Details |
|-------------|---------|
| Node.js | 18+ |
| ffmpeg | Required (audio/video processing) |
| Whisper | Auto-downloaded on first start (~150MB) |

**Configuration** (`.env`):
```bash
BUGCAPTURE_PORT=4327
WHISPER_LANGUAGE=english   # or: italian, french, spanish...
SERVERS_CONFIG=~/.config/machina/servers.json
```

**What it does:**
1. Records your screen + microphone in the browser
2. Sends the WebM recording to the server
3. Extracts 1 frame every 3 seconds (max 20)
4. Transcribes audio with offline Whisper
5. Exports a `.md` with embedded screenshots + transcript

Drop the `.md` into Copilot Agent, Claude, or any multimodal AI — it understands the bug immediately.

---

## ContextForge

**Auto-collect git diff + SSH logs + last BugCapture → one AI briefing**

| Requirement | Details |
|-------------|---------|
| Node.js | 18+ |
| SSH access | Optional (for remote log collection) |

**Configuration** (`.env`):
```bash
CONTEXTFORGE_PORT=4328
SERVERS_CONFIG=~/.config/machina/servers.json
BUGCAPTURE_OUTPUT_DIR=~/bugcapture-output
```

**servers.json format:**
```json
{
  "connections": [
    {
      "id": "prod-1",
      "label": "Production",
      "host": "your.server.com",
      "port": 22,
      "username": "deploy",
      "auth": "password",
      "password": "yourpassword",
      "logPaths": ["/var/log/apache2/error.log"]
    }
  ]
}
```

---

## LearnBoard

**Web UI for your AI learning layer**

| Requirement | Details |
|-------------|---------|
| Node.js | 18+ |
| LEARNING.md | Provided as template |

**Configuration** (`.env`):
```bash
LEARNBOARD_PORT=4331
LEARNING_FILE=./LEARNING.md
```

LearnBoard reads and edits `LEARNING.md` — the persistent memory file your AI reads at the start of each session. It surfaces the four key tables as editable, searchable UIs with stats and charts.

**To connect your AI to the learning file**, add to your `CLAUDE.md` or system prompt:
```
Read LEARNING.md at session start to remember preferences and past lessons.
```

---

## Transcriber

**Local Whisper transcription service — voice backend for browser tools**

| Requirement | Details |
|-------------|---------|
| Node.js | 18+ |
| ffmpeg | Required (audio conversion) |
| Whisper model | Auto-downloaded on first start (~150 MB) |

**Used by:** PromptBoard (Firefox/Brave voice fallback), BugCapture Web, LearnBoard Web.  
When Chrome/Edge is used, the Web Speech API handles transcription directly in the browser with no server needed. For all other browsers, the tools POST the recorded audio to `http://127.0.0.1:4324/transcribe`.

**Configuration** (`.env`):
```bash
TRANSCRIBER_PORT=4324
TRANSCRIBER_LANGUAGE=english   # or: italian, french, spanish, german…
TRANSCRIBER_MODEL=Xenova/whisper-base   # tiny | base | small
```

**API:**
```
GET  /health      → { ok, ready, model, language }
POST /transcribe  → body: audio blob (WebM/OGG) → { ok, text }
POST /shutdown    → graceful stop
```

---

## PromptBoard

**Visual canvas for building AI prompts**

| Requirement | Details |
|-------------|---------|
| Browser | Chrome/Edge recommended (Web Speech API for live dictation) |
| Server | None — open `index.html` directly |

**What it does:**

1. Drag **Text**, **Image**, and **Flow** blocks onto an infinite canvas
2. Connect blocks with arrows and annotate connections
3. Add images by drag-and-drop, file pick, or paste (Ctrl+V)
4. Dictate directly into any text block via microphone (Web Speech API)
5. Export the board as a structured `.md` with optional base64 images — ready for multimodal AI

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+S` | Save board |
| `Delete` | Delete selected block or arrow |
| `Escape` | Deselect / exit connect mode |

**No installation needed.** Boards are saved to `localStorage` — fully offline.

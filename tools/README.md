# Machina Tools

Three tools, one workflow: **record the bug → build the context → AI fixes it**.

## Quick start

```bash
# From the repo root:
bash setup.sh

# Start each tool in a separate terminal:
cd tools/bugcapture  && node server.mjs   # http://localhost:4327
cd tools/contextforge && node server.js   # http://localhost:4328
cd tools/learnboard  && node server.js    # http://localhost:4331
```

Then open `tools/bugcapture/index.html`, `tools/contextforge/index.html`,
and `tools/learnboard/index.html` in your browser.

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

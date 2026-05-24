---
layout: ../../layouts/PostLayout.astro
title: "How I Built BugCapture — and Why It Fixed a Real Bug in 40 Seconds"
date: "2026-05-24"
description: "Describing bugs in text wastes hours. I built a tool that records your screen, transcribes your voice with local Whisper, and generates a markdown file that AI agents can actually act on."
tag: "devtools"
---

I was debugging a form alignment bug on a client's server. Remote machine, no local environment, the kind of fix where you know exactly what you're seeing but translating it into words for your AI agent takes longer than finding it yourself.

"The second column in the input group is a few pixels wider than the first, but only when the browser is at an intermediate viewport width, and only after a user has interacted with the first field..."

You know this feeling. By the time you've finished describing the bug, you could have found it yourself.

That's the problem I wanted to solve.

## The insight

Modern AI agents — Claude, Copilot, GPT-4o — are multimodal. They can look at screenshots. They can read text. The gap between "I see a bug" and "AI can fix a bug" is the **format**: what does a bug report look like that an AI agent can actually act on?

A `.md` file with embedded screenshots and an audio transcript turns out to be the answer. The AI sees your screen, hears your description, and has the full context in a single file it can reason over.

BugCapture generates that file in one click.

## How it works

The workflow is three steps:

1. **Record** — click the button in the browser. BugCapture captures your screen and microphone simultaneously.
2. **Process** — the server extracts one frame every three seconds (up to 20 screenshots) and transcribes your audio with local [Whisper](https://github.com/openai/whisper) — no API key, no cloud, runs completely offline.
3. **Export** — you get a `.md` file: screenshots embedded as base64 + the full transcript, ready to drop into any AI.

There's also a LogLens toggle that captures SSH log lines from your server in parallel during the recording — so the AI gets runtime errors alongside the visual bug.

```
# BugCapture Report

**Date**: May 24, 2026
**Duration**: 47s
**Session ID**: bc_1716547200_a3f2

---

## Audio Transcript

So this form is misaligned — I can see that the label for the second
field is pushed about 12px to the right compared to the first one...

---

## Sequential Screenshots

### Screenshot 1 — 3s
![Screenshot 1](data:image/jpeg;base64,/9j/...)

### Screenshot 2 — 6s
...
```

That's the actual output. Drop it into Claude or Copilot Agent, and they have everything they need to trace the bug.

## The real test

The form alignment bug I mentioned — I recorded 47 seconds of screen and voice, exported the `.md`, and dropped it into Copilot Agent.

Copilot queried the database schema, traced the CSS through the component hierarchy, and identified a conflicting `width` rule in a child theme stylesheet. The fix was three lines of CSS.

Total time from "I see the bug" to "fix deployed": under two minutes. The 40 seconds of AI work was preceded by a 47-second recording and a couple seconds to paste the file.

I haven't described a bug in plain text since.

## The stack

BugCapture is a Node.js server (port 4327) with a browser frontend. The heavy lifting is:

- **`@xenova/transformers`** — Whisper running locally via ONNX, no Python required
- **ffmpeg** — frame extraction and audio conversion
- **`ssh2`** — optional SSH log capture during recording

Everything runs on your machine. No accounts, no subscriptions, no data leaving your network.

## Try it

BugCapture is part of [Machina](https://machina.chat) — an open source suite of tools that close the gap between "I see the bug" and "AI fixes the bug."

```bash
git clone https://github.com/machina-tools/machina.git
cd machina
bash setup.sh
cd tools/bugcapture && node server.mjs
```

Then open `tools/bugcapture/index.html` in your browser.

The first run downloads the Whisper model (~150MB). After that, everything works offline.

---

*Next up: [ContextForge](/blog) — how to auto-build a complete AI briefing from git diff + SSH logs + the last BugCapture, before every debug session.*

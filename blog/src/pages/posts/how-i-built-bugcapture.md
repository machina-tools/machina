---
layout: ../../layouts/PostLayout.astro
title: "How I Built BugCapture — From Screen Recording to AI-Ready Bug Report in One Click"
date: "2026-05-24"
description: "Describing bugs in text wastes hours. BugCapture records your screen and voice, transcribes with offline Whisper, and generates a structured .md file that any AI agent can act on immediately. Here's why I built it and how it works under the hood."
tag: "devtools"
readingTime: 8
youtube: "Qb9yuSKKojU"
github: "https://github.com/machina-tools/machina/tree/main/tools/bugcapture"
---

I was debugging a form alignment issue on a client's production server. Remote machine, no local environment. The kind of problem where you know exactly what you're seeing but translating it into words for your AI agent takes longer than finding the bug yourself.

*"The second column in the input group is a few pixels wider than the first, but only when the browser is at an intermediate viewport width — somewhere between 768px and 900px — and only after a user has interacted with the first field. The offset appears to be about 12px..."*

By the time you've written that, you've already lost the time you were trying to save. And the AI's first three responses are clarifying questions, because even that description is ambiguous.

This is the problem BugCapture solves: it turns a 47-second screen recording into a structured file your AI agent can act on immediately — with no text description from you, no manual screenshots, no copy-pasting error messages.

## The insight: bugs have a natural format

Modern AI agents — Claude, Copilot, GPT-4o — are multimodal. They can look at screenshots. They can read transcripts. The question isn't whether they can understand a bug from visual evidence; they clearly can. The question is: **what format packages that evidence in a way that maximizes AI understanding?**

The answer, after a lot of iteration, is a Markdown file with:
1. A voice transcript from the developer reproducing the bug (what you're thinking while you click)
2. Sequential screenshots at regular intervals (what the screen looked like over time)
3. Optional SSH log capture (what the server was doing at the same time)

This combination gives the AI three independent channels of information about the same event. The transcript explains intent. The screenshots show the visual state. The logs show the runtime state. An AI reading all three can build a more accurate model of the bug than it could from any one source alone.

## How it works

The workflow is exactly three steps:

**1. Record** — click Record in the BugCapture browser interface. The page requests screen and microphone access. You reproduce the bug while narrating what you see: *"I click this field, the page shifts, I scroll down and the overlay overlaps the submit button..."* The recording is captured as a MediaRecorder stream — audio and video in parallel, fully local.

**2. Process** — when you click Stop, the server runs two pipelines simultaneously:
- **Frame extraction**: ffmpeg extracts one screenshot every 3 seconds (configurable), converts them to JPEG at 85% quality. Up to 20 frames per recording.
- **Transcription**: `@xenova/transformers` runs the Whisper `base.en` model on the audio — fully offline, no API key, no data upload. The model runs via ONNX in Node.js. On a modern laptop, a 47-second recording transcribes in about 8 seconds.

**3. Export** — you get a `.md` file: screenshots embedded as base64 + the full transcript, structured for AI consumption.

The output looks like this:

```markdown
# BugCapture Report

**Date**: May 24, 2026
**Duration**: 47s
**Session ID**: bc_1716547200_a3f2
**Frames captured**: 16

---

## Audio Transcript

So the form is misaligned — I can see that the label for the second
field is pushed about 12 pixels to the right compared to the first one.
This only happens after I click the first field. When the page loads
it looks correct.

---

## Sequential Screenshots

### Screenshot 1 — 0s
![Initial state](data:image/jpeg;base64,/9j/...)

### Screenshot 2 — 3s
![After first click](data:image/jpeg;base64,/9j/...)

### Screenshot 4 — 9s
![Misalignment visible](data:image/jpeg;base64,/9j/...)
```

Drop that into Claude's context or Copilot Agent's workspace, and the AI has everything it needs. No text description from you. No manual screenshot upload.

## LogLens: adding the server side

One thing missing from a pure screen recording is what the server is doing. For bugs involving API calls, database queries, or SSR rendering, the visual output is often a symptom of a server-side cause.

BugCapture has an optional **LogLens** mode: enable it before recording, and the server opens an SSH connection (via `ssh2`) to your remote machine and tails the configured log files in parallel with the screen capture. When you export, the `.md` includes a timestamped log capture alongside the visual evidence:

```
## Server Logs (SSH: prod-server-01)

[09:42:11] INFO  GET /api/form-config 200 (12ms)
[09:42:14] WARN  FormRenderer: field width override applied for session 8a3f
[09:42:14] ERROR CSS injection failed: media query parse error at char 47
```

That error on the last line — a CSS injection failure — is the actual root cause of the visual bug. Without LogLens, the AI would have been guessing from screenshots. With it, the cause is explicit.

## The real test

The form alignment bug I mentioned: I recorded 47 seconds of screen and voice, exported the `.md`, and dropped it into Copilot Agent.

Copilot queried the database schema (the form config was partially stored there), traced the CSS through the component hierarchy, and identified a conflicting `width` rule in a child theme stylesheet that was being applied conditionally after the first user interaction triggered a re-render.

The fix was three lines of CSS. Total time from "I see the bug" to "fix deployed": **under two minutes**.

I haven't written a text bug description since.

## Key strengths

**Completely offline.** The Whisper model runs locally via ONNX. No transcription API, no upload, no account. The ~150MB model downloads once on first run and works without internet after that.

**AI-agnostic output.** The `.md` file works with any AI that accepts text: Claude, Copilot, GPT-4, Gemini, local models via Ollama. You're not locked into any specific tool.

**Zero configuration for basic use.** Install Node.js, clone the repo, `node server.mjs`. No API keys, no accounts, no config files required to get started.

**Deterministic output.** The same recording always produces the same `.md` structure. AI agents can reliably parse it without prompt engineering.

**Composable with the Machina suite.** ContextForge can automatically include your last BugCapture in the AI briefing it generates before each debug session — so you don't even need to manually attach the file.

## The technical stack

| Component | Technology |
|-----------|-----------|
| Screen + audio capture | Web MediaRecorder API |
| Frame extraction | ffmpeg (system dependency) |
| Transcription | `@xenova/transformers` + Whisper ONNX |
| SSH log capture | `ssh2` |
| Output format | Markdown with base64 JPEG |
| Server | Node.js ESM, no framework |

The frontend is a single HTML file with vanilla JavaScript — no build step, no bundler. The server itself is under 300 lines. The architecture is intentionally simple: a recording endpoint, a processing pipeline, and a static file server. Nothing proprietary, nothing hard to fork.

## What's next

A few things on the roadmap, driven by real friction in daily use:

- **Selective frame export** — a frame picker UI so you choose which screenshots go into the `.md`
- **Direct AI integration** — a "Send to Claude" button that POSTs the `.md` to the Claude API and opens the response in a side panel
- **Video attachment** — option to include the compressed MP4 alongside the `.md`, for AIs that support video input

If you use BugCapture and hit friction, open an issue on GitHub — that's how the roadmap gets built.

## Try it

BugCapture is part of [Machina](https://machina.chat) — an open source suite of tools that close the gap between "I see the bug" and "AI fixes the bug."

```bash
git clone https://github.com/machina-tools/machina.git
cd machina
bash setup.sh
cd tools/bugcapture && node server.mjs
```

Then open `tools/bugcapture/index.html` in your browser.

The first run downloads the Whisper model (~150MB). After that, everything works offline. The model is cached locally and reused across sessions.

---

*Related: [LearnBoard](/blog/posts/how-i-built-learnboard) — how to build a persistent memory layer so your AI stops starting from zero every session.*

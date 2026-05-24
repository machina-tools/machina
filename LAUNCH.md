# Machina — Launch Copy

Ready-to-paste copy for Product Hunt and Hacker News.

---

## Product Hunt

### Name
Machina

### Tagline
Record your screen, let AI fix your bugs

### Description

Machina is a suite of three open source tools that close the gap between "I see a bug" and "AI fixes the bug."

**BugCapture** records your screen + voice, extracts screenshots every 3 seconds, transcribes audio with local Whisper, and exports a `.md` with everything embedded. Drop it into Claude or Copilot Agent — the AI sees your screen, hears your description, and has full context to act on. No cloud, no API key, offline-first.

**ContextForge** builds a complete AI briefing before each debug session: git diff, SSH logs from your remote server, last BugCapture report — one prompt that gives the AI the full picture before you type a word.

**LearnBoard** is the memory layer. It tracks lessons learned, suggestions, and patterns across sessions in a `LEARNING.md` file that your AI reads at the start of each conversation. The longer you use it, the fewer things you have to explain twice.

All three tools are local, free, and self-hosted. They run on Node.js with no external services required.

→ [github.com/machina-tools/machina](https://github.com/machina-tools/machina)

### Maker's first comment

Hey PH! I'm Alessandro, the developer behind Machina.

I built BugCapture out of frustration: I was debugging a form layout bug on a client's remote server and spent more time describing the bug to my AI agent than it would have taken to find it myself. 

The insight: AI agents are multimodal. They can see screenshots. The missing piece was a format that put everything — visual context, voice explanation, runtime logs — in a single file an AI could act on.

BugCapture generates that file in one click. I tested it on the original bug: 47 seconds of recording, drop the `.md` into Copilot Agent, bug fixed in under two minutes. The AI traced the CSS through the component tree, found the conflicting rule, wrote the fix.

I've been using this workflow for three months now. ContextForge and LearnBoard grew from the same need — context that you shouldn't have to rebuild from scratch every session.

Everything runs locally. Whisper transcription is fully offline. No accounts, no subscriptions.

Happy to answer any questions below!

---

## Hacker News — Show HN

### Title
Show HN: Machina – open source tools that generate AI-ready bug reports from screen recordings

### Body

I built three tools to close the gap between "I see a bug" and "the AI fixes it."

**The core problem**: describing bugs in text is slow and lossy. AI agents are multimodal — they can look at screenshots — but there's no standard format for giving them the full visual + audio context of a bug.

**BugCapture** solves this. It records your screen + microphone in the browser, extracts frames with ffmpeg, transcribes audio with local Whisper (via @xenova/transformers — no Python, no cloud, runs offline), and outputs a `.md` with base64-embedded screenshots + transcript. Drop it into Claude or Copilot Agent and the AI has everything it needs to start working.

**ContextForge** builds a full AI briefing before each session: git diff, SSH logs from your remote server, last BugCapture report, auto-detected project type. One prompt, full context.

**LearnBoard** is a web UI for a `LEARNING.md` file — lessons learned, patterns, preferences — that your AI reads at the start of each conversation. It's the persistent memory layer that makes AI assistance improve over sessions rather than starting from zero.

All three run locally on Node.js. No external services, no API keys (except ffmpeg + Whisper model download on first use).

The workflow was validated on a real bug: a Joomla form alignment issue I couldn't describe well in text. BugCapture + Copilot Agent fixed it in under two minutes. The AI queried the DB schema and traced the CSS through the component hierarchy — without me specifying where to look.

GitHub: https://github.com/machina-tools/machina

---

## Twitter/X thread

**Tweet 1**
I got tired of describing bugs to AI. So I built a tool that records your screen instead.

BugCapture: 47 seconds of recording → .md with screenshots + transcript → drop into Claude or Copilot → bug fixed.

That's the entire workflow. 🧵

**Tweet 2**
The insight: AI agents are multimodal. They can see screenshots.

BugCapture uses local Whisper (offline, no API key) + ffmpeg to generate a .md with:
- Screenshots every 3s
- Full audio transcript  
- Optional SSH logs captured in parallel

**Tweet 3**
First real test: a form layout bug on a client's remote Joomla server.

I recorded 47s. Pasted the .md into Copilot Agent.

It queried the DB, traced the CSS through the component tree, found the conflicting rule, wrote the fix.

I haven't described a bug in plain text since.

**Tweet 4**
It's part of Machina — three open source tools for AI-powered debug workflows:

🎥 BugCapture — screen → .md for AI
⚡ ContextForge — git + logs → briefing before each session  
🧠 LearnBoard — persistent memory that improves over sessions

All local. All free.

→ machina.chat
→ github.com/machina-tools/machina

---

## LinkedIn post

Three months ago I had a debugging problem: I could see the bug clearly, but translating it into words for my AI agent took longer than finding it myself.

So I built BugCapture.

It records your screen + microphone, extracts screenshots every 3 seconds, transcribes your voice with offline Whisper, and generates a `.md` that AI agents can actually act on.

The test: a form layout bug on a client's remote server. 47 seconds of recording. Pasted the file into Copilot Agent. Fixed in under two minutes — the AI queried the database schema, traced the CSS, identified the conflicting rule.

I've since built two more tools around the same workflow:
→ **ContextForge**: assembles git diff + SSH logs + last recording into a complete briefing before each debug session
→ **LearnBoard**: a web UI for a persistent AI memory file that gets smarter over sessions

Machina is open source, free, and runs entirely locally. No cloud, no subscriptions.

→ machina.chat
→ github.com/machina-tools/machina

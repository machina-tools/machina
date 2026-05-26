---
layout: ../../layouts/PostLayout.astro
title: "How I Built PromptBoard — A Visual Canvas for Building AI Prompts"
date: "2026-05-26"
description: "Typing a complex AI prompt into a single text box is a bottleneck. PromptBoard is a drag-and-drop canvas that lets you build prompts visually — combining text blocks, images, flow diagrams, and voice dictation into a single structured .md file you paste into any AI."
tag: "devtools"
readingTime: 8
github: "https://github.com/machina-tools/machina/tree/main/tools/promptboard"
---

There's a class of AI prompts that don't fit in a text box.

Not because the ideas are too long — you can always write more. The problem is that the *structure* of what you want to communicate is inherently visual. You're describing a flow. You're pointing at an image. You're listing constraints that apply to some parts of the context but not others. You're trying to give the AI a briefing, not a paragraph.

The text box forces everything into one dimension. And the AI, however capable, has to reconstruct the structure you had in your head from a flat string of text.

PromptBoard solves this by flipping the approach: you build the prompt visually first, then export it.

## The problem with prompting complex tasks

Every developer who uses AI agents regularly hits a pattern like this:

You have a bug to fix. It's not a simple bug — it involves a flow you need to explain, a screenshot of the broken state, three or four constraints the fix has to respect, and a description of what the correct behavior should look like.

You start typing. You write the task description, then realize you need to explain the flow first. You paste in a screenshot and then write around it. You add the constraints at the end but they're not clearly linked to the specific parts they apply to. By the time you hit send, the prompt is a 400-word wall of text with an image in the middle, and you're hoping the AI can extract the structure you had in mind.

The AI can often do this. But you're asking it to do structural inference that you could have done once, clearly, in a canvas.

The deeper issue is that prompts have a **natural graph structure**: nodes (concepts, constraints, examples) with labeled relationships between them. A text box serializes that graph into a linear sequence and throws away the relationship labels. You're paying a tax every time you prompt something non-trivial.

## The design: blocks, arrows, export

PromptBoard is built around three concepts:

**Blocks** are the nodes. There are three types:
- **Text** — free-form content, the main carrier of context. Can have an optional label.
- **Image** — drag-and-drop or paste a screenshot. Gets embedded as base64 in the export.
- **Flow** — a process/decision/terminal node for describing logic visually.

**Arrows** connect blocks and carry a label. "This constraint applies to this flow step." "This screenshot is evidence for this bug description." The relationships are explicit, not inferred from reading order.

**Export** serializes the canvas back to text — but structured text. Blocks are rendered in top-to-bottom, left-to-right order. Arrows become a `## Flow` section listing every connection with its label. Images are embedded as base64 or referenced by description (configurable). The output is a Markdown file that any AI can parse immediately.

## Why a canvas, not a form

The first version of this tool was a form. Title field, description field, constraints field, image upload, flow diagram. Structured, explicit, readable.

It was unusable.

The problem with forms is that they impose a fixed schema. Your context doesn't always have a title and a description and three constraints. Sometimes it's just two things that are connected. Sometimes you have five images and no text yet. Sometimes the flow is the primary thing and the text is the annotation.

A canvas has no schema. You start with an empty surface and put things where they make sense. The structure emerges from the layout, not from a pre-defined form. That's exactly how you think through a problem before you explain it — spatially, not linearly.

There's also a practical benefit: the canvas layout is itself communicative. Two blocks close together with an arrow suggests a tight relationship. A block isolated in the corner is an aside. The AI doesn't read the spatial layout, but *you* do — and it helps you build a more coherent prompt before you export it.

## How voice dictation works

PromptBoard has voice dictation on every text block and arrow label. Two modes, depending on your browser:

**Chromium (Chrome, Edge):** uses the Web Speech API with `continuous: true` and `interimResults: false`. You click the mic button, talk, and transcribed text appends to the block in real time. Stop talking, click the mic again. No server, no API, no latency — the model runs in the browser.

**Firefox and others:** MediaRecorder captures the audio, then sends it to a local Whisper server (Transcriber, port 4324) for transcription. If Transcriber isn't running, a dialog appears with the audio playback and a textarea — you can type what you said, or replay the audio and transcribe manually.

The asymmetry is intentional: Chromium's built-in speech recognition is good enough for note-taking velocity. Whisper is better for longer or more technical dictation, and the Transcriber integration is there for users who want it.

## The export format

The full export for a PromptBoard session looks like this:

```markdown
# Fix the checkout form

**Goal**
Fix the checkout form — Cart component won't submit after the last refactor

**Constraints**
No new deps · TypeScript strict · keep under 50 lines

![cart-screenshot](data:image/png;base64,...)

**[▭ Process]** Cart validates form fields

**[◇ Decision]** Payment API responds?

**[○ Terminal]** Show success or error state

## Flow

Cart validates form fields → Payment API responds? (calls POST /api/checkout)
Payment API responds? → Show success or error state (on failure: surface error message)
```

When the AI reads this, it has:
- The task in plain language
- The constraints explicitly stated (not buried in prose)
- The screenshot as direct visual evidence
- The flow as a labeled graph, not an implied sequence

The difference in response quality, compared to an equivalent free-text prompt, is significant — especially for longer tasks where the AI would otherwise have to infer structure.

## Technical architecture

PromptBoard is a single HTML file, around 1,100 lines. No build step, no `npm install`, no server. Open it in a browser and it works.

**State:** a single `S` object holds all blocks, arrows, history stack, and interaction state. Everything is JSON-serializable. Boards are saved to `localStorage` (up to 20 boards).

**Undo/redo:** a snapshot-based history (`JSON.stringify` + `JSON.parse` of the state). Up to 60 snapshots. `Ctrl+Z` / `Ctrl+Y` work everywhere outside a text input.

**Arrows:** rendered as SVG quadratic Bézier curves (`Q` command) with a slight perpendicular offset to avoid overlapping block edges. Hit areas are 14px-wide transparent paths over 1.5px visible paths — wide enough to click but invisible. Arrowhead markers are defined in `<defs>`.

**Canvas:** 3000×2000px scrollable area. Blocks are `position:absolute` divs. Drag uses `mousedown` on the header + `mousemove` + `mouseup` on `document`. Resize uses a `14px × 14px` handle at the bottom-right corner.

**Block types:** `text`, `image`, `flow`. Flow nodes have three shapes (Process, Decision, Terminal) controlled by a CSS class + a slight border-radius override for Terminal.

## Strengths

**No installation.** The tool lives in one file. You can put it on a USB drive, serve it from any static host, or just keep it in your project folder and open it with a double-click.

**Multimodal output.** The base64 image embedding means the exported `.md` is a self-contained document — images travel with the text. You can paste the entire thing into Claude or GPT-4o and the screenshots are right there.

**Voice-first friendly.** For developers who think faster than they type, or who are debugging a live environment and need both hands, the voice dictation makes PromptBoard usable without touching a keyboard.

**Composable with the rest of Machina.** The `.md` export is the same format BugCapture produces. A natural workflow is: BugCapture records the bug, ContextForge adds the git diff and logs, PromptBoard adds the visual structure and constraints. Three files (or one merged file) give the AI a complete briefing.

## What's next

A few directions this could go:

- **Templates** — pre-built board layouts for common prompt types: bug report, feature spec, code review brief, architecture decision.
- **Transcriber integration** — when the local Whisper server is running, show its status in the UI and use it automatically for Firefox users without the dialog step.
- **Export presets** — different export formats for different AI tools: Claude's CLAUDE.md format, a Copilot workspace file, a structured JSON payload for API use.

For now, it does the one thing it needs to do well: turns a messy collection of context into a structured brief your AI can act on.

PromptBoard is part of [Machina](https://machina.chat) — a free, open-source suite of AI developer tools.  
→ [View on GitHub](https://github.com/machina-tools/machina/tree/main/tools/promptboard)

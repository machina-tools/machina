---
layout: ../../layouts/PostLayout.astro
title: "How I Built LearnBoard - The UI That Makes Your AI Remember You"
date: "2026-05-25"
description: "Every AI session starts from zero. LearnBoard is the interface layer that solves this: a real-time dashboard for the structured memory file your AI reads at the start of every session. View, edit, and control what your AI knows about you."
tag: "devtools"
readingTime: 9
youtube: "IZajNU90-Fg"
github: "https://github.com/machina-tools/machina/tree/main/tools/learnboard"
---

There's a problem that every developer who works with AI agents eventually runs into: **the AI doesn't remember you.**

You spend twenty minutes at the start of every session re-explaining your stack, your preferences, the constraint you discovered last week, the mistake you almost made twice. You know you have to do this. The AI has no memory of your last conversation. Every session, it starts fresh.

This is the biggest hidden cost of working with AI agents. It's not the hallucinations or the wrong answers - those are visible failures you can debug. The bigger cost is the invisible overhead: the context-building you do every single time, the lessons that get re-learned, the preferences that get ignored, the mistakes that happen again because the AI didn't know they were mistakes.

LearnBoard is the tool I built to solve this.

## The core idea: structured memory as a file

The insight that made this possible is simple: if you want an AI to remember something persistently, put it in a file it reads at session start.

This isn't a new idea. CLAUDE.md works this way. Many productivity workflows work this way. But the gap was always the *management layer* - there was no way to see what was in the memory, search it, or edit it without opening a raw text editor and hoping you understood the schema.

LearnBoard is the management interface for a structured memory file called `LEARNING.md`. Everything the AI has learned about you - your workflow preferences, patterns it has recognized, mistakes to avoid, successful approaches to revisit - lives in that file. LearnBoard makes that invisible layer visible, searchable, and editable in real time.

## How it works

LearnBoard is a Node.js server (port 4331) that serves a web dashboard for your `LEARNING.md` file. The file uses a structured Markdown format with defined sections:

- **Lessons** - explicit things the AI has learned ("always prefer local tools over cloud APIs")
- **Tools** - the tools and versions in your environment
- **Suggestions** - pending ideas from the AI that haven't been implemented yet
- **Stats** - success rates, session counts, learning velocity

The server watches the file with `chokidar` and pushes updates to the UI over Server-Sent Events - so when you open a second terminal and the AI writes to the file, you see it appear in the dashboard in real time.

```
LEARNING.md excerpt:

## Lessons Learned

| # | Category | Lesson | Confidence | Sessions |
|---|----------|--------|------------|---------|
| 15 | tooling | Always prefer local/free solutions - never propose paid APIs without exhausting local alternatives first | high | 12 |
| 18 | ux | User prefers autonomous tools that find context on their own - not manual forms to fill in | high | 8 |
| 23 | ops | Always restart via the bash script that rebuilds nvm - `nohup node` fails silently without nvm env | confirmed | 5 |

## Pending Suggestions

| # | Suggestion | Status | Votes |
|---|-----------|--------|-------|
| 4 | Add keyboard shortcut to export BugCapture without clicking | pending | +3 |
| 7 | Auto-detect project from git remote in ContextForge | in-review | +2 |
```

That file, prefixed to the AI's system prompt, means the agent starts every session already knowing your preferences, your environment, and what approaches have worked or failed before.

## The dashboard

The web UI has four main views:

**Lessons table** - all lessons with confidence score (low / medium / high / confirmed), category filter, free-text search, and inline editing. Click any cell to edit. New lessons append instantly. The AI can add lessons via a CLI flag; you see them appear in real time.

**Tools inventory** - your stack: language versions, frameworks, key dependencies. LearnBoard reads your `package.json`, `.nvmrc`, and SSH environment automatically to bootstrap this section. You correct or add entries; the AI uses them to make stack-appropriate suggestions.

**Suggestions queue** - pending ideas from the AI, with a +1/−1 voting system. Ideas that accumulate positive votes get promoted to the "accepted" column, which the AI treats as confirmed guidelines for future sessions.

**Session stats** - a lightweight histogram of session count, fix rate, and learning velocity over time. This is the part that makes the "learning" claim concrete: you can see that lesson #15 has appeared in 12 sessions and has a 94% success rate. The AI isn't just guessing - it has evidence.

## The innovation: meta-AI

The thing that makes LearnBoard different from a personal wiki or a note-taking tool is that it's **designed to be read by the AI, not by you**.

The structured format is chosen specifically to be unambiguous to a language model. The confidence scores, session counts, and vote history are signals the AI uses to weight lessons against each other. When two lessons conflict, the one with more sessions and higher confidence wins.

The AI can also *write* to the file. After a successful session, you can ask Claude or Copilot to "add a lesson to LEARNING.md about what we just discovered." It knows the schema, writes to the right section, and the dashboard updates immediately.

This is what I call the **learning loop**: the AI learns from each session, stores the lesson, and is better informed for the next one. LearnBoard makes that loop visible and controllable - you're not just hoping the AI improved, you can see the evidence.

## Key strengths

**Full local operation.** No cloud, no sync, no account. `LEARNING.md` is a plain text file you can read, commit, backup, and share without any vendor dependency. The server is a ~200-line Node.js file.

**AI-agnostic.** The file format works with Claude, Copilot, GPT-4, Gemini, or any agent that accepts system prompt context. You're not locked into a specific tool.

**Human-readable fallback.** When no dashboard is running, the memory layer is just a Markdown file. Open it in any editor, read it, edit it with a plain text cursor. No black box.

**Incremental adoption.** You can start with five lessons and an empty tools section. The structured format grows with your needs - there's no minimum viable configuration.

**Survivable architecture.** When a new AI model comes out, you don't migrate data. The file stays the same. The new model reads the same lessons on day one.

## The technical stack

LearnBoard keeps the stack minimal deliberately:

- **Node.js server** - serves the dashboard, reads/writes `LEARNING.md`, handles the file watcher
- **`chokidar`** - cross-platform file watching, pushes SSE events to all connected clients on file change
- **`marked`** - Markdown parsing to extract structured sections from the file
- **`js-yaml`** - optional YAML frontmatter for metadata
- **Vanilla JavaScript frontend** - no framework, no build step, one HTML file with inline CSS

The server starts in under a second and uses less than 50MB of RAM. There's nothing to configure beyond pointing it at your `LEARNING.md` file.

## The real test

I've been running LearnBoard on every project for four months. In that time, my `LEARNING.md` file has grown to 34 lessons, 18 pending suggestions, and a tools inventory for 6 active projects.

The sessions where I don't preload the context are noticeably worse. The AI proposes solutions I've already ruled out, asks questions I've already answered, and sometimes makes the exact mistakes that are documented in the file. The sessions where I do are faster, more targeted, and more accurate.

The most concrete evidence: lesson #19 documents a deployment pattern specific to one client's server setup. I've referenced it in 7 sessions since adding it. Every time, the AI uses it without being told. That's 7 explanations I didn't have to give.

## Try it

LearnBoard is part of [Machina](https://machina.chat) - an open source suite of tools that close the gap between how you work and how your AI understands you.

```bash
git clone https://github.com/machina-tools/machina.git
cd machina
bash setup.sh
cd tools/learnboard && node server.mjs
```

Then open `http://localhost:4331` in your browser.

The first time you open it with an empty `LEARNING.md`, the dashboard prompts you with a template to get started. You can also create the file manually - the format is human-readable and easy to bootstrap.

---

*Next up: [ContextForge](/blog) - how to auto-assemble a complete AI briefing from git diff, SSH logs, and your last BugCapture session, before you even open a chat window.*

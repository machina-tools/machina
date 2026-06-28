---
layout: ../../layouts/PostLayout.astro
title: "How I Brought My AI Tools to the Browser - No Server Required"
date: "2026-06-07"
description: "BugCapture and LearnBoard were Node.js tools that needed servers. Here's how I rewrote them as zero-install browser apps - replacing ffmpeg with canvas, the file system with localStorage, and adding a voice cascade that degrades gracefully across every browser."
tag: "devtools"
readingTime: 10
github: "https://github.com/machina-tools/machina"
---

There's a real friction in asking someone to try a tool you've built.

"Clone the repo, run `npm install`, start the server, open the HTML file." Four steps, each with its own failure mode. If the person doesn't have Node.js installed, they stop at step one. If they do have it, they might hit a permission error, a port conflict, a missing dependency. By the time the tool opens, they've already decided whether they trust it.

PromptBoard avoided this by being a single HTML file with no server. You open it. It works. That simplicity was the right call - but it left BugCapture and LearnBoard behind the barrier.

So I built web versions of both. This is the story of what that required.

## The constraint: no server, no install

The goal was simple to state: both tools should be openable in a browser tab with no prior setup. The challenge was that each tool relied on server-side capabilities that seemed irreplaceable:

- **BugCapture** used `ffmpeg` to extract video frames and `whisper.cpp` to transcribe audio
- **LearnBoard** used Node.js to read and write `LEARNING.md` on disk, and `chokidar` to watch for changes

The obvious answer - "just make the server optional" - doesn't work if the core feature requires it. So I had to find browser-native equivalents for each dependency.

## Replacing ffmpeg with the Canvas API

BugCapture's core workflow is: record screen + audio → extract one frame every three seconds → transcribe the audio → produce a `.md` with screenshots and a transcript.

The ffmpeg dependency was doing two things: video demuxing (pull frames at specific timestamps) and audio conversion (WebM to WAV at 16 kHz for Whisper). Replacing both with browser APIs turned out to be cleaner than I expected.

**Frame extraction** works by creating a `<video>` element in memory, seeking to each target timestamp, and drawing the current frame to a `<canvas>`:

```javascript
async function extractFrames(videoBlob, intervalSec = 3) {
  const url = URL.createObjectURL(videoBlob);
  const video = document.createElement('video');
  video.src = url;
  video.muted = true;
  await new Promise(r => video.addEventListener('loadedmetadata', r, { once: true }));

  // Firefox quirk: WebM recorded with MediaRecorder has duration = Infinity
  // Seeking to a large value forces the browser to resolve the actual duration
  if (!isFinite(video.duration)) {
    video.currentTime = 1e9;
    await new Promise(r => video.addEventListener('timeupdate', r, { once: true }));
  }

  const duration = video.duration;
  const frames = [];
  const canvas = document.createElement('canvas');
  canvas.width = 1280; canvas.height = 720;
  const ctx = canvas.getContext('2d');

  for (let t = 0; t < duration && frames.length < 20; t += intervalSec) {
    video.currentTime = t;
    await new Promise(r => video.addEventListener('seeked', r, { once: true }));
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    frames.push(canvas.toDataURL('image/jpeg', 0.8));
  }

  URL.revokeObjectURL(url);
  return frames;
}
```

The Firefox duration issue was the only real edge case. MediaRecorder on Firefox writes WebM files without a duration header - the browser reports `Infinity` for `video.duration`. Seeking past the end of the video forces it to scan the file and resolve the actual length. It's a documented quirk, and the fix is a single seek.

**Audio transcription** in the browser version uses the same three-level cascade as PromptBoard: Web Speech API for Chrome and Edge (live, during recording), then a POST to the local Transcriber service at port 4324, then a fallback textarea where you can type what you said.

## Replacing the file system with localStorage

LearnBoard's web version needed to do everything the server version does - read and write structured Markdown tables - but with no access to the file system.

The solution was `localStorage`. The web version stores the entire `LEARNING.md` content as a string under the key `learnboard-web-data`. On load, it parses the stored content into the four section tables. On any edit, it serializes back to Markdown and writes to storage.

The Markdown parser is a small custom function that finds section headers (`### Lessons Learned`, `### Pending Suggestions`, etc.), reads the table rows below each one, and builds a JavaScript object. The serializer does the reverse - it takes the in-memory tables and produces valid Markdown with the same headers and pipe formatting:

```javascript
function parseMarkdown(raw) {
  const sections = { lessons: [], suggestions: [], tools: [], observations: [] };
  const sectionMap = {
    'Lessons Learned': 'lessons',
    'Suggestions Log': 'suggestions',
    'User Requests': 'tools',
    'AI Observations': 'observations'
  };
  let current = null;
  for (const line of raw.split('\n')) {
    const header = Object.keys(sectionMap).find(h => line.includes(h));
    if (header) { current = sectionMap[header]; continue; }
    if (current && line.startsWith('|') && !line.startsWith('| ---') && !line.startsWith('| #')) {
      const cells = line.split('|').slice(1, -1).map(c => c.trim());
      if (cells.length >= 3) sections[current].push(cells);
    }
  }
  return sections;
}
```

For users who have an existing `LEARNING.md` from the server version, the web version includes an import dialog: paste the Markdown text, click import, and all four tables are populated. Export works in both directions - copy to clipboard or download as `.md` - so you can move data between the browser version and the local server version without any friction.

The main limitation is storage size. localStorage has a 5 MB limit per origin. A `LEARNING.md` file with hundreds of lessons could approach that limit. In practice, most users' files are well under 100 KB - but the web version shows a storage indicator and warns if you're approaching the limit.

## Voice in every browser

Both web versions inherit PromptBoard's voice cascade. The implementation follows a consistent pattern:

1. **Chromium (Chrome, Edge):** Web Speech API - live dictation, no server, results appear as you talk
2. **Firefox, Brave, others:** MediaRecorder captures audio → POST to `http://127.0.0.1:4324/transcribe` (Transcriber service) → if unreachable, show a textarea with the audio playback so you can type what you said

The browser detection runs once on page load:

```javascript
const IS_BRAVE = !!navigator.brave;
const IS_FIREFOX = navigator.userAgent.includes('Firefox');
const HAS_SPEECH = 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
const VOICE_MODE = HAS_SPEECH && !IS_BRAVE ? 'live' : 'remote';
```

The UI adapts to show which mode is active - a green pill for live Web Speech, orange for Transcriber, red/manual for the fallback. Users know what to expect before they click the mic.

## What the web versions can't do

Parity with the server versions isn't complete, and I'm not pretending otherwise. Both web tools show a visible notice explaining the differences:

**BugCapture web** is missing:
- Lossless frame extraction (canvas JPEG introduces compression)
- ZIP download (requires the server to bundle files)
- Native screen capture permission persistence (the browser re-prompts each session)

**LearnBoard web** is missing:
- Real-time sync with `LEARNING.md` on disk (localStorage is isolated from the file system)
- The file watcher that updates the UI when another process writes to the file
- Automatic project detection from git remote

These aren't gaps I plan to close - they're fundamental constraints of the browser environment. The web versions are for trying the tools without installing anything. The server versions are for using them in production.

The trade-off is explicit in the UI: every web version shows a banner that says what it can't do and links to the install instructions for the full version.

## The deployment change

To ship the web versions, I added two lines to the GitHub Actions build that copies the web HTML files into the Pages artifact alongside the blog:

```yaml
mkdir -p _site/tools/learnboard
cp tools/learnboard/index-web.html _site/tools/learnboard/
mkdir -p _site/tools/bugcapture
cp tools/bugcapture/index-web.html _site/tools/bugcapture/
```

The landing page now has "Try [Tool] →" buttons that link directly to these hosted versions. No install, no terminal, no `npm`. Click, wait two seconds for the tab to open, and the tool is running.

## What this changes

The primary goal was reducing friction for new users. But the secondary effect was more interesting: building the browser versions forced me to re-examine every dependency.

ffmpeg was doing frame extraction. Canvas can do frame extraction. The ffmpeg call was two lines in the server - but it required a system install, a path check, a spawn + error handling pipeline. The canvas version is twenty lines of JavaScript that runs in the browser process with no external dependency. The browser version is more auditable and more portable, even if it produces slightly smaller output files.

`chokidar` was watching for file changes. localStorage can store the same data with a simpler read/write API. The file watcher was the right tool for the server context - but in the browser, the event model is already event-driven; there's nothing to watch.

Sometimes the constraint of a new environment produces the cleaner solution.

---

*The web versions are live at [machina.chat](https://machina.chat) - try LearnBoard and BugCapture directly in your browser. For production use with full feature parity, install via `bash setup.sh` from the [GitHub repo](https://github.com/machina-tools/machina).*

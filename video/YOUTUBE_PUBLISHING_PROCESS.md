# YouTube Publishing Process — Machina Demo Videos

Complete workflow from design to landing page embed.

---

## 1. Video design

**Goal:** Show the tool solving a real problem end-to-end in under 90 seconds.

**Structure used:**
- 0–5s: problem statement (the pain point)
- 5–20s: record the bug or open the tool
- 20–60s: show the tool working (key steps, no dead time)
- 60–75s: result + AI fixes / confirms the fix
- Final beat: call to action (GitHub link or machina.chat)

**Voiceover:** AI voice "Liam" via ElevenLabs (or similar TTS). Write the script first, then generate audio, then sync to screen recording.

**Screen recording settings:**
- Resolution: 1920×1080 (or 2x retina at 3840×2160 scaled to 1080p)
- Frame rate: 30fps
- No cursor during navigation pauses — use Kdenlive/DaVinci cuts
- Terminal font size: 16–18px minimum for legibility

**Files:**
- Raw recording: `video/raw/`
- Final edited: `video/demo-video-final.mp4`
- Script: `video/script.md` (optional but recommended)

---

## 2. Video editing

Tools used: DaVinci Resolve (free) or Kdenlive.

Checklist:
- [ ] Cut dead time at start/end
- [ ] Add intro card (Machina logo + tool name, 2s)
- [ ] Sync voiceover to actions (slight delay is fine — AI voice needs 0.3s lead)
- [ ] Add lower-third title if showing multiple steps
- [ ] Color grade: match dark theme — boost contrast slightly for screen legibility
- [ ] Export: H.264, MP4, 1080p, 10–15 Mbps bitrate

---

## 3. YouTube upload

1. Go to [youtube.com/upload](https://www.youtube.com) and sign in with the Machina account (or personal account used for the channel).
2. Upload the final `.mp4` file.
3. **Title format:** `[Tool name] — [one-line benefit] | Machina`  
   Example: `BugCapture — Bug to AI-ready context in 47 seconds | Machina`
4. **Description template:**
   ```
   [Tool name] is part of Machina — an open-source suite of AI developer tools.
   
   [One paragraph describing what you see in the video.]
   
   → GitHub: https://github.com/machina-tools/machina
   → Site: https://machina.chat
   → [Direct tool link if available]
   
   #AI #DeveloperTools #OpenSource #MachinaTools
   ```
5. **Thumbnail:** Screenshot of the best frame, or create a 1280×720 dark-theme card with the tool name and a key stat (e.g. "Bug fixed in 47s").
6. Set visibility to **Public** (or Unlisted for testing the embed first).
7. After upload, copy the **video ID** from the URL: `https://www.youtube.com/watch?v=XXXXXXXXXXX` → the ID is `XXXXXXXXXXX`.

**Published video IDs:**
| Tool | YouTube ID | Title |
|------|-----------|-------|
| BugCapture | `Qb9yuSKKojU` | BugCapture demo |
| LearnBoard | `IZajNU90-Fg` | LearnBoard demo |
| PromptBoard | `GcjuYV4cH04` | PromptBoard — Build AI Prompts Visually | Machina |

---

## 4. Updating the landing page

Find the video section in `index.html` (search for `<!-- ── VIDEOS ──`).

The embed template:
```html
<div style="position:relative;padding-bottom:56.25%;height:0;border-radius:12px;overflow:hidden;border:1px solid var(--border);box-shadow:0 0 40px rgba(0,200,224,.10);">
  <iframe
    src="https://www.youtube.com/embed/VIDEO_ID_HERE"
    title="Tool name demo"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen
    style="position:absolute;top:0;left:0;width:100%;height:100%;">
  </iframe>
</div>
<div style="margin-top:16px;">
  <div style="font-weight:700;font-size:16px;color:var(--text);margin-bottom:4px;">Tool Name</div>
  <div style="font-size:14px;color:var(--text-dim);">One-line benefit shown in the video.</div>
</div>
```

The `padding-bottom:56.25%` maintains the 16:9 aspect ratio regardless of container width. The `box-shadow` color should match the tool's accent (teal for BugCapture, purple glow for LearnBoard).

After updating:
1. Commit and push to trigger the GitHub Actions deploy.
2. Wait for the Actions run to complete (~1–2 min).
3. Verify with `curl -s https://machina.chat | grep "youtube.com/embed"` — should return both video IDs.

---

## 5. Gotchas

- **Unlisted vs Public:** Unlisted videos embed fine, but won't appear in YouTube search. Set Public before the launch post.
- **Cookie consent:** YouTube embeds trigger a cookie notice in the EU. If this becomes a problem, use `youtube-nocookie.com` as the embed domain instead of `youtube.com`.
- **Autoplay:** Do NOT add `autoplay=1` to the embed URL — it silently fails on most mobile browsers and annoys desktop users.
- **GitHub Pages cache:** After a deploy, CloudFlare or GitHub's CDN may serve stale HTML for up to 5 minutes. Hard refresh (`Ctrl+Shift+R`) or check via `curl` to verify the actual deployed content.

# PromptBoard Demo — Narration Script (ElevenLabs)

**Total duration:** ~68 seconds  
**Tone:** calm, direct, developer-to-developer. Not a sales pitch.  
**Note:** `[Xs]` = seconds from PLAY click — use these to sync audio cuts in DaVinci Resolve.

---

## Script

[0s]
You're about to prompt an AI with a real problem.

[3s]
A bug report. An image. A flow diagram. Specific constraints.

[6s]
One text box isn't going to cut it.

[9s]
This is PromptBoard. A visual canvas where you build prompts — not type them.

[14s]
Add a text block for the task. Type it, or speak it.

[18s]
Add a constraints block. The AI needs to know the limits before it writes a single line.

[23s]
Drag in a screenshot directly from your clipboard.

[27s]
Add flow nodes for the logic you're describing.

[31s]
Connect them with arrows. Label each relationship.

[36s]
Hit the mic on any block — voice dictation transcribes directly into the canvas.

[42s]
When you're done, click Copy prompt.

[45s]
Everything — text, images, flow context — compiled into one structured brief.

[49s]
Paste it into Claude, ChatGPT, or Copilot. The AI reads your intent, your constraints, your visual context. All at once.

[55s]
No more fragmenting complex context across five chat messages.

[58s]
PromptBoard. Build AI prompts visually.

[61s]
Part of Machina. Free, open source. No installation needed.

[65s]
machina dot chat.

---

## ElevenLabs settings

- **Model:** Eleven Turbo v2
- **Voice:** Liam (same as LearnBoard video — calm, clear, developer tone)
- **Stability:** 0.55 · **Similarity:** 0.75 · **Style:** 0.10
- Add 0.5s silence at the start
- Export: **WAV 44.1kHz** for DaVinci Resolve import

## Sync in DaVinci Resolve

Import the WAV, place it on audio track. Use the `[Xs]` markers above to align each sentence to the corresponding scene in `promptboard-autoplay.html`. The scene transitions are JS-timed — audio and visuals should sync naturally after the PLAY button click.

**Key sync points:**
- `[9s]` → PromptBoard UI slides in
- `[14s]` → Goal text block appears + typing starts
- `[18s]` → Constraints block appears
- `[23s]` → Image block drops in
- `[27s]` → Flow nodes appear
- `[31s]` → Arrows animate
- `[36s]` → Voice indicator activates
- `[42s]` → Export panel opens
- `[45s]` → Copy button click + toast
- `[49s]` → AI interface shows
- `[58s]` → CTA card

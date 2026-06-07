# Shorts Producer — Memory

Persistent notes the agent maintains across sessions. Keep concise; update when
the user states a preference, corrects something, or a build reveals a lesson.

## User preferences
- Wants a natural voice (NOT the robotic macOS `say`). Uses local Kokoro neural TTS.
- Wants a **different voice per video**, chosen to fit the narrative.
- **Voice speed: 1.2x** for all shorts. This is the default in `build_short.py` DEFAULTS. Scene hold times automatically match because the builder uses `TTS clip duration + gap` per step.
- **NO background music** — the procedural music sounds monotonic and not musical.
  Focus on visual richness (animations, diagrams, icons) instead.
- Shorts are viewed on mobile → fonts and visuals must be large.
- Respect YouTube UI safe zones (right action rail, bottom title/description) and
  device-crop edge buffers.
- **CRITICAL: Use visual animations, NOT just text.** Every short must include images,
  icons, diagrams, illustrations, or animated graphics. Use HTML/CSS/JS with libraries
  like D3.js, SVG animations, or Canvas to create engaging visual content. Text-only
  animations are not sufficient — viewers need visual richness (system diagrams, data
  viz, animated icons, flowing graphics, etc.).

## Voice guide (Kokoro, en-us / en-gb)
Pick by narrative tone. Sampled favorites:
- `am_fenrir` — US male, deep & dramatic → bold hooks, "trailer" energy.
- `am_puck` — US male, bright & punchy → energetic explainers.
- `am_onyx` — US male, deep & smooth → authoritative, calm gravitas.
- `am_michael` — US male, clear neutral → default explainer.
- `af_heart` — US female, warm & expressive → friendly/inviting.
- `af_bella` — US female, lively → upbeat storytelling. (User liked this one.)
- `af_nicole` — US female, soft/intimate → calm, close-mic.
- `bm_george` / `bm_fable` — UK male → documentary tone.
- `bf_emma` / `bf_isabella` — UK female → crisp, polished.
54 voices total; list with `Kokoro.get_voices()`.

## Music moods (vo_music.py)
`dark` (tense/serious), `tense` (suspense), `calm` (gentle), `bright` (upbeat),
`mystery` (curious). Seed = slug → unique track per video. Mix gain ≈ −24 dB.

## Per-video log (keep variety; avoid repeating the last voice/mood)
| slug | voice | notes |
|---|---|---|
| shorts_fault_injestion_and_chaos_engineering | af_bella | first build; chaos-engineering explainer (lives under presentation/, pre-agent) |
| fault-injection-chaos-engineering | am_fenrir | 49s; visual-rich with SVG animations, D3 flowcharts, system diagrams, particles; dramatic hook (engineer destroying systems) → educational explanation; deep authoritative voice |
| fault-injection-chaos-engineering (v2) | af_bella @ 1.2x | 52.9s; rebuilt from presentation/ plan for mobile (safe-content scaffold, large px fonts, no music); 8 scenes — racks+cable yank, sabotage acts, hacker/engineer reveal, fault vs failure D3 rings+cascade, engineer's job shield, inject syringe, workflow loop, CHAOS reveal. User rejected am_onyx (too dull/slow); af_bella at 1.2x preferred. |
| hw-sw-bugs | bm_george | 75.5s; built from presentation/shorts_hw_and_sw_bugs/; 8 scenes — hook (CATASTROPHIC glitch), D3 disk grid (random then simultaneous failure), leap-second world map + city cascade, SSD int16 counter ramp + overflow brick, comparison cards, takeaway; documentary UK male voice matching the factual/historical tone. |
| tail-latency-amplification | am_puck @ 1.2x | 46.4s; reused the ready-made visual from presentation/shorts_tail_latency_amplification/ (copied index.html/style.css/script.js); 8 scenes — p99 hook, engineer complaint, D3 fan-out (1→10 queries), 10 stacked bars w/ Q7 p99-hit, 10×10 query grid + 1%→10% stat cards, ⚡ amplification reveal, D3 scaling curve, "watch the tail" takeaway; punchy male voice for the gotcha reveal. First short with burned-in karaoke subtitles. |

## Highlighted subtitles (word-by-word karaoke)
Added to `build_short.py` — enabled for ALL shorts by default (no index.html changes needed).
- **Approach: post-processed burn-in (like withsubtitles.com), NOT injected into HTML.**
  After the silent video is recorded, an `.ass` subtitle file is generated and burned in
  with ffmpeg `-vf ass=...` during the final encode. This keeps only a SHORT chunk
  (≈1 line / ≤5 words / ≤26 chars) on screen at a time instead of the whole scene line.
- Style: Arial 54px bold, white before highlight → gold (#FFD700) as each word is "sung",
  thick black outline. Positioned via ASS margins (L60 / R180 / V390) above the YouTube
  safe-bottom zone and clear of the right action-rail.
- Timing: each scene's TTS clip duration is split across its caption chunks; within a
  chunk, `\k` karaoke tags distribute time per word (by word length). Captions only show
  while narration plays (none during inter-scene gaps).
- To **disable** for a specific short: add `"subtitles": false` in that short's `voiceover.json`.
- Implementation: `generate_ass()` builds `_work/subs.ass`; `mux()` burns it in. Constants
  `_SUB_*` (font, colours, margins, chunk sizing) at top of `build_short.py`.
  - Earlier version injected an HTML overlay showing the whole scene line at once — replaced
    because too many lines were visible at a time.

## Lessons learned
- Hold each scene for (TTS clip length + ~0.8s gap); keep sub-animations shorter
  than the spoken line so they finish before advancing.
- Record headless at exactly 1080×1920; the builder hides the HUD and stage chrome.
- Kokoro needs Python 3.12 venv (`.venv-tts`) + espeak-ng; ffmpeg via Homebrew.
- **Music is DISABLED** — procedural music sounds monotonic, not musical. Focus on
  visual richness (animations, diagrams, icons, transitions) instead.
- **Subtitles are burned in via ffmpeg ASS karaoke** (not HTML injection). The builder
  generates `_work/subs.ass`: short ≤5-word chunks, one line at a time, words flip
  white→gold (`\k` tags), placed at MarginV 390 (above the 360px bottom safe zone),
  MarginL 60 / MarginR 180 (clear of the right rail). Toggle with `"subtitles": false`.
- **Reusing a presentation/ deck:** copy index.html/style.css/script.js into the short
  folder, then (1) make init start BLANK at step 0 (builder's first ArrowRight reveals
  scene 1, so narration line N ↔ scene N — presentation decks call `goToStep(1)` on
  init, which misaligns by one), and (2) append an "Agent overrides" CSS block to
  reserve a ~470px bottom band (`padding-bottom`) and hide the deck's own captions that
  duplicate the narration — otherwise they collide with the burned-in subtitle. QA by
  extracting frames from final.mp4 (or a layout-only Playwright screenshot pass).

# Shorts Producer — Memory

Persistent notes the agent maintains across sessions. Keep concise; update when
the user states a preference, corrects something, or a build reveals a lesson.

## User preferences
- Wants a natural voice (NOT the robotic macOS `say`).
- **TTS engine = Chatterbox (Resemble AI), since the CQRS short.** Replaced Kokoro:
  user said Kokoro sounded uninteresting AND mispronounced words (its espeak-ng
  phonemizer garbles acronyms/terms). Chatterbox is an LLM-backbone TTS — more
  natural/expressive + far better text normalization. `vo_tts.py` now wraps
  `chatterbox.tts.ChatterboxTTS` (CPU on this host). See "TTS engine notes" below.
- Wants a **different voice per video**, chosen to fit the narrative.
- **Speed: 1.1x** (applied via ffmpeg atempo since Chatterbox has no speed knob).
  Older Kokoro builds used 1.2x. Scene hold times still match (builder uses TTS
  clip duration + gap per step).
- **Spell out acronyms in narration** (e.g. write `C-Q-R-S`, not `CQRS`) so the TTS
  reads the letters instead of a garbled word.
- **NO background music** — the procedural music sounds monotonic and not musical.
  Focus on visual richness (animations, diagrams, icons) instead.
- Shorts are viewed on mobile → fonts and visuals must be large.
- **USE THE FULL FRAME. Do NOT reserve YouTube-UI safe zones** (right action rail,
  bottom title/description). The user explicitly asked to remove that buffer
  accounting — it was shrinking visuals and leaving empty right/bottom gutters.
  Keep ONLY a small uniform edge margin (~40px, `--safe-edge`) for device-crop
  safety. Fill the whole 1080×1920 canvas.
- **Make icons/images BIG** — these are phone shorts. Enlarge diagram elements to
  fill the width; small icons are a recurring complaint.
- **NO yellow/gold TEXT on slides.** The burned karaoke SUBTITLES are gold
  (`#FFD700`, `_SUB_PRIMARY` in build_short.py), so any gold/yellow on-screen text
  visually collides with them. Keep gold ONLY for flow lines/arrows/markers and
  badge BACKGROUNDS (dark text on gold is fine). For emphasized text use a distinct
  accent — purple `#d2a8ff` (`C.hiText`) works well on the dark bg.
- **Seesaw / balance-scale visuals are WELCOMED.** The user liked the animated
  trade-off seesaw (rocking beam with two pans) in normalization-vs-denormalization
  scene 11. Reuse this kind of metaphor-driven animated visual for "trade-off /
  vs / balance" concepts.
- **Tables: ALWAYS draw a plain, simple titled grid — NO decorative rounded
  "oval"/footer strip at the table bottom.** That bottom strip obstructs the last
  row's text. The `table`/`wideTable` glyphs must be just: colored title header +
  square-bottom body panel + row lines (the panel's own border closes the bottom).
  Do NOT re-add a `rect` at `y + h - 16` with `rx`. Use generous cell fonts
  (~28–30px) so they're legible on mobile. (User asked for this twice.)
- **Tables need MULTIPLE rows (2–5) to be convincing.** A single-row table doesn't
  show WHY normalizing helps — the viewer can't see the duplication being removed.
  Use a small shared dataset and split it so the "lookup" tables stay small
  (customers ×2, products ×3) while the linking table grows (orders ×4); that makes
  "stored once even though referenced many times" visually obvious. Keep the SAME
  rows across the normalize → FK → update → JOIN scenes so positions stay fixed.
- **Use explicit ARROWS** (arrowheads) to show data/control flow, not bare lines.
- **Flow direction is free:** left→right is fine, not always top→bottom. Use the
  full width for two-stage / horizontal relationships when it reads better.
- **CRITICAL: Use visual animations, NOT just text.** Every short must include images,
  icons, diagrams, illustrations, or animated graphics. Use HTML/CSS/JS with libraries
  like D3.js, SVG animations, or Canvas to create engaging visual content. Text-only
  animations are not sufficient — viewers need visual richness (system diagrams, data
  viz, animated icons, flowing graphics, etc.).
- **Diagrams > text. Prefer ANIMATED FLOW DIAGRAMS:** draw the system as nodes
  (writer, DB cylinder, reader, timeline boxes) and animate PACKETS (moving dots)
  travelling between them along edges. The user explicitly wants "image and animation,"
  not text walls. Per scene keep at most a short title + one tiny caption; everything
  else is a moving diagram. Example the user loved: writer→central DB (write), then
  reader query → follows → followee → tweets DB → home timeline → reader (read path).
- **MANDATORY: show the HTML deck for manual review BEFORE building the video.** The
  user opens `shorts/<slug>/index.html` in a browser / VS Code preview and steps
  through it (click or → to advance). Do NOT run TTS/record/mux until they approve —
  building wastes time if visuals are wrong. (Builder still has `--safe-preview` for
  the safe-zone overlay frame. The agent may self-QA by screenshotting scenes via a
  quick headless Playwright pass, but the user-facing gate is the live HTML.)
- **Consistency is critical:** use ONE icon/shape per concept across ALL scenes — same
  table look (a single grid glyph) for every table, the SAME avatar (one 👤 glyph,
  colour-coded per handle) for every user. Never swap icons mid-deck.
- **Big visuals:** match the diagram viewBox aspect to the (now nearly full-frame)
  container so it does NOT letterbox — a viewBox much wider than the tall container
  shrinks everything. Fill the width; make tables/icons large.
- **Explain hard ideas step-by-step:** break a tricky mechanism into numbered scenes
  (1,2,3…) and keep shared elements in FIXED positions across those scenes so each
  transition only adds the next highlight — much easier to follow. Audience may have
  only basic knowledge; favour clarity and length over brevity (2 min is fine).
- **Reusable diagram toolkit** lives in this short's script.js: `mkSvg`, `table`
  (titled grid table w/ rows), `avatar`, `gridGlyph`, `edge`, `caption`, `stepBadge`,
  `flow` (packet along waypoints), `hlRow`, `readStage`. Copy it as a starting point.

## TTS engine notes (Chatterbox — current)
- Env: `.venv-tts` (Python 3.12). Install: `pip install chatterbox-tts soundfile
  numpy playwright` then `pip install "setuptools<81"` (CRITICAL — perth, the
  Chatterbox watermarker, imports `pkg_resources`, removed in setuptools 81+;
  without the downgrade `perth.PerthImplicitWatermarker` is None →
  `TypeError: 'NoneType' object is not callable` at model load). Also
  `python -m playwright install chromium`.
- Weights auto-download from HuggingFace (~3.2 GB) on first `from_pretrained`, cached.
- CPU-only here (no GPU): ~25-30s to synth a ~5s line; a ~75s 11-line short took
  ~3m45s end-to-end (TTS dominates). Fine, just not instant.
- Tone tuned via `exaggeration` (0.3-0.4 = calm/measured; 0.5 default = animated)
  and `cfg_weight` (~0.5 steady pacing) in `vo_tts.DEFAULTS`.
- `voice` param now = optional path to a reference .wav for voice cloning
  (`audio_prompt_path`); `null`/missing → built-in narrator. Kokoro `am_*`/`af_*`
  voice names NO LONGER apply. `lang` is ignored (English model).
- Speed handled by ffmpeg `atempo` chain in `vo_tts.synth` (not the model).

## Voice guide (Kokoro — LEGACY, pre-Chatterbox; names no longer used)
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
| twitter-timeline-fanout | af_heart @ 1.2x | 53.5s; built fresh from template scaffold; interview-framed "Design Twitter" walkthrough (audience cares about the INTERVIEW, not the product). **Rebuilt visuals v2: animated SVG FLOW DIAGRAMS (D3), minimal text** after user said v1 was too text-heavy. 8 scenes — (1) follower network pulsing around an app, (2) writer→DB cylinder write packet, (3) THE read multi-hop: you→follows→tweets DB→feed→you looping query packet (user's exact example), (4) the catch: short green write hop vs long red read zigzag, (5) per-user pre-built feed boxes filling, (6) fan-out: one post packet copies into 6 follower boxes + spinning bg gear, (7) celebrity: 👑 + count-up to 100M + storm of red packets flooding a 14×6 grid + shake, (8) hybrid: push(green)+pull(blue) streams merge→your timeline. Warm female explainer voice; Twitter-blue accent (#1d9bf0). Reusable diagram toolkit in script.js: mkSvg/node/dbCyl/edge/caption/flow(looping packet along waypoints). |
| twitter-timeline-fanout (v3) | af_heart @ 1.2x | **Rebuilt AGAIN to 13 scenes** after user said v2 visuals were small / icons inconsistent / transitions unclear. Now BIG, CONSISTENT tables (one grid-icon table look reused for tweets/follows/home) and ONE avatar (👤 colour-coded) for every user. Read problem explained step-by-step: (1) two tables, (2) post=append a row, (3) read=3 steps, (4) step1 follows highlight, (5) step2 scan tweets for matches, (6) step3 build home view + sort, (7) repeats every refresh=slow, (8) pre-build per-user home tables, (9) fan-out appends "new!" row into each follower home, (10) read=one lookup, (11) celebrity 100M write-storm flooding grid of home tables, (12) hybrid push+pull→merge→timeline, (13) takeaway two tables+phone. viewBox 840×1210 fills screen. NOT YET BUILT — awaiting user's HTML review. |
| twitter-timeline-fanout (v4) | af_heart @ 1.2x | **BUILT — final.mp4 1080×1920, 77.2s.** Same 13-scene structure as v3 but **full-frame + bigger + arrows** after user removed all YouTube-UI buffer accounting. Diagram tables widened to fill the (now nearly full) frame; tables/avatars enlarged; explicit ARROWHEAD markers (incl. gold `hi`) on every edge/flow; hybrid scene laid out LEFT→RIGHT (push left + pull right → merge → timeline). Template/safe_zones/builder all switched to a single ~40px uniform `--safe-edge` (no right-rail/bottom reserve); subtitles now full-width L50/R50/V110. |
| sql-vs-nosql-linkedin | am_michael @ 1.2x | 12 scenes; relational vs NoSQL using one LinkedIn profile. Reusable diagram toolkit in script.js (mkSvg/table/avatar/jsonDoc/arrow/packetOnPath/proCon/chip/badge). SQL=blue, NoSQL=green, links=gold. |
| normalization-vs-denormalization | af_bella @ 1.2x | **BUILT — final.mp4 1080×1920, 100.2s.** 14 scenes; e-commerce order example, conceptual (no 1NF/2NF/3NF jargon). normalize=blue, denormalize=orange, dup/anomaly=red, join/flow=gold; emphasized TEXT = purple `#d2a8ff` (`C.hiText`) to avoid clashing with the gold karaoke subtitles. One 👤 Alice avatar + 📦 product + grid-table + JSON glyph reused; FIXED table positions across steps. **Realistic multi-row tables** (shared dataset: customers ×2, products ×3, orders ×4) so the "stored once but referenced many times" win is visible. Scenes: hook→related-data→fat table→duplication→update anomaly→normalize split→FKs→write win+storage bars→read JOIN→denormalize precomputed view→trade-off SEESAW (animated rocking beam — user loved this)→relational(cylinder+cached read model + "cache that result")→NoSQL(embed+write fan-out ×4)→takeaway cards "Pick per query, not per dogma." Forked the sql-vs-nosql toolkit; added `appearPulse`, `seesaw`/`pulseRed` keyframes, `wideTable` cellSize/colSize opts, plain (no oval-bottom) tables. |
| sql-vs-nosql-linkedin | am_michael @ 1.2x | **BUILT — final.mp4 1080×1920, 82.2s.** "SQL vs NoSQL — one LinkedIn profile, two ways" explainer. Built fresh from template scaffold (no D3 — pure inline-SVG toolkit). 12 scenes — (1) one profile → SQL+NoSQL split, (2) profile = related data (experience/skills/connections chips), (3) relational splits into 4 tables (2×2), (4) users table + PRIMARY KEY badge, (5) experience + FOREIGN KEY curved gold arrow, (6) skills+connections all linked by user id, (7) JOIN converging arrows rebuild profile, (8) relational upside (no dupes + integrity), (9) relational cost (rigid schema + ×N joins), (10) NoSQL one JSON document (nested arrays), (11) NoSQL upside (one read + flexible schema/new field), (12) NoSQL cost (dupes + cross-profile queries) + side-by-side takeaway card. Reusable SVG toolkit in script.js: `E/T` (el+text), `mkSvg` (with arrowhead markers arrGold/arrSql/arrNoSql/arrRed), `avatar`, `profileCard`, `table` (titled grid, keyCol gold highlight, `_geom`), `jsonDoc`, `badge`, `arrow` (drawn path + marker), `proCon`, `chip`, anim helpers `appear/drawArrow/goldPulse`. SQL=blue #58a6ff, NoSQL=green #3fb950, links/joins=gold #ffd700. Clear neutral explainer voice for a balanced comparison; keeps variety vs af_heart/am_puck/bm_george. |
| cqrs-command-query-segregation | Chatterbox (built-in narrator) @ 1.1x | **BUILT — final.mp4 1080×1920, 75.2s.** **FIRST short on Chatterbox TTS** (switched from Kokoro per user: dull + mispronounced). Calm/authoritative architect tone (exaggeration 0.35, cfg_weight 0.5). 11-scene step-by-step CQRS explainer (e-commerce orders). write=blue #58a6ff / read=green #3fb950 / events=gold #ffd700 / problem=red #f85149 / emphasis-text=purple #d2a8ff. Scenes: (1) overloaded single DB w/ write+read arrow storm, (2) COMMAND vs QUERY cards, (3) one shared Order Model + table, (4) opposite needs chips + 1:100 ratio bar, (5) split into WRITE/READ models, (6) command write-path packet, (7) query read-path + denormalized view tables, (8) gold event arrow syncing stores (projection), (9) eventual-consistency lag→catch-up, (10) independent scaling: 1 write vs ×N green read replicas, (11) takeaway ✓diverge/✗CRUD. Narration spells "C-Q-R-S" so letters read cleanly. Inline-SVG toolkit in script.js: E/txt/mkSvg(arrow markers)/box/dbCyl/arrow/drawIn/appear/packet(WAAPI transform)/badge + splitBase fixed-layout helper. Lesson reused: ratio bar grows via transform scaleX, NOT WAAPI width on SVG rect (doesn't tween). |
| star-schema-fact-and-dimensions | af_bella @ 1.2x | **BUILT — final.mp4 1080×1920, 84.1s.** 12-scene **radial-star (D3)** explainer of the star schema; user explicitly asked for af_bella + literal star/snowflake memory anchors. The schema diagram ITSELF is the star: gold `SALES` fact at the CENTER, 4 blue dimension tables on the glowing arm tips (Customer/Product/Store/Date); arms light up one-by-one. Scenes: (1) ⭐ hook, (2) big fact wideTable, (3) FK vs measures colSpan highlights, (4) row counter ramp to 1B+, (5) **name the Star Schema** — dims on arms + a `dimension table` callout badge, (6) inside a dimension (Product detail), (7) arms = dimensions, the shape is a STAR, (8) query: only Product+Date arms light gold → SUM result bars, (9) why analysts love it (intuitive/simple joins/fast — green cards), (10) **reverse drill-down** advantage (green TOTAL pill → drill arrow center→Store → rows highlight), (11) **snowflake**: fact STAYS at center hub (faint star arms), ONE highlighted gold arm → 1st-level dim (Product), one highlighted orange 2nd-level arm → 2nd-level dim (Category) + faint ❄ crystal, (12) takeaway: **a snowflake is just a more normalized star** (STAR → normalize → SNOWFLAKE, no pros/cons diff list). Built on the normalization-vs-denormalization toolkit + D3 v7 (CDN) for staggered arm reveals. fact=gold/dim=blue/snow=orange/good=green. af_bella reused (also normalization short) but user requested it here. |

## Highlighted subtitles (word-by-word karaoke)
Added to `build_short.py` — enabled for ALL shorts by default (no index.html changes needed).
- **Approach: post-processed burn-in (like withsubtitles.com), NOT injected into HTML.**
  After the silent video is recorded, an `.ass` subtitle file is generated and burned in
  with ffmpeg `-vf ass=...` during the final encode. This keeps only a SHORT chunk
  (≈1 line / ≤5 words / ≤26 chars) on screen at a time instead of the whole scene line.
- Style: Arial 58px bold, white before highlight → gold (#FFD700) as each word is "sung",
  thick black outline. Positioned via ASS margins (L50 / R50 / V180) — raised one line
  height up from the old V110 (was too low on screen), full width (we no longer reserve
  the right action-rail / bottom UI — use the whole frame).
- Timing: each scene's TTS clip duration is split across its caption chunks; within a
  chunk, `\k` karaoke tags distribute time per word (by word length). Captions only show
  while narration plays (none during inter-scene gaps).
- To **disable** for a specific short: add `"subtitles": false` in that short's `voiceover.json`.- Implementation: `generate_ass()` builds `_work/subs.ass`; `mux()` burns it in. Constants
  `_SUB_*` (font, colours, margins, chunk sizing) at top of `build_short.py`.
  - Earlier version injected an HTML overlay showing the whole scene line at once — replaced
    because too many lines were visible at a time.

## Lessons learned
- **TWO blocking approval checkpoints — wait for the USER at each, every time.**
  (1) script/plan.md approval, (2) HTML-deck visual approval (user manually steps
  through every slide). My own QA screenshots do NOT count as approval. Mistake I
  made on sql-vs-nosql-linkedin: I self-QA'd the deck then went straight to building
  the video without waiting for the user's slide-by-slide sign-off. The agent file
  now has these as explicit Phase 2 (CHECKPOINT 1) and Phase 3 (CHECKPOINT 2) gates;
  Phase 4 = render only after CHECKPOINT 2.
- **SVG group positioning vs CSS entrance animations COLLIDE.** If a `<g>` is
  positioned with a `transform="translate(…)"` *attribute* and you then run a CSS
  `@keyframes` that animates `transform` (e.g. `pop`/`rise` ending at `scale(1)`),
  the inline animation `transform` OVERRIDES the attribute → the group snaps to
  (0,0) and piles onto the header. Fix: bake the offset into element coordinates
  (pass y into the draw fn) instead of a group transform, OR animate only
  `opacity`. Bit me in sql-vs-nosql-linkedin scenes 8/9/11/12.
- **Don't rely on SVG `animateMotion`/`mpath` packets in headless recording** —
  they rendered as a stray dot + malformed arcs. Prefer drawn arrows
  (`stroke-dasharray` draw-in) for flow; reserve moving packets for live-only.
- **Always QA every scene with a headless screenshot pass BEFORE building** (tiny
  throwaway Playwright script: ArrowRight, wait ~2.6s for entrances, shoot). Caught
  3 layout bugs that would've wasted a full TTS+record+mux cycle.
- **`appear(el)` then `redPulse/goldPulse(el)` clobbers opacity** (second
  `style.animation` overwrites the fade, element stays opacity:0/invisible). Combine
  into ONE `style.animation = 'fade …both, pulseRed …infinite'` (helper `appearPulse`).
  Same trap with a bare `pulse(el)` after `appear()`: the `pulseGold` keyframe only
  animates `filter`, so the leftover inline `opacity:0` from `appear()` keeps the
  element hidden — set `el.style.opacity='1'` at the start of `pulse()` (fixed in the
  star-schema build).
- **SVG presentation-attribute transitions don't tween:** `rect.setAttribute('width',N)`
  + CSS `transition:width` does NOT animate. Use CSS `transform:scaleX()` or a static
  before/after comparison instead.
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

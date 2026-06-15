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
- **Env: `.venv-tts` (Python 3.12) — READY & VERIFIED (June 14, 2026).**
  Install: `pip install chatterbox-tts torchaudio soundfile numpy playwright` then
  `pip install "setuptools<81"` (CRITICAL — perth, the Chatterbox watermarker,
  imports `pkg_resources`, removed in setuptools 81+; without the downgrade
  `perth.PerthImplicitWatermarker` is None → `TypeError: 'NoneType' object is not
  callable` at model load). Also `python -m playwright install chromium`.
- Weights auto-download from HuggingFace (~3.2 GB) on first `from_pretrained`,
  cached in `~/.cache/huggingface/`. **Models downloaded and cached.**
- CPU-only here (no GPU): ~25-30s to synth a ~5s line; a ~75s 11-line short takes
  ~3-4 min end-to-end (TTS dominates). Synthesis verified working.
- Tone tuned via `exaggeration` (0.3-0.4 = calm/measured; 0.5 default = animated)
  and `cfg_weight` (~0.5 steady pacing) in `vo_tts.DEFAULTS` and `build_short.py`.
- `voice` param now = optional path to a reference .wav for voice cloning
  (`audio_prompt_path`); `null`/missing → built-in narrator (high quality default).
  Kokoro `am_*`/`af_*` voice names NO LONGER apply. `lang` is ignored (English model).
- Speed handled by ffmpeg `atempo` chain in `vo_tts.synth` (not the model).
  Default speed now 1.1 (was 1.2 for Kokoro; Chatterbox is naturally more expressive).
- **Migration complete:** All docs updated (README, agent file, template). See
  `CHATTERBOX_MIGRATION.md` for full setup details.

## Music moods (vo_music.py — currently DISABLED by default)
`dark` (tense/serious), `tense` (suspense), `calm` (gentle), `bright` (upbeat),
`mystery` (curious). Seed = slug → unique track per video. Mix gain ≈ −24 dB.

## Per-video log (historical — newer videos use Chatterbox built-in narrator)
**Note:** Videos from "cqrs-command-query-segregation" onwards use Chatterbox TTS.
Older videos used Kokoro with the voice names listed below (for reference only).

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
| property-graphs-connect-everything | Chatterbox (built-in narrator) @ 1.1x | **BUILT — final.mp4 1080×1920, 87.4s.** Illustration-first explainer of the **labeled property graph** model (vertices+edges with labels & properties), from DDIA. Calm narrator. vertex=blue #58a6ff / edge=green #3fb950 / label=gold #ffd700 / property=orange #f0883e / allergen=orange / food=#da7756 / emphasis-text=purple #d2a8ff / bad=red. 12 scenes: (1) hook — mystery vertices+edges connect → "CONNECT EVERYTHING", (2) vertex anatomy (big circle + id/label/property chips + in/out edge stubs), (3) edge anatomy (two nodes + arrow + LIVES_IN label + since prop + tail/head callouts), (4) full 6-node graph (Lucy/Alain/London/Paris/England/France) edges draw in, (5) TWO relational tables (vertices blue + edges green) with gold box round tail/head cols, (6) NO SCHEMA — rogue gold London→France TRADE_PARTNER edge + "Any→Any ✓", (7) traverse forward (looping gold packet Lucy→London→England), (8) traverse backward (packet London←Lucy), (9) EVOLVE add Peanuts/Gluten allergen nodes + ALLERGIC_TO edges + green "+" markers, (10) EXTEND foods Pad Thai/Bread/Salad + FOUND_IN edges, (11) QUERY what's safe for Lucy — packet Lucy→Peanuts→PadThai ✗, Bread/Salad ✓ + result strip, (12) takeaway "BUILT FOR CHANGE" over dimmed live graph + 3 green checks. **v2 illustration-first redesign** after user said v1 "too much text" — removed all summary lists/breadcrumbs/notes, narration+subtitles carry words, tightened viewBoxes per scene so graphs fill frame, suppressed per-node type badges in dense graph (showType flag). Inline-SVG toolkit: mkSvg(markers arrGold/arrGreen/arrRed/arrMut/arrBlue/arrGood + glow filter), vtx/labelBadge/propChip/edgeArrow/drawIn/appear/pop/glowPulse/packet/table(titled grid w/ separate col-header band)/badge + N{} fixed node positions + drawGraph/graphEdge/drawGraphEdges helpers. **LESSON: SVG `marker-end` arrowheads render at the path endpoint IMMEDIATELY even while the line is hidden by stroke-dashoffset — they "stick" oddly at scene start. Fix: in `drawIn`, strip `marker-end` before animating and re-add it on the animation's `finish` event so the arrowhead only appears once the line finishes drawing.** |
| designing-a-hash-map | Chatterbox (built-in narrator) @ 1.1x | **BUILT — final.mp4 1080×1920, 130.6s (~2:11, longest yet — user said time not a hard constraint).** Beginner-friendly, interview-framed explainer of **how Python's dict really works** (open addressing, NOT generic chaining). Animated narrator tone (exaggeration 0.42, cfg_weight 0.5). **ILLUSTRATION-FIRST redesign (v2)** after user rejected v1: "Too much text, text too big, images too small." v2 = each scene is a bare full-frame SVG diagram with only a tiny 54px `kicker` label at top; narration + burned-in subtitles carry all words. **Tall viewBox 1000×1690** matching the full-frame container (no letterbox); `.diagram` absolute full-bleed with 150px bottom clearance for subtitle. 17 scenes: (1) `d["alice"]→42→?`, (2) KEY→VALUE+⚡+O(1) badge, (3) BIG full-width 8-slot array, slot 5 jump, (4) `"apple"→⚙HASH→big number`, (5) number→mod box→drop in slot 3, (6) 3 hash-property rows + clumped-vs-even compare, (7) collision two keys→slot 3 red, (8) chaining (dimmed) vs Python's open-addressing (bright, "one array"), (9) probing cherry slot3→hop slot4, (10) lookup replay hops→found, (11) **compact dict**: sparse indices strip → dense insertion-ordered entries, clean ORTHOGONAL pointers routed through left gutter (v1 had crossing curved arrows over text — rerouted), (12) load factor fill→meter 75%, (13) grow 8→16 rehash dual arrays, (14) shrink 16→8, (15) amortized O(1) chart (green ticks + 1 red resize spike + purple avg line), (16) 4-step interview checklist (numbered badges), (17) trade-off Average O(1) vs Worst O(n) + "O(1) is best-effort, not a guarantee." Inline-SVG toolkit in script.js: mkSvg(viewBox 1000×1690, markers arrGold/arrKey/arrGood/arrBad/arrMuted), rect/txt/line/path/kicker, anims appear/pop/drawLine/flash/glow, slotArray(big sw=114 h=178)/fillSlot/keyChip. key/hash=blue #58a6ff, filled/grow=green #3fb950, collision/worst=red #f85149, probe/flow=gold #ffd700, emphasis TEXT=purple #d2a8ff. **Lesson: for "too much text" feedback, strip ALL h1/captions from HTML, make diagram full-bleed with tall viewBox matching container aspect, keep one small kicker only.** |

| triple-stores-subject-predicate-object | Chatterbox (built-in narrator) @ 1.1x (exaggeration 0.42, cfg_weight 0.5) | **BUILT — final.mp4 1080×1920, 67.3s.** Illustration-first explainer of the **triple store data model** (Subject·Predicate·Object). User pivoted twice: plain S·P·O data-model → rewrote into a "Semantic Web failed comeback" 13-scene arc → **"Redo the slides — more illustrations & transitions, NO semantic web."** Current build drops ALL semantic-web framing, refocuses on the data model. 10 scenes, each a tiny `.kicker` pill + big animated SVG (viewBox 1000×1280; `.safe-content` bottom:430px clears karaoke subs): (1) DB shapes (table/JSON/key-val) collapse via gold arrows into "FACTS about things", (2) S·P·O anatomy blue Subject—gold→green Object + Predicate badge + "= one TRIPLE", (3) two triples build, shared **Bob** glows + dashed "same Bob" link, (4) dot-bg knowledge graph + looping multi-hop gold packet, (5) SPARQL dashed `?who—follows→Bob` pattern + mono SELECT + scan sweep → Alice/Carol chips, (6) schema-free split: relational `ALTER TABLE 🐌` shakes vs one new gold `Alice—founded→AcmeCorp`+✓, (7) inference `Alice isA Employee isA Person` + spinning purple gear + dashed gold **inferred** Alice⇢Person, (8) 2×2 cards 🧠🛒🚨🤖 (AI glows), (9) **real production users** — 4 company rows each itself a triple: 🔍 Google→Knowledge Graph, 🌐 Wikidata→public SPARQL DB, 📺 BBC→World Cup data, 🧬 UniProt→billions of protein facts (blue subject pill —gold arrow→ green object pill), (10) **seesaw** trade-off (Flexibility✓ up / Storage↑·Speed↓ down), (11) takeaway S·P·O reassembles + purple headline. **MORE TRANSITIONS:** per-scene CSS entrance variety on `.safe-content` (t-zoom/t-rise/t-slideL/t-slideR/t-wipe) + draw-ins/pops/glows/packets. Toolkit (script.js): E/txt/mkSvg(markers + dot pattern), pill/node(both multiline)/predBadge(gold bg+dark text)/chip/card/arrow(returns{g,path}, marker added ON draw-finish to avoid premature arrowhead)/dotBg, anims appear/pop(transform-box:fill-box)/drawArrow/glowPulse/spin/packet(WAAPI translate thru points). subj=blue/obj=green/pred=gold(arrows+badges only)/bad=red/warn=orange/emphasis=purple. Narration spells `S-P-A-R-Q-L`/`A-L-T-E-R T-A-B-L-E`/`A-I`. **LESSON: uncommitted short source files were discarded between turns (likely checkpoint restore) — only final.mp4/safezone_preview.png/_work/ survived; the old 13-scene build had actually FINISHED before files vanished. Short `.html/.css/.js/.txt/.json` are NOT git-tracked → a revert wipes them. User also co-edits files between turns — ALWAYS re-read before editing.** |

| hash-index-append-only-log | Chatterbox (built-in narrator) @ 1.1x (exaggeration 0.36, cfg_weight 0.5) | **BUILT — final.mp4 1080×1920, 74.1s. START OF THE LSM-TREE SERIES (#1).** Lives in nested folder `shorts/lsm-tree/hash-index-append-only-log/`. DDIA Ch.3 "Hash Indexes": append-only log (`db_set`) + in-memory hash map (key→byte offset), then its 4 problems, ending with a tease to future videos. 11 illustration-first scenes (tiny `kicker` pill + big SVG): (1) **6-row zebra append-only log** (offset·key,value, "↓ grows downward", append arrow on newest row), (2) reads slow — red scan sweep, (3) hash map right of file, gold link arrows from map LEFT edge to file rows, (4) write path ①append ②update-index (old offset struck out), (5) read path packet lookup→seek→read→result chip, (6) "⚡ MUCH FASTER … but 4 problems" banner + **four EMPTY dashed numbered slots**, (7)-(10) **cumulative reveal** — each scene fills the NEXT problem card (①Space never reclaimed 💾, ②Slow restarts 🔄, ③Must fit in RAM 🧠, ④No range scans 🔍); identical shared layout so the cross-fade looks like one slide animating box-by-box in sync with the spoken problem; newest card pops+glows, icon in top-right corner (cyTop+64) to avoid title overlap, title 46px / detail 34px, (11) takeaway WIN(⚡fast point reads) vs LIMITS columns + "Next videos: we fix all four" gold arrow → "the LSM-tree journey". log/data=blue #58a6ff, hash map/index=green #3fb950, offset/flow=gold #ffd700 (arrows only), problems=red #f85149, emphasis-text=purple #d2a8ff. narration spells "I-O". **REUSABLE PATTERNS:** `problemsScene(el,id,count)` draws an identical banner + 4 fixed-position cards, fills `i<count` (red) else dim dashed slot — clone across scenes with rising count for a one-by-one reveal; `logFile(svg,x,y,w,rows,{rh,zebra})` titled blue panel w/ offset gutter + zebra + stale strikethrough (exposes `_rows/_cx/_rowCY(i)` etc); `hashMap` green key→offset panel. Built from template scaffold; self-QA via `_qa/shoot.py` (Python Playwright, forces 1080×1920, ArrowRight + screenshot per scene). User revisions applied this session: removed an over-scope filesystem-cache scene (12→11), enriched scene-1 log to 6 rows, made the 4 problems appear one-by-one. |
| sstable-sorted-sparse-index | Chatterbox (built-in narrator) @ 1.1x (exaggeration 0.36, cfg_weight 0.5) | **BUILT — final.mp4 1080×1920, 69.6s. LSM-TREE SERIES (#2).** Lives in `shorts/lsm-tree/sstable-sorted-sparse-index/`. Narrator kept CONSISTENT with #1 for series cohesion. Scope: ONLY introduce SSTable indexing (sorted file + blocks + sparse index + seek-and-scan + block compression); table construction/merging/tombstones deliberately teased for later. **Original non-DDIA example to avoid copyright:** keys apple/cherry/grace/grand/grant/mango; 3 blocks [apple,cherry][grace,grand][grant,mango]; sparse index holds only block-start keys (apple→0, grace→40, grant→80); demo lookup `grand` sits between grace & grant. 10 illustration-first scenes (tiny `.kicker` pill + big SVG, per-scene entrance transitions t-zoom/t-slideR/t-rise/t-slideL): (1) recap two red limits from #1 (🧠 must fit in RAM ✗, 🔍 no range scans ✗ — ✗ stamp in card TOP-RIGHT corner not on title line), (2) fix=keep keys **sorted** (jumbled chips→sorted + "✓ each key appears once" + purple "SSTable"), (3) one sorted file centered, (4) grouped into ~4 KB blocks (3 orange brackets), (5) **sparse index** in RAM (green panel left, file right, gold arrows to block-start rows only, non-start rows dimmed, purple note "only the first key of each block" at BOTTOM-center to avoid arrow collision), (6) index tiny vs tall file tower + "≪" + RAM card flips green, (7) lookup **grand** — index grace/grant hot + purple "between" bracket, (8) seek to block then scan→`grand = 63` (gold seek arrow + scan highlight), (9) blocks compress ~4 KB→~1 KB (gzip arrow + hatched blocks + 3 vertically-stacked win/cost chips 💾 less disk / 📉 less I/O / 🔥 a little CPU), (10) payoff 3 green ✓ + purple tease "how SSTables are built & merged" → "the LSM-tree journey continues". Colors: file/SSTable=blue #58a6ff, sparse index=green #3fb950, seek/flow=gold #ffd700 (arrows/markers only), old pain=red #f85149, blocks/compression=orange #f0883e, emphasis-text=purple #d2a8ff (never gold — clashes w/ gold subtitles). narration spells "I-O". **REUSABLE GLYPHS:** `sstFile(svg,x,yTop,w,rows,{rh,blocks,bands,title})` blue titled key→value panel w/ optional orange block separators+bands (pass short `title:'SSTable'` for narrow ≤420px panels or the default title overflows; exposes `_rows/_x/_w/_yTop/_h/_rowCY(i)/_blockTop/_blockBot`); `sparseIndex(svg,x,yTop,w,entries,{rh})` green key→offset RAM panel (e.hot→purple; exposes `_rows/_rowCY(i)/_rightX`). `chip` honors explicit `o.w` — set it when stacking chips so auto-width doesn't overlap. **QA bugs caught & fixed before render:** ✗ stamp overlapping title (→corner), narrow-panel title overflow (→short title), note/arrow collision (note→bottom), 3 bottom chips overlapping (→stacked vertically w/ explicit width). Self-QA via `_qa/shoot.py`.
| lsm-tree-write-path | Chatterbox (built-in narrator) @ 1.1x (exaggeration 0.36, cfg_weight 0.5) | **BUILT — final.mp4 1080×1920, 76.2s. LSM-TREE SERIES (#3).** Lives in `shorts/lsm-tree/lsm-tree-write-path/`. Narrator kept CONSISTENT with #1/#2. DDIA Ch.4 "Constructing & merging SSTables" — the full **write path**: memtable in RAM → flush to sorted segment → multiple segments → background compaction/merge (keep newest value) → WAL for crash recovery. **Original non-DDIA dataset (reused style from #2):** `SEG1`(older)=[apple 91, grand 63, mango 35]; `SEG2`(newer)=[apple 12, cherry 58, grace 17]; `MERGED`=[apple 12, cherry 58, grace 17, grand 63, mango 35]; **`apple` is the overwritten key** (91→12) so compaction/read-newest has a real duplicate to resolve. 9 illustration-first scenes (tiny `.kicker` pill + one big SVG, per-scene entrance transitions zoom/slideR/rise/slideL): (1) **why writes are hard** — sorted blue SSTable + a new key must slot in the MIDDLE (red dashed wedge between grace/mango + curved arrow) → two red ✗ cards "Append breaks the sort order" / "Rewriting it all is too slow", (2) **hybrid idea** — green ⚡append-fast chip + blue 🔍sorted-reads chip → two gold arrows converge into purple "log-structured" + "this is the LSM-tree", (3) **memtable** — arrival chips (any order) → crossing gold arrows sort them into a green memtable (RAM, balanced tree/skip list) "any order in → kept sorted", (4) **flush** — full green memtable + fill meter (scaleY transform, origin bottom — SVG rect height does NOT tween via WAAPI) → gold flush↓ arrow → blue Segment 1 materializes + fresh empty memtable, (5) **segments** — stacked blue Segment 2 (green NEWEST badge) over Segment 1 (grey OLDER badge) + index chips + "older segments are never modified 🔒", (6) **read path** — GET apple: memtable miss (flash red) → Segment 2 hit (apple=12 glows green, result chip ✓) → Segment 1 apple=91 dimmed + red strike "stale" → "newest write wins", (7) **compaction/merge** — two input segments side-by-side → merged segment below (mergesort reveal), duplicate apple: SEG1 91 gets red strike + 🗑, SEG2 12 glows → "duplicate key → keep the newest value", (8) **WAL** — "write: grace,17" forks to green memtable (red 💥 lost on crash) AND orange append-only write-ahead-log; recovery: "crash? replay the log → rebuild the memtable" + "after a flush, that log slice is discarded", (9) **takeaway** — horizontal 3-node pipeline green memtable→blue segments→gold compaction + orange WAL·safety box + 3 green ✓ (append-only writes / reads stay sorted / crash-safe with a log) + purple tease "Next: deletes, tombstones & bloom filters". Colors: segment/SSTable=blue #58a6ff, memtable=green #3fb950, write-problem/stale=red #f85149, WAL=orange #ffa657, seek/flow=gold #ffd700 (arrows/markers only), emphasis-text=purple #d2a8ff. **NEW REUSABLE GLYPHS (forked from #2 toolkit):** `memtable(svg,x,yTop,w,rows,o)` green RAM panel (title default 'memtable · RAM'; exposes `_rows/_x/_w/_yTop/_h/_hh/_rh/_cx/_rowCY/_rowTop`); `walLog(svg,x,yTop,w,entries,o)` orange append-only panel (entries are plain strings; title default 'write-ahead log'); plus `sstFile` blue key/value panel carried from #2. Self-QA via `_qa/shoot.py` (9 scenes, 3600ms each) — caught + fixed scene-9 label/card text overflow (shrank node label 38→33, trimmed win-card strings).

| lsm-tree-tombstones-bloom-filters | Chatterbox (built-in narrator) @ 1.1x (exaggeration 0.36, cfg_weight 0.5) | **BUILT + UPLOADED — final.mp4 1080×1920, 73.6s. LSM-TREE SERIES (#4, series finale of the storage-engine arc).** Lives in `shorts/lsm-tree/lsm-tree-tombstones-bloom-filters/`. Uploaded to Command Line, scheduled 2026-06-18 08:15 UTC (Thu) — https://youtu.be/xZujKuCHBzM. Narrator kept CONSISTENT with #1/#2/#3. DDIA Ch.4 ("Deleting records" → tombstones; "Performance optimizations" → bloom filters): deletes on immutable segments via tombstones, reads/compaction honoring them, then the missing-key read cost solved by per-segment bloom filters. **Original example (continues the key universe):** deleted key = `grace` (17) → tombstone marker `⊘`; missing key for bloom demo = `lemon` vs present `apple`; 12-bit array, `apple` sets bits 1/5/9, `lemon` probes a 0 bit → certain miss. 9 illustration-first scenes (tiny `.kicker` pill + one big SVG, transitions zoom/slideR/rise/slideL): (1) **delete problem** — blue "Segment · on disk" (apple91/cherry58/grace17/mango35), red "DELETE grace" chip, struck grace row + red 🔒 "cannot modify a file on disk", (2) **tombstone** — gold arrow → big red rounded-top headstone glyph `⊘` / grace / DELETED + "a delete is just another write", (3) **write path** — green memtable holds `grace ⊘` → gold flush↓ → Segment 2 (newest, tombstone row red-tinted) over Segment 1 (older, grace,17 blue-tinted) + dashed connector "marker sits on top of the old value", (4) **read honors tombstone** — GET grace → memtable miss (flash) → Segment 2 tombstone glows red "tombstone → stop" → result "grace = not found ✓"; Segment 1 dimmed "never reached" + "newest wins", (5) **compaction drops it** — two input segs converge → both grace rows struck + "grace → dropped 🗑" → merged segment WITHOUT grace, (6) **read-cost problem** — GET lemon (never written) → 4 stacked levels (memtable + Seg3/2/1) each "miss ✗" with a descending red dashed sweep → "not found — after scanning everything", (7) **bloom per segment** — 3 stacked blue segments each w/ a small green "bloom" bit-pill + gold connector, "ask the filter first — skip the disk on a miss", (8) **how it works** — `bitArray` 12 green cells; insert apple → 3 gold hash arrows set bits 1/5/9 (label flips 0→1); probe lemon → red dashed arrows, one hits a 0 cell → ✗ "a 0 bit → definitely NOT here → skip disk" + "all bits set ⇒ maybe present. a 0 ⇒ certain miss", (9) **takeaway** — "two final pieces" + 2 ✓ cards (Tombstones → safe deletes on immutable files / Bloom filters → skip pointless disk reads) + engine badges LevelDB·RocksDB·Cassandra + purple "this is how real databases store data". Colors identical to #3 (segment=blue, memtable/bloom=green, problem/deleted/tombstone=red, WAL=orange, flow=gold arrows-only, emphasis=purple). **NEW REUSABLE GLYPHS:** `tombstone(svg,cx,cyTop,w,h,key)` red rounded-top headstone w/ `⊘`+key+DELETED; `bitArray(svg,x,y,n,{cw,gap,h,index})` green-outlined bit cells w/ hidden fills + `setBit(g,i,delay)` that pops the fill and flips the 0→1 label; `strikeRow(svg,x1,cy,x2,delay)` animated red strike line. `setBit` flips the label via setTimeout (mid-animation), not WAAPI. Self-QA caught 2 bugs: scene-8 bottom note too wide (clipped both sides → shortened + size 33→34) and scene-9 header duplicated the kicker text ("the LSM-tree, complete" → "two final pieces"). Image viewer CACHES by path — copy a screenshot to a NEW filename to force a fresh view after re-rendering QA. |

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

## YouTube publishing (yt_uploader)
- **PHASE 5 — PUBLISH (new, AFTER Phase 4 render).** Once `final.mp4` exists, the
  workflow gains one more process step: schedule the short to YouTube via the
  uploader. This is the LAST phase.
- **CHECKPOINT 3 — FINAL-VIDEO verification gate (BLOCKING).** This is a SEPARATE,
  additional gate that sits AFTER Phase 4 renders `final.mp4` and BEFORE Phase 5
  upload — even after CHECKPOINT 2 (HTML deck) already passed. The user does NOT
  approve from the HTML alone: they download the actual rendered `final.mp4` and
  watch it. Wait for their explicit "good to publish" on the VIDEO. My own QA /
  screenshots do NOT count, and HTML-deck approval (CHECKPOINT 2) does NOT substitute
  for this. Always confirm the TITLE with the user here too. Only after their "yes"
  do I run the real upload. Order of gates: CHECKPOINT 1 (plan) → CHECKPOINT 2 (HTML
  deck) → Phase 4 render → CHECKPOINT 3 (download & watch final.mp4) → Phase 5 upload.
- **Default publish flow once the video is approved:** add the verified short to the
  manifest in `yt_upload_config_sysdesign.yaml` (file/title/description), run
  `--dry-run` to confirm channel + schedule, then run the real upload. One video per
  day at the next free 08:15 slot. Always end by recording the upload (video id +
  slot) in this file.
- **Channel: "Command Line"** (id `UCmCl3D8M62lTCCj0a3hrLeA`). The account owns
  multiple channels, so the uploader has a `channel_name` guard that aborts unless
  the authenticated channel title matches (case-insensitive). Note the real title
  has a SPACE: "Command Line", not "CommandLine".
- **System-design shorts config:** `code/yt_uploader/yt_upload_config_sysdesign.yaml`.
  OAuth `client_secrets.json` lives in `code/shorts_agent/` (config points there via
  relative `client_secrets_file`); `token.json` cached in `code/yt_uploader/`.
- **Schedule policy:** one video per day at **08:15** (config `schedule_hour/minute`),
  starting the **next free day** = day after the latest already-scheduled `publishAt`
  across the channel's uploads (or tomorrow if none). Server clock is UTC → 08:15 UTC.
- **How to run:** `source .venv/bin/activate` (deps live in repo-root `.venv`, NOT
  `.venv-tts`), then
  `python code/yt_uploader/yt_uploader.py --config code/yt_uploader/yt_upload_config_sysdesign.yaml`.
  Add `--dry-run` first to verify channel + print schedule without uploading.
- **Uploader supports a `videos:` manifest** (per-video file/title/description), optional
  empty `playlist_id` (skips playlist add), `defaultLanguage`+`defaultAudioLanguage=en`,
  private + `publishAt` scheduling, and OAuth on FIXED port 8080 with `open_browser=False`
  (random port `port=0` didn't forward on the remote server — user opens the printed URL,
  must forward port 8080 in VS Code Ports). Uploaded files are moved to per-folder `ARCHIVE/`.
- **Per-video `publish_at` override:** a manifest entry may set `publish_at: "YYYY-MM-DD HH:MM"`
  (local tz) to pin an exact slot (e.g. an afternoon 15:15 video); entries WITHOUT it
  auto-fill the next free 08:15 days, skipping any day already taken by an explicit slot.
  Two videos CAN share a date at different times (used Fri morning + Fri 15:15).
- **LSM-tree series uploaded** to Command Line, scheduled (all 08:15 UTC):
  - 06-15 — How Databases Index Your Keys | Hash Index + Append-Only Log — https://youtu.be/kyaUqEEHoi8
  - 06-16 — How SSTables Make Reads Fast | Sorted Files + Sparse Index — https://youtu.be/suY-vBMoPDk
  - 06-17 — LSM Tree Write Path | Memtables — https://youtu.be/CBWxxbg-tKA
  - 06-18 (Thu) — How LSM Trees Delete Data | Tombstones + Bloom Filters — https://youtu.be/xZujKuCHBzM (uploaded 2026-06-14; explicit `publish_at` set in manifest)
- **Manifest caveat:** the uploader hard-exits (`sys.exit`) if ANY `videos:` entry's
  `file` is missing. Uploaded files get moved to per-folder `ARCHIVE/`, so before each
  new upload REPLACE the manifest with ONLY the not-yet-uploaded video (don't leave the
  archived ones in). Set an explicit `publish_at: "YYYY-MM-DD HH:MM"` (interpreted in the
  server's local tz = UTC) when a specific day is required, instead of relying on auto.
- **Graph data models uploaded (2026-06-14)** to Command Line, config
  `yt_upload_config_graphdata.yaml`, scheduled:
  - 06-19 08:15 — Triple Stores Explained | Subject - Predicate - Object — https://youtu.be/YVkyehfz-aY
  - 06-19 15:15 — Property Graphs | Connect Everything with Vertices and Edges — https://youtu.be/8AQ4y-5EIJ8

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
- **Environment ready:** `.venv-tts` with Chatterbox TTS, all dependencies installed.
  See `CHATTERBOX_MIGRATION.md` for setup details. Synthesis verified working.
- **Music is DISABLED** — procedural music sounds monotonic, not musical. Focus on
  visual richness (animations, diagrams, icons, transitions) instead.
- **Subtitles are burned in via ffmpeg ASS karaoke** (not HTML injection). The builder
  generates `_work/subs.ass`: short ≤5-word chunks, one line at a time, words flip
  white→gold (`\k` tags), placed at MarginV 390 (above the 360px bottom safe zone),
  MarginL 60 / MarginR 180 (clear of the right rail). Toggle with `"subtitles": false`.
- **`"music": null` in voiceover.json CRASHES the build** at `mix_music`
  (`AttributeError: 'NoneType' object has no attribute 'get'`). To disable music use
  `"music": {"enabled": false}` or OMIT the key (DEFAULTS already has
  `{"enabled": False, ...}`). The config loader only deep-merges `music` when it's a
  dict; a bare `null` overwrites the default dict with None. Bit me on the
  triple-stores build (all TTS finished, then crashed before record/mux).
- **Reusing a presentation/ deck:** copy index.html/style.css/script.js into the short
  folder, then (1) make init start BLANK at step 0 (builder's first ArrowRight reveals
  scene 1, so narration line N ↔ scene N — presentation decks call `goToStep(1)` on
  init, which misaligns by one), and (2) append an "Agent overrides" CSS block to
  reserve a ~470px bottom band (`padding-bottom`) and hide the deck's own captions that
  duplicate the narration — otherwise they collide with the burned-in subtitle. QA by
  extracting frames from final.mp4 (or a layout-only Playwright screenshot pass).

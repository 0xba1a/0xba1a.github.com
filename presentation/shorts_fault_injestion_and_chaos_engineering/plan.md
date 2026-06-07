# Fault Injection & Chaos Engineering — YouTube Short (60s)

**Format:** 9:16 (1080×1920), HTML/CSS/JS with D3.js animations, click-through scenes
**Topic:** Why elite engineering teams break their own systems on purpose — fault vs failure, fault injection, and chaos engineering
**Reference raw script:** `plan.original.md`

---

## Voiceover Script (~150 words, ~60s at natural pace)

> Somewhere inside Google, an engineer is killing services on purpose. Pulling network cables. Cutting power to a top-of-rack switch. Unplugging an entire data center.
>
> No, they are not under attack. They are running an experiment.
>
> Here is the distinction every engineer should know. A **fault** is one component going bad — a disk dies, a service crashes, a link drops. A **failure** is when the fault takes down the whole system.
>
> You cannot stop your components from being faulty. But as an engineer, you must stop faults from becoming failures.
>
> So how do you know your system actually survives? Only one way — break it on purpose. Inject the fault. Watch how it heals. Find the cracks before your customers do.
>
> That is fault injection. Do it continuously, in production, with discipline — and it has a name.
>
> **Chaos engineering.**

---

## Scene Plan (8 click-through steps)

### Scene 1 — Hook (0s–6s)
**Visual:** Dark background. A datacenter row of server racks (six rectangles in a row, soft green LEDs blinking) appears. A red gloved "hand" cursor swings in from the right and yanks a network cable out of one rack — sparks fly briefly. Title slams in:
`They break it on PURPOSE.`
**Voiceover:** "Somewhere inside Google, an engineer is killing services on purpose."
**D3 / animation notes:**
- Six racks as `<rect>` with stacked tiny `<circle>` LEDs (CSS keyframe blink, 1.2s loop).
- Cable as an SVG `path` with `stroke-dasharray` reveal; on "yank", animate `transform: translateX` away from the rack and toggle the rack LEDs to red.
- Spark = 6 short outward `<line>` strokes with `stroke-dashoffset` reveal + fade.
- Title with a heavy `text-shadow` + 1-frame red flash on the word `PURPOSE`.

---

### Scene 2 — Three Acts of Sabotage (6s–14s)
**Visual:** Three icons appear in sequence, each with a label fading in below:
1. A microservice node (hexagon) with a red `kill -9` tag → dims out.
2. A network cable being unplugged → red `X` overlay.
3. A whole datacenter building → power switch flips to OFF, lights go dark.
A counter at the top: `Faults injected: 1 → 2 → 3`.
**Voiceover:** "Pulling network cables. Cutting power to a top-of-rack switch. Unplugging an entire data center."
**D3 notes:**
- Three `<g>` groups laid out vertically; each enters with `opacity 0 → 1` + small `scale(0.9 → 1)` over 350ms, staggered 700ms apart.
- The datacenter "lights out" effect: an array of small `<rect>` windows that flip from `#3fb950` to `#21262d` with a stagger sweep.
- Counter as a monospace `<text>` updated inside each scene's `.on("start")` callback.

---

### Scene 3 — "Not an Attack" Reveal (14s–20s)
**Visual:** Two side-by-side faces / labels:
- Left card (red): `HACKER` — strikethrough animates across it.
- Right card (green): `ENGINEER` — pulses, with a small clipboard / experiment-flask icon.
A banner across both: `It's an experiment.`
**Voiceover:** "No, they are not under attack. They are running an experiment."
**D3 notes:**
- Two `<g>` cards slide in from outside the stage.
- Strikethrough = `<line>` with `stroke-dasharray` reveal over 500ms.
- Engineer card scale-pulse 1.0 → 1.04 → 1.0, 700ms ease-in-out.

---

### Scene 4 — Fault vs Failure (20s–32s)
**Visual:** Split screen.
- **Left half — FAULT:** A small system diagram (5 service nodes in a ring connected by edges). One node turns red and shrinks out. The rest auto-route around it; the system stays green. A label `FAULT` underlines that one node.
- **Right half — FAILURE:** Same diagram. One node turns red. A red wave propagates outward through every edge, turning every node red one by one. The whole system collapses. Label: `FAILURE`.
A summary line at the bottom: `Fault = 1 component bad · Failure = whole system down`.
**Voiceover:** "Here is the distinction every engineer should know. A fault is one component going bad — a disk dies, a service crashes, a link drops. A failure is when one fault takes down the whole system."
**D3 notes:**
- Layout each ring with `d3.range(5)` and polar coordinates. Edges as `<line>` between adjacent + diagonal node pairs.
- **Fault side:** transition the bad node to red + `r → 0`; surviving edges thicken slightly to imply rerouting; remaining nodes pulse green once.
- **Failure side:** propagation cascade — recursive `setTimeout` walks the adjacency, recoloring nodes and edges red with 200ms steps. End with all nodes red and a thin red flash overlay on the right half.
- Summary line fades in at the end of the scene with a small upward translate.

---

### Scene 5 — The Engineer's Job (32s–40s)
**Visual:** A single bold equation centered on screen, drawn in a code-card style:
```
Faulty components  ✓  unavoidable
Failed system      ✗  not on your watch
```
Below, a small shield icon with a checkmark forms around the second line.
**Voiceover:** "You cannot stop your components from being faulty. But as an engineer, you must stop faults from becoming failures."
**D3 notes:**
- Two-line code card; left column (status) animates first, then right column (text).
- Shield = SVG path with `stroke-dasharray` reveal (700ms), then fill-in.
- Subtle background grid (5% opacity) for "engineering" feel.

---

### Scene 6 — Inject the Fault (40s–48s)
**Visual:** A clean system diagram (3×3 grid of service nodes, all green). A syringe-style icon labeled `inject()` glides in from the top, hovers over a random node, and "injects" red ink that spreads through that node only. Nearby nodes flicker yellow (under stress), then settle back to green. The injected node remains red but **isolated** — circuit breaker indicator (a small open switch) appears on its edges.
A side meter shows `Recovery: …` ticking from `100ms` to `200ms` and stops.
**Voiceover:** "So how do you know your system actually survives? Only one way — break it on purpose. Inject the fault. Watch how it heals. Find the cracks before your customers do."
**D3 notes:**
- Syringe as a pre-built SVG path; `transform: translate` tween to target node; needle stroke extends with `stroke-dasharray`.
- Ink spread: a small `<circle>` at the node, growing `r` while fading from `#f85149` to transparent.
- Neighbor flicker: short `attr("fill", "#f0883e")` toggle, 3 cycles of 120ms, then back to `#3fb950`.
- Circuit breaker = an `<line>` cut into two segments with a small gap, drawn on each edge of the bad node.
- Recovery counter: `tween("text", ...)` from 0 to 200, with `ms` suffix.

---

### Scene 7 — The Practice (48s–55s)
**Visual:** A timeline / workflow strip with three stacked blocks animating in:
```
Inject → Observe → Fix → repeat
```
Above it, three pill-tags labeled `Continuously · In Production · With Discipline`. Each tag pulses on as it is mentioned.
**Voiceover:** "That is fault injection. Do it continuously, in production, with discipline — and it has a name."
**D3 notes:**
- Workflow blocks: rounded `<rect>`s connected by short arrows (`marker-end`); blocks fade in left-to-right, 220ms apart.
- The `repeat` arrow loops back to `Inject` — animate by drawing an arc path with `stroke-dashoffset`.
- Tags: pill-shaped `<rect>`s with rounded corners; pulse via CSS keyframe (scale 1.0 → 1.06 → 1.0, 600ms).

---

### Scene 8 — The Reveal & Takeaway (55s–60s)
**Visual:** Everything dims to dark. Two words land hard, one after the other, in huge type:
`CHAOS`
`ENGINEERING`
A subtle red→blue gradient sweeps across the letters. Below them:
```
Break it on purpose.
So it never breaks by accident.
```
Channel handle / subscribe CTA fades in at the bottom.
**Voiceover:** "Chaos engineering."
**D3 notes:**
- Title words: separate `<div>`s with scale-in (1.4 → 1.0) + opacity transition, 350ms apart.
- Gradient sweep using a CSS `background-clip: text` + `linear-gradient` whose `background-position` animates over 1.2s.
- Subtitle and CTA cascade in with 200ms staggers.

---

## Color Scheme (GitHub Dark)

| Element                  | Color     | Hex       |
|--------------------------|-----------|-----------|
| Background               | Dark      | `#0d1117` |
| Healthy / system OK      | Green     | `#3fb950` |
| Faulted / failed         | Red       | `#f85149` |
| Stressed / under load    | Orange    | `#f0883e` |
| Engineer / experiment    | Blue      | `#58a6ff` |
| Highlight / discipline   | Yellow    | `#ffd700` |
| Dim text / inactive      | Gray      | `#8b949e` |
| Bright text              | White     | `#e6edf3` |
| Accent / chaos title     | Purple    | `#d2a8ff` |

---

## Data & Constants

- **Scene 1 racks:** 6 racks, each with 4 LEDs vertically. 1 rack chosen as the "yanked" target (deterministic, index 3).
- **Scene 2 sabotage steps:** 3 events, 700ms apart. Counter ticks 1 → 2 → 3.
- **Scene 4 fault diagram:** 5 nodes in a ring, fully connected pentagon (10 edges). Same topology mirrored on left and right.
- **Scene 4 failure cascade:** propagation depth ~3 hops, 200ms per hop → entire ring red within ~1s.
- **Scene 6 grid:** 3×3 service nodes (9 total). Inject target: deterministic index 4 (center) for visual symmetry.
- **Scene 6 recovery counter:** ticks 0 → 200ms over 1.2s, easing `easeQuadOut`.
- **Scene 7 workflow:** 3 forward blocks (`Inject`, `Observe`, `Fix`) + a curved `repeat` arrow back to `Inject`.

---

## Technical Notes

- **Engine:** D3.js v7 for transitions, scales, layouts, and selections. CSS keyframes for shake / glitch / blink.
- **Timing:** Each scene advances on click (consistent with the rest of `presentation/shorts_*`). Internal sub-animations auto-play on scene enter.
- **Aspect ratio:** 1080×1920 vertical. Stage uses CSS `aspect-ratio: 9 / 16`.
- **Fonts:** `JetBrains Mono` / `Fira Code` for code cards and counters; `Inter` (or system sans) for narrative labels.
- **Recording:** Screen-record the page at vertical resolution; overlay voiceover and a low ambient drone in post.
- **Accessibility / replay:** keyboard `→` advances, `r` or `←` resets (matches `shorts_avg_median_percentile` and `shorts_hw_and_sw_bugs`).

---

## Fact-check anchors (for the dev agent)

- **Chaos engineering origin:** Term coined at Netflix; their tool **Chaos Monkey** (2011) randomly terminates production VMs. The broader **Simian Army** added latency injection, zone outages, etc. Google's internal **DiRT (Disaster Recovery Testing)** program runs cross-datacenter failure drills annually. The VO uses Google as the framing because the user's draft does — it's accurate; both companies practice this. Avoid claiming a specific tool name on screen unless desired.
- **Fault vs failure:** Standard reliability-engineering distinction (IEC 61508, Avizienis et al. 2004 "Basic Concepts and Taxonomy of Dependable and Secure Computing"). A fault is the underlying defect; a failure is the deviation of delivered service from correctness. Keep this exact framing in the VO.
- **"In production, with discipline":** Aligns with Netflix's published Principles of Chaos Engineering — minimize blast radius, run in production, automate experiments, learn from steady-state hypotheses.

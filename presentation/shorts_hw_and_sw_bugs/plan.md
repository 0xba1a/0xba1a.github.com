# Hardware Bugs vs Software Bugs — YouTube Short (60s)

**Format:** 9:16 (1080×1920), HTML/CSS/JS with D3.js animations, click-through scenes
**Topic:** Why correlated software failures are far more dangerous than random hardware failures — and why fault-injection testing matters
**Reference raw script:** `plan.original.md`

---

## Voiceover Script (~150 words, ~60s at natural pace)

> Hardware bugs are bad. Software bugs are catastrophic.
>
> A poorly manufactured hard disk has a higher failure rate. One drive dies here, another dies there — scattered, random, annoying. But your RAID handles it. Your replicas handle it. The system stays up.
>
> Software failures don't look random. They look like a coordinated attack.
>
> On June 30th, 2012, a single leap second slipped past the Linux kernel. Servers across the planet hung at the same instant. Reddit, LinkedIn, Yelp, The Pirate Bay — half the internet went down together.
>
> Years later, an integer overflow inside an SSD firmware made the drive brick itself after exactly 32,768 hours of uptime. Identical drives, deployed on the same day, all died on the same second. Entire data centers along with your data — gone.
>
> One bad disk is bad luck. One bad line of code is every disk, everywhere, at the same moment.
>
> Design for it. Inject faults. Test the impossible.

---

## Scene Plan (8 click-through steps)

### Scene 1 — Hook (0s–5s)
**Visual:** Dark background. Title punches in on two lines:
`Hardware bugs are bad.`
`Software bugs are CATASTROPHIC.`
The word `CATASTROPHIC` glitches/shakes red. Subtitle fades in below: `Why correlated failures kill systems.`
**Voiceover:** "Hardware bugs are bad. Software bugs are catastrophic."
**D3 / animation notes:** Pure CSS keyframe — text scale-in (1.0 → 1.05 → 1.0) and a quick `text-shadow` glitch on `CATASTROPHIC` (RGB split, 200ms loop, 3 iterations).

---

### Scene 2 — Random Hardware Failures (5s–14s)
**Visual:** A grid of ~80 small server/disk icons (10×8) fills the screen. Over time, individual disks turn red and fade out — randomly, one at a time, scattered across the grid. A counter ticks: `Failed: 1 … 2 … 3 … 5`. Most of the grid stays green.
**Voiceover:** "A poorly manufactured hard disk has a higher failure rate. One drive dies here, another dies there — scattered, random, annoying. But your RAID handles it. Your replicas handle it. The system stays up."
**D3 notes:**
- `d3.range(80)` → grid of `<rect>` cells, computed via `i % 10` and `Math.floor(i/10)`.
- Pick 6 random indices, stagger their failure: `selection.transition().delay((d,i) => i*1200).attr("fill", "#f85149")`.
- Counter as a `<text>` element updated inside each transition's `.on("start", ...)`.
- Background keeps a calm green pulse (`#3fb950`) on remaining cells.

---

### Scene 3 — The Twist (14s–19s)
**Visual:** Same grid, but a hard cut: every server icon turns red **simultaneously** in a single frame. Screen shake. Text slams in: `Software failure ≠ random.`
Subtitle: `It's a coordinated attack.`
**Voiceover:** "Software failures don't look random. They look like a coordinated attack."
**D3 notes:**
- `d3.selectAll(".cell").transition().duration(120).attr("fill", "#f85149")` — no stagger.
- CSS `@keyframes shake` on the SVG container (translate ±6px, 6 iterations).
- Title uses a heavier font weight + red underline drawn with `stroke-dasharray` reveal.

---

### Scene 4 — The Leap Second Bug (19s–30s)
**Visual:** A large digital clock front-and-center showing `23:59:59 UTC`. It ticks to `23:59:60` (the leap second) — the digits flicker red. A world map appears below; dots representing servers (Reddit, LinkedIn, Yelp, Mozilla, Cloudflare nodes) light up in green, then **all turn red within ~1 second** of the clock hitting `23:59:60`.
Date label: `June 30, 2012`. Caption: `Linux kernel · futex deadlock`.
**Voiceover:** "On June 30th, 2012, a single leap second slipped past the Linux kernel. Servers across the planet hung at the same instant. Reddit, LinkedIn, Yelp — half the internet went down together."
**D3 notes:**
- Clock as styled `<text>` with `font-family: monospace`. Use `d3.interval` ticking every 1s; on hitting `:60`, swap fill to `#f85149` and trigger map cascade.
- World map: a simplified SVG path of continents (or use `d3-geo` with `geoNaturalEarth1` projection on a small TopoJSON, ~50KB). Pre-place ~12 city dots with hardcoded `[lon, lat]`.
- Server dots: green circles → red via `d3.transition().duration(200)`, all fired in the same tick.
- Add a faint red ripple (`<circle>` with growing `r` and fading opacity) emanating from each dot.

---

### Scene 5 — Setup for SSD Bug (30s–37s)
**Visual:** Transition to a clean view. A single SSD icon centered, with a large hour counter beneath it ticking up rapidly: `00001 hrs … 10000 hrs … 20000 hrs …`. The number is rendered in monospace. Below the counter, a binary-bits row appears showing the value in 16-bit signed form.
Caption: `SSD firmware uptime counter (int16)`.
**Voiceover:** "Years later, an integer overflow inside an SSD firmware…"
**D3 notes:**
- Counter: `d3.transition().duration(4000).tween("text", function() { const i = d3.interpolateNumber(0, 32768); return t => this.textContent = Math.floor(i(t)).toLocaleString() + " hrs"; });`
- Binary row: 16 small `<rect>` boxes, each toggled based on the current integer value. Use `(n >> bit) & 1` to set fill (`#58a6ff` for 1, `#21262d` for 0).

---

### Scene 6 — The Overflow (37s–46s)
**Visual:** Counter hits `32,768`. The 16-bit row shows `1000 0000 0000 0000` — the sign bit flips. The counter snaps to `-32,768` and turns red. The single SSD icon clones into a grid of ~40 SSDs (representing a data center). Every single one flashes red and shows a `BRICKED` overlay — at the **same moment**.
Caption: `Same firmware. Same boot day. Same death second.`
**Voiceover:** "…made the drive brick itself after exactly 32,768 hours of uptime. Identical drives, deployed on the same day, all died on the same second. Entire data centers — gone."
**D3 notes:**
- Highlight the sign bit with a yellow stroke right before the flip.
- Use `selection.transition().duration(200).attr("fill", "#f85149")` on all SSDs simultaneously.
- A short screen-shake (CSS class toggle) and a low-frequency "thud" implied by the visual jolt (no audio in plan, but leave a beat in VO timing).

---

### Scene 7 — The Lesson (46s–54s)
**Visual:** Two cards side by side:
```
HARDWARE BUG          SOFTWARE BUG
1 disk dies           Every disk dies
random time           same second
RAID saves you        Nothing saves you
```
The right card is bordered in red, pulses once. An arrow connects them with the label `correlation = blast radius`.
**Voiceover:** "One bad disk is bad luck. One bad line of code is every disk, everywhere, at the same moment."
**D3 notes:**
- Two `<g>` cards, slide in from left and right (`transform: translateX`).
- Pulse on right card: scale 1.0 → 1.04 → 1.0 over 600ms.
- Arrow as an SVG path with `marker-end` arrowhead, drawn via `stroke-dasharray` reveal.

---

### Scene 8 — Takeaway (54s–60s)
**Visual:** Clean dark card with the actionable rule:
```
Design for correlated failure.
Inject faults. Test the impossible.

  ▸ Chaos engineering
  ▸ Fault injection
  ▸ Diversify firmware / kernel versions
```
Channel handle / subscribe CTA fades in at the bottom.
**Voiceover:** "Design for it. Inject faults. Test the impossible."
**D3 notes:** Pure HTML/CSS card with staggered `opacity` transitions on each bullet (200ms apart).

---

## Color Scheme (GitHub Dark)

| Element              | Color     | Hex       |
|----------------------|-----------|-----------|
| Background           | Dark      | `#0d1117` |
| Healthy / green      | Green     | `#3fb950` |
| Failure / red        | Red       | `#f85149` |
| Accent / highlight   | Orange    | `#f0883e` |
| Active bit / link    | Blue      | `#58a6ff` |
| Warning sign-bit     | Yellow    | `#ffd700` |
| Dim text / inactive  | Gray      | `#8b949e` |
| Bright text          | White     | `#e6edf3` |

---

## Data & Constants

- **Hardware grid:** 80 cells (10 × 8). Random failures: 6 cells, picked once on scene enter; stagger 1.2s apart over Scene 2.
- **Leap second cities:** Reddit (Virginia), LinkedIn (Sunnyvale), Yelp (San Francisco), Mozilla (Mountain View), Cloudflare (12 PoPs spread across continents). Hardcode `[lon, lat]` pairs.
- **SSD count in Scene 6:** 40 (8 × 5 grid).
- **Counter target:** 32,768. Animate over ~4 seconds, then overflow snap to `-32,768`.
- **16-bit row:** little-endian visual MSB→LSB. Sign bit on the far left.

---

## Technical Notes

- **Engine:** D3.js v7 for transitions, scales, and the world map. CSS keyframes for shake/glitch effects.
- **Timing:** Each scene advances on click (consistent with `shorts_avg_median_percentile`). Internal sub-animations auto-play on scene enter.
- **Aspect ratio:** 1080×1920 vertical; SVG `viewBox="0 0 1080 1920"`.
- **Fonts:** `JetBrains Mono` / `Fira Code` for counters and code cards; `Inter` for narrative labels.
- **World map asset:** Use a tiny TopoJSON (`world-110m.json`, ~100KB) with `d3-geo` + `topojson-client`. Or pre-render the path strings to keep dependencies minimal.
- **Recording:** Screen-record at 1080×1920 60fps; overlay voiceover and a subtle ambient drone in post.

---

## Fact-check anchors (for the dev agent)

- **Leap second, June 30, 2012:** Real. Linux kernel `hrtimer` / futex bug caused high CPU and hangs on many distros; Reddit, LinkedIn, Yelp, Mozilla, FourSquare, StumbleUpon all reported outages. "Half the internet" is rhetorical — keep the named brands accurate.
- **SSD 32,768-hour bug:** Real. HPE SAS SSDs (firmware HPD7 and earlier), 2020 advisory. Drives became unrecoverable after exactly 32,768 hours of power-on time due to a signed-32-bit-style integer issue. Identical fleets deployed together died together. Avoid naming the vendor in the VO unless desired; the visual makes the point.

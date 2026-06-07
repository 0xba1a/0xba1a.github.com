# Short: Tail Latency Amplification

**Format:** 9:16 (1080×1920) YouTube Short, ~55s, HTML/CSS/JS animation + voiceover
**Theme:** GitHub Dark — `#0d1117` background, blue `#58a6ff`, slow/red `#f85149`, fast/green `#3fb950`, yellow `#ffd700` for emphasis
**Raw idea preserved in:** [plan.original.md](plan.original.md)

---

## Polished Voiceover (~150 words, ~55s)

> Your database has a p99 of twenty milliseconds.
>
> Only one in a hundred requests is slow. You feel pretty good about yourself.
>
> But your backend engineer is yelling that the service feels sluggish. How?
>
> Here's the catch. Every user search fans out into ten database queries. And the response can't go back until all ten are done.
>
> Ten queries per user. So a hundred queries across ten users.
>
> One of those hundred will land in the slow tail. And that one slow query drags an entire user request with it.
>
> Suddenly, one in ten users sees a slow response.
>
> Your one-percent problem just became a ten-percent problem.
>
> This is **tail-latency amplification**. The more you fan out, the worse it gets.
>
> So when you design a service — don't just stare at the average. Watch the tail. Because at scale, the tail is what your users actually feel.

---

## Scene Plan

### Scene 1 — Hook (0s–4s)
**Visual:** Dark background. A big bold metric punches in:
```
p99 = 20 ms
```
A green check `✓` flashes next to it. Subtitle fades in: *"You feel pretty good…"*
**Animation:** Scale-in on the metric, flash on the check.
**Voiceover:** "Your database has a p99 of twenty milliseconds. Only one in a hundred requests is slow."

---

### Scene 2 — The Complaint (4s–9s)
**Visual:** A "Backend Engineer" avatar (pixel/emoji style) slides in from the right with a speech bubble: *"The service is slow!"* The earlier green check morphs into a red question mark.
**Animation:** Avatar slide-in, speech bubble pop, `?` shake.
**Voiceover:** "But your backend engineer is yelling that the service feels sluggish. How?"

---

### Scene 3 — The Fan-out (9s–18s)
**Visual:** A `User` icon on the left fires a single request arrow into a `Backend` box in the middle. From the backend, **10 query arrows** fan out to a `Database` cluster on the right. Arrows are numbered Q1…Q10.
**Animation:** Single request arrow draws first; then 10 fan-out arrows stagger-draw (50ms apart). Each lands on a DB node and shows a tiny clock.
**Voiceover:** "Every user search fans out into ten database queries. And the response can't go back until all ten are done."

---

### Scene 4 — The Slowest Wins (18s–24s)
**Visual:** All 10 query bars start filling left-to-right (horizontal progress bars stacked). 9 finish quickly in green. **Q7** keeps growing, slow, in red. A label appears next to it: *"p99 hit"*. The user-response gate stays closed behind the slowest bar.
**Animation:** 9 bars fill in ~0.6s; bar #7 slowly fills in ~2s, color shifts to red. A `wait…` ghost label sits over the user.
**Voiceover:** "One slow query drags the entire user request with it."

---

### Scene 5 — The Math (24s–34s)
**Visual:** Grid of 100 small dots — **10 rows × 10 columns**. Each row = one user, each dot = one of their 10 queries.
- 5a: All 100 dots fade in green.
- 5b: One dot in the entire grid flips red (1 in 100, the p99 tail).
- 5c: A red glow outlines the **entire row** that contains the red dot. Label: *"this user is slow"*.
- 5d: More red dots flip on, one at a time, on different rows → those whole rows light up red. End state: ~9–10 rows highlighted.

A live counter ticks at the bottom:
```
slow queries: 1%   →   slow users: ~10%
```
**Animation:** Dots fade in; red dots toggle one at a time (200ms apart); rows glow red on each toggle; counter increments live.
**Voiceover:** "Ten queries per user. So a hundred queries across ten users. One in a hundred lands in the slow tail. But because each user waits for all ten, one in ten users sees a slow response."

---

### Scene 6 — The Amplification Reveal (34s–42s)
**Visual:** Two big stat cards slide up side-by-side:
```
   1%               →               ~10%
slow queries                    slow users
```
Between them, a yellow lightning bolt `⚡` and the title **"Tail-Latency Amplification"** drops in with a slight bounce.
**Animation:** Cards slide up; arrow draws; title scales + glows.
**Voiceover:** "Your one-percent problem just became a ten-percent problem. This is tail-latency amplification."

---

### Scene 7 — Scaling It Up (42s–48s)
**Visual:** Compact chart. X-axis = "queries per request (fan-out)", Y-axis = "% of slow users". A curve draws from 1 → 99% as fan-out grows from 1 → ~500. Markers pop on:
- `fan-out=1 → 1%`
- `fan-out=10 → ~10%`
- `fan-out=100 → ~63%`

**Animation:** Axes draw, curve sweeps in via stroke-dash, markers pulse.
**Voiceover:** "The more you fan out, the worse it gets."

---

### Scene 8 — Takeaway (48s–55s)
**Visual:** Clean closing card:
```
Don't watch the average.
Watch the tail.
```
Smaller mono text below:
```
p99, p999 — that's what users feel.
```
Channel handle / subscribe CTA fades in at the bottom.
**Animation:** Lines type-in; CTA fades.
**Voiceover:** "So when you design a service — don't just stare at the average. Watch the tail. Because at scale, the tail is what your users actually feel."

---

## Scene Timing Summary

| #  | Scene                   | Start | End  | Duration |
|----|-------------------------|-------|------|----------|
| 1  | Hook (p99 = 20ms)       | 0s    | 4s   | 4s       |
| 2  | The Complaint           | 4s    | 9s   | 5s       |
| 3  | The Fan-out             | 9s    | 18s  | 9s       |
| 4  | The Slowest Wins        | 18s   | 24s  | 6s       |
| 5  | The Math (10×10 grid)   | 24s   | 34s  | 10s      |
| 6  | Amplification Reveal    | 34s   | 42s  | 8s       |
| 7  | Scaling It Up (chart)   | 42s   | 48s  | 6s       |
| 8  | Takeaway                | 48s   | 55s  | 7s       |

---

## Math Notes (for on-screen accuracy)

- Per-query slow probability: `p = 0.01` (p99 ⇒ 1% are slow)
- Per-user slow probability with fan-out `n`: `1 − (1 − p)^n`
  - `n = 1`   → 1.00%
  - `n = 10`  → **9.56%** (rounded to "~10%" on screen)
  - `n = 100` → 63.4%
  - `n = 500` → 99.3%
- The headline "one in ten users" is the simplification of 9.56%.
- Caveat (don't say on-screen, but worth knowing): this assumes independence between queries; in practice correlated slowness can make it even worse.

---

## Color & Type

- Background: `#0d1117`
- Fast / good: `#3fb950` (green)
- Slow / tail: `#f85149` (red)
- Amplification highlight: `#ffd700` (yellow)
- Neutral accent: `#58a6ff` (blue) for arrows, axes
- Sans labels: Inter / system sans
- Numbers, code-style cards: JetBrains Mono / `ui-monospace`
- Mobile-safe sizes: ≥ 36px body, ≥ 72px headline cards

## Animation Notes

- CSS keyframes for entrances (fade, slide, scale).
- `stroke-dasharray` + `stroke-dashoffset` for arrows and the curve reveal.
- A small JS timeline (`setTimeout` chain or `requestAnimationFrame`) drives scene transitions, keyed to the timing table above.
- Canvas: fixed 1080×1920; screen-record the page and overlay the VO in editing.

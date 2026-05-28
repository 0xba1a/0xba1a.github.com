# Average vs Percentile — YouTube Short (60s)

**Format:** 9:16 (1080×1920), HTML/CSS/JS with D3.js animations, click-through
**Topic:** Why averages lie for performance metrics; use percentiles instead

---

## Voiceover Script (~150 words, ~60s at natural pace)

> "What's the average response time?" — If an interviewer asks this, it's a trap.
>
> Performance metrics should never be measured as an average.
>
> Say your API serves 6,000 requests per minute. Most finish in a few milliseconds — but some take over 10 seconds. Take the average and those outliers drag it way to the right. The number looks fine, but it hides the pain.
>
> Instead, sort all response times from fastest to slowest and pick the middle one. That's the median — the 50th percentile. Half your users were faster than this.
>
> Go further: the 99th percentile — or p99 — tells you 99% of requests were faster than this value. Only the worst 1% were slower.
>
> Now you can set real targets. An SLO like "p99 under 200ms" means 99% of requests complete in under 200 milliseconds.
>
> So next time — skip the average. Use percentiles.

---

## Scene Plan (9 click-through steps)

### Scene 1 — Hook (0s–5s)
**Visual:** Dark background. Text punches in: `"Average Response Time?"` with a red strikethrough animating across "Average". Subtitle fades in: `"It's a trap."`
**Voiceover:** "What's the average response time?" — If an interviewer asks this, it's a trap.
**D3 notes:** Text animation only. CSS transitions for punch-in and strikethrough.

---

### Scene 2 — The Data (5s–12s)
**Visual:** A D3 bar chart appears — ~30 bars representing individual request response times. Most bars are short (20–80ms range), but 3–4 bars are very tall (2s–10s outliers). X-axis: "Requests", Y-axis: "Response Time (ms)". Bars grow upward with staggered animation.
**Voiceover:** Say your API serves 6,000 requests per minute. Most finish in a few milliseconds — but some take over 10 seconds.
**D3 notes:** `d3.scaleBand` for x, `d3.scaleLinear` for y. Bars transition from height 0 with staggered delay. Outlier bars colored orange/red, normal bars colored blue/green.

---

### Scene 3 — The Average Lie (12s–20s)
**Visual:** A horizontal dashed line sweeps in from the left at the average value. Label: `"Average: 420ms"`. The line sits noticeably above most bars but below the outliers. An arrow from the tallest outlier bar points to the average line with text `"Outliers drag it right"`. The average line pulses red briefly.
**Voiceover:** Take the average and those outliers drag it way to the right. The number looks fine, but it hides the pain.
**D3 notes:** Animate line with `transition().attr("x2", width)`. Arrow as SVG path + text annotation.

---

### Scene 4 — Sort the Bars (20s–27s)
**Visual:** The bars re-sort themselves from shortest to tallest with a smooth D3 transition (bars slide horizontally into sorted positions). The average line fades out. X-axis label changes to `"Requests (sorted)"`. The sorted arrangement makes the outlier tail clearly visible on the right.
**Voiceover:** Instead, sort all response times from fastest to slowest…
**D3 notes:** Re-compute `x` positions based on sorted order. Use `transition().duration(800)` on bar `x` attributes. Key function on original index for object constancy.

---

### Scene 5 — The Median / p50 (27s–34s)
**Visual:** A vertical line drops down at the middle bar (position 50%). Label: `"Median (p50): 45ms"`. The left half of bars dims slightly. The right half stays bright. A bracket below the left half labels `"50% faster"`.
**Voiceover:** …and pick the middle one. That's the median — the 50th percentile. Half your users were faster than this.
**D3 notes:** Vertical line at `x(sortedData[Math.floor(n/2)])`. Dim bars with `opacity: 0.4` transition on left half.

---

### Scene 6 — p99 (34s–42s)
**Visual:** A second vertical line drops at position 99%. Label: `"p99: 190ms"`. The region between p50 and p99 gets a subtle fill/highlight. The 1% outlier tail on the right is colored red. A small annotation: `"Worst 1%"` with an arrow pointing to the red tail.
**Voiceover:** Go further: the 99th percentile — or p99 — tells you 99% of requests were faster than this value. Only the worst 1% were slower.
**D3 notes:** Second vertical line. Use `rect` for shaded region between p50 and p99. Color rightmost 1% of bars red.

---

### Scene 7 — SLO (42s–50s)
**Visual:** A horizontal green line at 200ms with label `"SLO: p99 < 200ms"`. The p99 line (190ms) is below it — a green checkmark appears. The chart dims and a clean card overlays: `"99% of requests under 200ms ✓"`.
**Voiceover:** Now you can set real targets. An SLO like "p99 under 200ms" means 99% of requests complete in under 200 milliseconds.
**D3 notes:** Horizontal line + label. Conditional check icon (SVG path for checkmark). Overlay card with fade-in.

---

### Scene 8 — Comparison Summary (50s–55s)
**Visual:** Split view — left side shows the average line (red, with "misleading" label), right side shows percentile lines p50/p99 (green, with "actionable" label). Or a simple summary card:
```
❌  Average: 420ms  — hides outliers
✓  p50:      45ms  — typical user
✓  p99:     190ms  — worst case (almost)
```
**Voiceover:** So next time — skip the average.
**D3 notes:** Can be pure HTML/CSS card or a simplified D3 mini-chart comparison.

---

### Scene 9 — Takeaway (55s–60s)
**Visual:** Clean dark card with the key formula:
```
Don't use Average.
Use p50, p99, p99.9

SLO = "p99 < 200ms"
```
Follow/subscribe CTA with channel branding.
**Voiceover:** Use percentiles.
**D3 notes:** Text-only. CSS fade-in transitions.

---

## Color Scheme

| Element         | Color     | Hex       |
|-----------------|-----------|-----------|
| Background      | Dark      | `#0d1117` |
| Normal bars     | Blue      | `#58a6ff` |
| Outlier bars    | Orange    | `#f0883e` |
| Average line    | Red       | `#f85149` |
| p50 line        | Green     | `#3fb950` |
| p99 line        | Purple    | `#d2a8ff` |
| SLO line        | Green     | `#3fb950` |
| Dim text/bars   | Gray      | `#8b949e` |
| Bright text     | White     | `#e6edf3` |

## Data Generation

Use a realistic distribution: ~26 bars with response times from a log-normal distribution (bulk 20–80ms), plus 4 outlier bars (2000ms, 4500ms, 7000ms, 10000ms). Precompute values and hardcode in JS for reproducibility.

## Original Raw Script (preserved)

> When the interviewer asks "What will be the average response time?", remember, It's a trap.
>
> Because the response time or any such performance metric shouldn't be measured in average.
>
> Consider you are designing an API service that is serving 100 hits per second. So, you would have served 6000 requests in a minute. Some of them could have taken few milliseconds and some would have taken even more than 10 seconds. If you take an average, it will be affected by the outlier. If the slowest responses with very high response time will right shift the average.
>
> So, you should actually arrange the requests in an ascending order and take the middle one. It is called the median or the 50th percentile. It means half of the requests were served within that time. Similarly you can take 90th percentile and 99th percentile or 99.9th percentile.
>
> Based on the importance of the service and the tolerance of your clients, you can fix an appropriate Service Level Objective (SLO) for your service - something like it will serve 99% of the requests in less than 200 ms. So, it means your SLO is p99 is 200ms.
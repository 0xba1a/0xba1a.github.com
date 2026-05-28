# Short: Describing Performance — The Right Way

## Raw script (handwritten, preserved in plan.original.md)
How will you measure the performance of service you design? There are two important metrics. Response Time and Throughput. They affect one another. Push throughput up → response time spikes due to queuing. Push response time down → you need to scale, so throughput drops. Find the optimal middle ground and describe performance like: **1 million hits per second with p99 at 20 ms**.

## Polished VO (~55s, ~145 words)
1. **Hook (0–4s)** — "How do you describe a system's performance?"
2. **Two metrics (4–10s)** — "Two numbers matter. Response time — how fast each request finishes. Throughput — how many requests per second you handle."
3. **Tradeoff intro (10–16s)** — "Here's the catch. They fight each other."
4. **Push throughput up (16–24s)** — "Push throughput too high — queues build up and response time explodes."
5. **Push throughput down (24–30s)** — "Keep response time tiny — throughput collapses."
6. **Sweet spot (30–38s)** — "The trick is finding the knee. The sweet spot before things blow up."
7. **The statement (38–50s)** — "So don't say 'it's fast'. Say it like an engineer: one million requests per second at p99 twenty milliseconds."
8. **Takeaway (50–55s)** — "Talk in numbers. Get taken seriously."

## Scene plan (8 steps)
| # | Scene          | Visual                                                      | Animation                                                        |
|---|----------------|-------------------------------------------------------------|------------------------------------------------------------------|
| 1 | Hook           | Big title "Describe Performance?" + subtitle "Use numbers." | Scale-in + subtitle fade                                         |
| 2 | Two metrics    | Two stacked cards: Response Time / Throughput               | Cards slide in                                                   |
| 3 | Empty chart    | X=Throughput (req/s), Y=Response Time (ms)                  | Axes draw in, title fades                                        |
| 4 | Tradeoff curve | Hockey-stick curve appears                                  | Path stroke-dasharray reveal (1200ms)                            |
| 5 | Push right     | Marker moves to high-throughput zone, RT spikes red         | Dot transitions along path; danger label fades in                |
| 6 | Push left      | Marker moves to low-throughput zone                         | Dot back to left; "wasted capacity" label                        |
| 7 | Sweet spot     | Pulsing green dot at knee + crosshair lines to axes         | Dot scale pulse + dashed lines drop                              |
| 8 | The statement  | Big code-style card: `1M req/s @ p99 = 20ms`                | Card slides up, mono font, CTA fades in                          |

## Color use (GitHub Dark)
- Curve: `#58a6ff` (blue)
- Danger / high-RT marker: `#f85149` (red)
- Wasted / low-throughput marker: `#f0883e` (orange)
- Sweet spot: `#3fb950` (green)
- Statement card border / accent: `#d2a8ff` (purple)
- Annotation text: `#ffd700` (yellow)

## Curve model
Queuing approximation: `RT(λ) = RT₀ / (1 - λ/μ)` with μ = 1.2M req/s capacity, RT₀ = 4ms. Tuned so the knee at 1M req/s yields ~20ms — matches the punchline.

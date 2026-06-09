# Layout & Edge Margin — YouTube Shorts (1080×1920)

**Use the FULL frame.** We do **not** reserve space for the YouTube action rail
(like/comment/share), the title, the handle, or the description anymore. Those UI
chrome elements fade/scroll away, and reserving big buffers for them was making
the visuals small with awkward empty gutters on the right and bottom.

The only thing we keep is a **small uniform edge margin** (~40 px) so nothing
critical sits on the extreme edge, where some phones crop slightly.

These values are CSS variables in `template/style.css` and visualized by the
guide (`build_short.py --safe-preview` → `safezone_preview.png`).

## The one rule: uniform edge margin

| Region | Size | Why |
|---|---|---|
| **Edge margin** | ~40 px on all four sides | Device crop differs slightly between phones |

So the usable content box on a 1080×1920 canvas is essentially the whole frame:

- **x:** 40 px → 1040 px  (width ≈ 1000 px)
- **y:** 40 px → 1880 px  (height ≈ 1840 px)

In the template this is the `.safe-content` container. Fill it. Backgrounds may
bleed to the very edge.

## Practical rules

1. **Go BIG.** A Short is watched on a phone at arm's length. Make icons, images,
   tables, and diagrams large — fill the width. If it looks "designed for
   desktop," it's too small. Use the `--fs-*` scale (hero 150 / title 100 /
   subtitle 62 / body 50).
2. **Use the whole width and height.** No right gutter, no bottom gutter.
3. **Arrows for flow.** Show data/control movement with explicit arrowheads, not
   just lines or proximity.
4. **Direction is free.** Flows can run **left→right** as well as top→bottom —
   pick whatever reads most naturally for the step. Horizontal layouts often use
   the wide frame better for two-stage relationships.
5. **Edge breathing room only.** Keep meaning out of the extreme ~40 px so a crop
   never clips something important.
6. **High contrast.** Light on dark, or a scrim behind text over busy art.

## Adjusting per short

Override `--safe-edge` at the top of a short's `style.css` if you want even more
bleed. Re-run `--safe-preview` to confirm before the final build.

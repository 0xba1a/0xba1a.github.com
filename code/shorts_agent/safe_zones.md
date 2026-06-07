# Safe Zones & Mobile Layout — YouTube Shorts (1080×1920)

A Short is watched on a phone, inside the YouTube app, whose UI overlays your
video. Different phones also crop the frame slightly differently. So treat the
1080×1920 canvas as having a smaller **content-safe area** in the middle.

These values are encoded as CSS variables in `template/style.css` and visualized
by the safe-zone guide (`build_short.py --safe-preview` → `safezone_preview.png`).

## Reserved regions (do NOT place critical text/visuals here)

| Region | Size | Why it's reserved |
|---|---|---|
| **Device crop buffer** | 54 px (~5%) on all 4 edges | Phones crop edges differently |
| **Top** | 150 px | Status bar / Shorts top controls |
| **Right action rail** | 180 px | Like, Dislike, Comment, Share, Remix, Sound disc |
| **Bottom** | 360 px | Channel handle, title, description, CTA, progress bar |
| **Left (lower)** | 60 px | Channel handle text (bottom-left) |

## Content-safe box (put everything important here)

With the above, the usable box on a 1080×1920 canvas is roughly:

- **x:** 60 px → 900 px  (width ≈ 840 px)
- **y:** 150 px → 1560 px (height ≈ 1410 px)

In the template this is the `.safe-content` container — anchor all key text and
focal visuals inside it. Backgrounds/gradients may fill the full stage, but they
must remain non-essential (safe to crop).

## Practical rules

1. **Center the message.** Vertically center the key line; the eye lands mid-screen.
2. **Big type only.** Use the `--fs-*` scale (hero 132 / title 88 / subtitle 56 /
   body 46). If text feels "designed for desktop," it's too small.
3. **Right side clear.** Don't run text/diagrams into the right 180 px — it sits
   under the like/share rail.
4. **Bottom clear.** The lower 360 px is covered by title/description; keep CTAs
   and key facts above it. A "follow" line is fine but place it just above the
   bottom zone, not inside it.
5. **Edge breathing room.** Even decorative elements shouldn't have meaning at the
   extreme edges (crop buffer).
6. **High contrast.** Light text on dark, or add a scrim behind text over busy art.

## Adjusting per short

If a specific layout needs it, override the variables at the top of that short's
`style.css` (e.g. a full-bleed hook with no bottom text can shrink `--safe-bottom`
temporarily). Re-run `--safe-preview` to confirm before the final build.

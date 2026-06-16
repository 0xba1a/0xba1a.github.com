# Shorts Agent

A self-contained agent + toolkit that turns a topic into a finished, narrated
vertical **YouTube Short** (1080×1920) — entirely locally.

- **Voice:** Chatterbox neural TTS (Resemble AI, offline), with natural expressive
  voice and excellent text normalization.
- **Music:** real public-domain (CC0) lowkey-upbeat tracks from the local `music/`
  library, rotated per video (looped/trimmed/faded and mixed under the voice).
  A seeded procedural synth is still available as an alternative.
- **Layout:** mobile-first template with full-frame design and big fonts.
- **Recording:** Playwright renders the HTML deck headlessly at 1080×1920;
  `ffmpeg` muxes voice with karaoke subtitles.

## Workflow (3 phases)
1. **Discuss** — you give a topic + overview; the agent asks a few questions and
   proposes title, slug, voice, and music mood.
2. **Plan** — the agent writes `shorts/<slug>/plan.md` (full voiceover script +
   storyboard + safe-zone plan) and **waits for your review**.
3. **Produce** — on approval, it scaffolds the deck, then builds `final.mp4`.

The agent persona lives in `shorts_producer.agent.md` (installed to
`.github/agents/` so VS Code can select it). Its persistent memory is `memory.md`.

## Layout
```
code/shorts_agent/
  shorts_producer.agent.md   # the agent definition (source of truth)
  memory.md                  # agent's self-updating memory
  safe_zones.md              # full-frame layout spec (edge margin only)
  build_short.py             # TTS + music + record + mux pipeline
  vo_tts.py                  # Chatterbox neural TTS wrapper
  vo_music.py                # seeded procedural ambient music (alternative source)
  music/                     # public-domain (CC0) background tracks + manifest/credits
  template/                  # base deck: index.html, style.css, script.js, plan.md
  shorts/<slug>/             # one folder per video: deck + narration.txt +
                             #   voiceover.json + _work/ + final.mp4
```

## Per-short files
- `narration.txt` — one voiceover line per row (row N ↔ scene N).
- `voiceover.json` — voice, speed, timing, and music settings. Example:
  ```json
  {
    "voice": null,
    "speed": 1.1,
    "intro": 0.5, "gap": 0.35, "outro": 1.6,
    "trim_silence": true,
    "exaggeration": 0.35,
    "cfg_weight": 0.5,
    "music": { "enabled": true, "source": "library", "track": null, "gain_db": -26.0 },
    "subtitles": true
  }
  ```
  - `voice`: optional path to a .wav file for voice cloning; `null` uses built-in narrator
  - `exaggeration`: 0.3-0.4 for calm/measured, 0.5 for animated (default 0.35)
  - `cfg_weight`: ~0.5 for steady pacing (default 0.5)
  - `music`: ON by default. `source: "library"` rotates the CC0 tracks in `music/`
    (one per video, chosen from the slug); `track` pins a specific file; `gain_db`
    sets the level under the voice (≈ −24…−28). Use `source: "procedural"` + `mood`
    for the synth, or `"enabled": false` to mute. See `music/CREDITS.md`.

## Build commands
```bash
# one-time env already created at repo root: .venv-tts (Python 3.12 + Chatterbox + Playwright)
source .venv-tts/bin/activate

# QA the layout against edge margins (writes safezone_preview.png)
python code/shorts_agent/build_short.py code/shorts_agent/shorts/<slug> --safe-preview

# full build -> shorts/<slug>/final.mp4
python code/shorts_agent/build_short.py code/shorts_agent/shorts/<slug>
```

## Setup (if recreating the environment)
```bash
brew install python@3.12 ffmpeg
python3.12 -m venv .venv-tts && source .venv-tts/bin/activate
pip install chatterbox-tts soundfile numpy playwright torchaudio
pip install "setuptools<81"  # CRITICAL: perth watermarker needs pkg_resources
python -m playwright install chromium
```

**Important:** `setuptools<81` is required because Chatterbox's watermarker (`perth`)
imports `pkg_resources`, which was removed in setuptools 81+. Without the downgrade,
you'll get `TypeError: 'NoneType' object is not callable` at model load.

## Voice configuration
Chatterbox uses a built-in narrator voice by default. For voice cloning, provide a
path to a reference .wav file in `voiceover.json`.

**See [`VOICE_GUIDE.md`](VOICE_GUIDE.md)** for detailed voice parameter reference,
presets, and tuning tips.

The old Kokoro voice names (`am_*`, `af_*`, etc.) are no longer used.

## Documentation

- **[VOICE_GUIDE.md](VOICE_GUIDE.md)** - Voice parameter reference and presets
- **[CHATTERBOX_MIGRATION.md](CHATTERBOX_MIGRATION.md)** - Migration guide and setup details
- **[memory.md](memory.md)** - Agent's persistent memory (user preferences, lessons learned)
- **[safe_zones.md](safe_zones.md)** - Full-frame layout specification

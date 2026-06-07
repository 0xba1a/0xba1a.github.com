# Shorts Agent

A self-contained agent + toolkit that turns a topic into a finished, narrated
vertical **YouTube Short** (1080×1920) — entirely locally.

- **Voice:** Kokoro neural TTS (offline), a different voice per video.
- **Music:** procedural ambient generator, a unique low-key track per video.
- **Layout:** mobile-first template with YouTube **safe zones** and big fonts.
- **Recording:** Playwright renders the HTML deck headlessly at 1080×1920;
  `ffmpeg` muxes voice + music.

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
  safe_zones.md              # mobile / YouTube safe-zone spec
  build_short.py             # TTS + music + record + mux pipeline
  vo_tts.py                  # Kokoro neural TTS wrapper
  vo_music.py                # seeded procedural ambient music
  models/                    # Kokoro model + voices (offline)
  template/                  # base deck: index.html, style.css, script.js, plan.md
  shorts/<slug>/             # one folder per video: deck + narration.txt +
                             #   voiceover.json + _work/ + final.mp4
```

## Per-short files
- `narration.txt` — one voiceover line per row (row N ↔ scene N).
- `voiceover.json` — voice, speed, timing, and music settings. Example:
  ```json
  {
    "voice": "am_fenrir",
    "speed": 1.0,
    "intro": 0.5, "gap": 0.8, "outro": 1.6,
    "music": { "enabled": true, "mood": "dark", "seed": "<slug>", "gain_db": -24.0 }
  }
  ```

## Build commands
```bash
# one-time env already created at repo root: .venv-tts (Python 3.12 + Kokoro + Playwright)
source .venv-tts/bin/activate

# QA the layout against safe zones (writes safezone_preview.png)
python code/shorts_agent/build_short.py code/shorts_agent/shorts/<slug> --safe-preview

# full build -> shorts/<slug>/final.mp4
python code/shorts_agent/build_short.py code/shorts_agent/shorts/<slug>
```

## Setup (if recreating the environment)
```bash
brew install python@3.12 espeak-ng ffmpeg
python3.12 -m venv .venv-tts && source .venv-tts/bin/activate
pip install kokoro-onnx soundfile numpy playwright
python -m playwright install chromium
# models already in code/shorts_agent/models/ (kokoro-v1.0.onnx, voices-v1.0.bin)
```

## Voices & moods
See `memory.md` for the curated voice guide and music moods. List all voices:
`python -c "from kokoro_onnx import Kokoro; print(Kokoro('code/shorts_agent/models/kokoro-v1.0.onnx','code/shorts_agent/models/voices-v1.0.bin').get_voices())"`

# Chatterbox Voice Quick Reference

## Default Configuration (voiceover.json)

```json
{
  "voice": null,
  "speed": 1.1,
  "exaggeration": 0.35,
  "cfg_weight": 0.5,
  "intro": 0.5,
  "gap": 0.8,
  "outro": 1.6,
  "subtitles": true,
  "music": {
    "enabled": false
  }
}
```

## Voice Parameters

### `voice` (voice selection/cloning)
- `null` or omit → **Built-in narrator** (high-quality default, recommended)
- `"/path/to/reference.wav"` → Clone voice from reference audio

**Example voice cloning:**
```json
{
  "voice": "/Users/bala/audio_samples/narrator_sample.wav"
}
```

### `exaggeration` (emotional expressiveness)
Controls how animated/expressive the voice sounds.

| Value | Style | Best For |
|-------|-------|----------|
| `0.3` | Very calm, measured | Dense technical content, formal explanations |
| `0.35` | **Default** - Calm but authoritative | Architecture explainers, system design |
| `0.4` | Moderate engagement | Storytelling, tutorials |
| `0.5` | Animated, energetic | Marketing hooks, exciting reveals |

**Example - Excited tone:**
```json
{
  "exaggeration": 0.5
}
```

### `cfg_weight` (pacing control)
Controls speech pacing and rhythm consistency.

| Value | Pacing | Notes |
|-------|--------|-------|
| `0.3` | Faster, varied | More natural rhythm variation |
| `0.5` | **Default** - Steady | Consistent, predictable timing |
| `0.7` | Slower, deliberate | Emphasis on each word |

**Example - Faster pacing:**
```json
{
  "cfg_weight": 0.3
}
```

### `speed` (post-processing)
Applied via ffmpeg after synthesis (does not affect voice character).

| Value | Speed | Notes |
|-------|-------|-------|
| `1.0` | Normal | Natural speaking pace |
| `1.1` | **Default** - Slightly faster | Good balance for 60s shorts |
| `1.2` | Noticeably faster | Dense content, more information |
| `1.3`+ | Very fast | May sound rushed |

**Example - Normal speed:**
```json
{
  "speed": 1.0
}
```

## Common Presets

### Calm Technical Explainer (Default)
```json
{
  "exaggeration": 0.35,
  "cfg_weight": 0.5,
  "speed": 1.1
}
```
Good for: System design, architecture, data engineering

### Energetic Marketing/Hook
```json
{
  "exaggeration": 0.5,
  "cfg_weight": 0.4,
  "speed": 1.2
}
```
Good for: Product launches, exciting announcements, "gotcha" moments

### Measured Documentary
```json
{
  "exaggeration": 0.3,
  "cfg_weight": 0.6,
  "speed": 1.0
}
```
Good for: Historical context, factual deep-dives, formal presentations

### Engaging Storyteller
```json
{
  "exaggeration": 0.4,
  "cfg_weight": 0.5,
  "speed": 1.1
}
```
Good for: Tutorial narratives, case studies, journey stories

## Tips

1. **Start with defaults** - The default settings work well for most content
2. **Tune exaggeration first** - Biggest impact on tone/feel
3. **Adjust speed last** - After you've heard the natural pacing
4. **Test incrementally** - Change one parameter at a time
5. **Voice cloning is optional** - The built-in narrator is excellent

## Testing

Quick test of settings:
```bash
source .venv-tts/bin/activate
python -c "
import sys
sys.path.insert(0, 'code/shorts_agent')
import vo_tts

vo_tts.synth(
    'This is a test of the voice settings.',
    '/tmp/test.wav',
    speed=1.1,
    exaggeration=0.35,
    cfg_weight=0.5
)
print('Test audio: /tmp/test.wav')
"
```

Listen to `/tmp/test.wav` and adjust parameters as needed.

## Migration from Kokoro

Old voice names like `am_michael`, `af_bella`, etc. are **no longer used**.

**Before (Kokoro):**
```json
{
  "voice": "am_michael",
  "speed": 1.2
}
```

**After (Chatterbox):**
```json
{
  "voice": null,
  "speed": 1.1,
  "exaggeration": 0.35,
  "cfg_weight": 0.5
}
```

The built-in Chatterbox narrator is higher quality than any Kokoro voice.

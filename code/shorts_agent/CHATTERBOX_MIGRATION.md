# Chatterbox TTS Migration Summary

**Date:** June 14, 2026  
**Status:** ✅ Complete and Verified

## What Changed

The Shorts Producer system has been **fully updated** from Kokoro TTS to **Chatterbox TTS** (Resemble AI) for significantly improved voice quality.

## Key Improvements

### Voice Quality
- **Natural & Expressive:** Chatterbox uses an LLM-backbone architecture that sounds much more human and engaging than Kokoro
- **Better Text Normalization:** Properly handles numbers, acronyms, and technical terms without garbled pronunciation
- **Tunable Tone:** Control expressiveness via `exaggeration` (0.3-0.5) and pacing via `cfg_weight` (~0.5)

### Voice Configuration
- **Built-in Narrator:** High-quality default voice (no voice selection needed)
- **Voice Cloning:** Optional - provide a reference .wav file to clone any voice
- **No Voice Names:** Old Kokoro voice names (`am_michael`, `af_bella`, etc.) are deprecated

## Updated Files

### Documentation
- ✅ `README.md` - Updated setup instructions, removed Kokoro references
- ✅ `shorts_producer.agent.md` - Updated agent instructions
- ✅ `template/plan.md` - Updated voice configuration template
- ✅ `CHATTERBOX_MIGRATION.md` - This migration guide (new)

### Code
- ✅ `vo_tts.py` - Already migrated to Chatterbox (no changes needed)
- ✅ `build_short.py` - Updated defaults and parameter passing:
  - Default voice: `None` (built-in narrator)
  - Default speed: `1.1` (was `1.2` for Kokoro)
  - Default exaggeration: `0.35` (calm/measured)
  - Default cfg_weight: `0.5` (steady pacing)
  - Passes `exaggeration` and `cfg_weight` to TTS engine

### Environment
- ✅ Python 3.12 virtual environment (`.venv-tts`)
- ✅ Installed packages:
  - `chatterbox-tts` - Main TTS engine
  - `torchaudio` - Audio processing (PyTorch)
  - `setuptools<81` - Required for perth watermarker
  - All dependencies (~110+ packages)
- ✅ Model weights downloaded (~3.2GB, cached in HuggingFace)

## Configuration Format

### Old (Kokoro)
```json
{
  "voice": "am_michael",
  "speed": 1.2
}
```

### New (Chatterbox)
```json
{
  "voice": null,
  "speed": 1.1,
  "exaggeration": 0.35,
  "cfg_weight": 0.5
}
```

**Optional voice cloning:**
```json
{
  "voice": "/path/to/reference.wav",
  "speed": 1.1
}
```

## Setup Instructions

### Fresh Environment Setup
```bash
brew install python@3.12 ffmpeg
python3.12 -m venv .venv-tts && source .venv-tts/bin/activate
pip install chatterbox-tts soundfile numpy playwright torchaudio
pip install "setuptools<81"  # CRITICAL - required for perth watermarker
python -m playwright install chromium
```

### Activate Existing Environment
```bash
source .venv-tts/bin/activate
```

## Voice Tuning Parameters

### `exaggeration` (emotional expressiveness)
- `0.3` - Very calm, measured delivery (technical/factual content)
- `0.35` - **Default** - Calm but authoritative (architect/explainer)
- `0.4` - Moderate engagement (storytelling)
- `0.5` - Animated, energetic (marketing/hooks)

### `cfg_weight` (pacing control)
- `0.3` - Faster, more varied pacing
- `0.5` - **Default** - Steady, consistent pacing
- `0.7` - Slower, more deliberate

### `speed` (post-processing via ffmpeg)
- `1.0` - Normal speed
- `1.1` - **Default** - Slightly faster (was 1.2 for Kokoro)
- `1.2` - Noticeably faster (good for dense content)

## Verification

Test synthesis:
```bash
source .venv-tts/bin/activate
python code/shorts_agent/build_short.py code/shorts_agent/shorts/<slug>
```

The first run will download model weights (~3.2GB) which are then cached locally.

## Migration Notes

### Breaking Changes
- **Voice names removed:** Old `am_*` / `af_*` voice parameters no longer work
- **Speed default changed:** 1.2 → 1.1 (Chatterbox is naturally more expressive)
- **New parameters:** `exaggeration` and `cfg_weight` for tone control

### Compatibility
- **Narration files:** No changes needed - same `narration.txt` format
- **Build pipeline:** Fully compatible - same `build_short.py` command
- **Output format:** Same 1080×1920 MP4 with karaoke subtitles

### Known Issues
- ⚠️ Minor dependency conflict: `kokoro-onnx` wants `numpy>=2.0`, Chatterbox needs `numpy<2.0`
  - **Impact:** None - we're migrating away from Kokoro
  - **Action:** Can uninstall `kokoro-onnx` if desired: `pip uninstall kokoro-onnx`
- ⚠️ `pkg_resources` deprecation warning from perth watermarker
  - **Impact:** Cosmetic only - warning can be ignored
  - **Action:** None - works correctly with setuptools<81

## Performance

### Synthesis Speed (CPU on M1 Mac)
- ~25-30 seconds per 5-second audio clip
- Full 75-second short: ~3-4 minutes total build time
- First run: Add ~2 minutes for model download

### Model Storage
- Weights: ~3.2GB (HuggingFace cache: `~/.cache/huggingface/`)
- Per-video output: same as before (~5-15MB depending on length)

## Rollback (If Needed)

If you need to revert to Kokoro:

```bash
source .venv-tts/bin/activate
pip uninstall chatterbox-tts torchaudio
pip install kokoro-onnx numpy>=2.0
# Restore old vo_tts.py from git history
# Revert build_short.py DEFAULTS
```

However, this is **not recommended** - Chatterbox produces significantly better quality.

## Next Steps

1. ✅ Environment ready
2. ✅ Documentation updated  
3. ✅ Test synthesis verified
4. Ready to create new shorts with improved voice quality!

---

**Questions?** Check `memory.md` for user preferences and tone settings, or `vo_tts.py` for implementation details.

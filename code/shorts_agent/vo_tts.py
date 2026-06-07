"""Local neural TTS via Kokoro (kokoro-onnx). Runs fully offline.

Voices (a few good ones):
  am_michael, am_adam   – US male, clear narration
  af_heart, af_bella    – US female, warm
  bm_george, bm_lewis   – UK male
Full list: https://github.com/thewh1teagle/kokoro-onnx
"""
import os
import soundfile as sf
from kokoro_onnx import Kokoro

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_DIR, "models", "kokoro-v1.0.onnx")
_VOICES = os.path.join(_DIR, "models", "voices-v1.0.bin")

_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = Kokoro(_MODEL, _VOICES)
    return _engine


def synth(text, out_wav, voice="am_michael", speed=1.0, lang="en-us"):
    """Synthesize `text` to `out_wav` (24 kHz mono). Returns duration in seconds."""
    samples, sr = engine().create(text, voice=voice, speed=speed, lang=lang)
    sf.write(out_wav, samples, sr)
    return len(samples) / sr

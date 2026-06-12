"""Local neural TTS via **Chatterbox** (Resemble AI). Runs on CPU.

Replaces the older Kokoro (kokoro-onnx) engine: Chatterbox is an LLM-backbone TTS
that sounds noticeably more natural/expressive and normalizes text (numbers,
acronyms) far better — Kokoro's espeak-ng phonemizer mispronounced terms.

API kept compatible with build_short.py:  synth(text, out_wav, voice, speed, lang)
- `voice`   : optional path to a short reference .wav to clone a voice
              (audio_prompt_path). If None/missing, the built-in narrator is used.
- `speed`   : post-applied via ffmpeg atempo (Chatterbox has no speed knob).
- `lang`    : ignored (model is English); kept for signature compatibility.

Tone is tuned calm/authoritative via `exaggeration` + `cfg_weight` (see DEFAULTS).
"""
import os
import shutil
import subprocess
import wave
import contextlib

import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

_DIR = os.path.dirname(os.path.abspath(__file__))
_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

# Calm, authoritative architect tone:
#   exaggeration ~0.3-0.4 = measured delivery (0.5 default is more animated)
#   cfg_weight   ~0.5      = steady pacing
DEFAULTS = {"exaggeration": 0.35, "cfg_weight": 0.5, "temperature": 0.7}

_engine = None


def engine():
    global _engine
    if _engine is None:
        # CPU on this host (no CUDA); weights are fetched/cached from HuggingFace.
        _engine = ChatterboxTTS.from_pretrained(device="cpu")
    return _engine


def _wav_duration(path):
    with contextlib.closing(wave.open(path, "r")) as w:
        return w.getnframes() / float(w.getframerate())


def _atempo_chain(speed):
    """ffmpeg atempo supports 0.5–2.0 per filter; chain for values outside that."""
    s = float(speed)
    parts = []
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    parts.append(f"atempo={s:.4f}")
    return ",".join(parts)


def synth(text, out_wav, voice=None, speed=1.0, lang="en-us",
          exaggeration=None, cfg_weight=None):
    """Synthesize `text` to `out_wav`. Returns duration in seconds."""
    m = engine()
    kw = {
        "exaggeration": DEFAULTS["exaggeration"] if exaggeration is None else exaggeration,
        "cfg_weight": DEFAULTS["cfg_weight"] if cfg_weight is None else cfg_weight,
        "temperature": DEFAULTS["temperature"],
    }
    # `voice` may be a path to a reference clip for voice cloning.
    if voice and os.path.isfile(voice):
        kw["audio_prompt_path"] = voice

    wav = m.generate(text, **kw)          # torch tensor, shape (1, N)
    sr = m.sr

    if speed and abs(float(speed) - 1.0) > 0.01:
        tmp = out_wav + ".raw.wav"
        ta.save(tmp, wav, sr)
        subprocess.run(
            [_FFMPEG, "-y", "-loglevel", "error", "-i", tmp,
             "-filter:a", _atempo_chain(speed), out_wav],
            check=True,
        )
        os.remove(tmp)
    else:
        ta.save(out_wav, wav, sr)

    return _wav_duration(out_wav)

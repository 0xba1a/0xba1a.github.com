"""Procedural ambient background music — deterministic from a seed string.

Same (seed, mood) always produces the same track, but different videos (different
seeds) get distinct music. No samples, no copyright: everything is synthesized.

Usage:
    import vo_music
    vo_music.generate("out.wav", duration=58.0, seed="my-video", mood="dark")
"""
import hashlib
import numpy as np
import soundfile as sf

SR = 44100

# semitone offsets used to pick a "color" tone per scale
SCALES = {
    "minor":            [0, 3, 7, 10],
    "major":            [0, 4, 7, 11],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "lydian":           [0, 4, 7, 11, 14],
}

# mood → tonal centre (MIDI), scale, lowpass cutoff (Hz), air-noise level
MOODS = {
    "dark":   {"root_midi": 33, "scale": "minor",            "cutoff": 1100, "noise": 0.015},
    "tense":  {"root_midi": 34, "scale": "minor_pentatonic", "cutoff": 1400, "noise": 0.020},
    "calm":   {"root_midi": 36, "scale": "major",            "cutoff": 1700, "noise": 0.012},
    "bright": {"root_midi": 40, "scale": "lydian",           "cutoff": 2400, "noise": 0.010},
    "mystery":{"root_midi": 31, "scale": "minor",            "cutoff": 1300, "noise": 0.018},
}


def _rng(seed: str) -> np.random.Generator:
    h = hashlib.sha256(seed.encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _midi_to_freq(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _voice(freq, n, rng, sr=SR):
    """One detuned, slowly-tremoloed pad voice (warm, saw-ish)."""
    t = np.arange(n) / sr
    out = np.zeros(n)
    for detune, w in [(0.0, 1.0), (0.18, 0.5), (-0.15, 0.5)]:
        f = freq * (1 + detune / 100.0)
        for harm, hw in [(1, 1.0), (2, 0.35), (3, 0.18)]:
            phase = rng.uniform(0, 2 * np.pi)
            out += w * hw * np.sin(2 * np.pi * f * harm * t + phase)
    lfo_rate = rng.uniform(0.03, 0.1)
    lfo = 0.85 + 0.15 * np.sin(2 * np.pi * lfo_rate * t + rng.uniform(0, 6.28))
    return out * lfo


def _fft_filter(x, sr=SR, lp=None, hp=None):
    """Gentle, fast frequency-domain low/high-pass."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    H = np.ones_like(f)
    if lp:
        H *= 1.0 / np.sqrt(1.0 + (f / lp) ** 4)
    if hp:
        H *= (f ** 2) / (f ** 2 + hp ** 2)
    return np.fft.irfft(X * H, n=len(x))


def _build_channel(root, tones, cfg, n, rng, sr=SR):
    mix = np.zeros(n)
    for st in tones:
        mix += _voice(_midi_to_freq(root + st), n, rng, sr)
    # warm sub an octave below the root
    mix += 0.6 * np.sin(2 * np.pi * _midi_to_freq(root - 12) * np.arange(n) / sr)
    mix = _fft_filter(mix, sr, lp=cfg["cutoff"])
    # subtle "air" texture
    noise = _fft_filter(rng.normal(0, 1, n), sr, hp=4000, lp=9000)
    mix += cfg["noise"] * noise
    return mix


def generate(out_wav, duration, seed="default", mood="dark",
             target_dbfs=-15.0, fade=2.0, sr=SR):
    cfg = MOODS.get(mood, MOODS["dark"])
    scale = SCALES[cfg["scale"]]
    n = int(duration * sr)

    rng = _rng(f"{seed}|{mood}")
    # seed-driven voicing for per-video variety
    root = cfg["root_midi"] + 12 * int(rng.integers(0, 2))
    tones = [0, 7, 12, scale[1 % len(scale)]]
    if rng.random() < 0.5:
        tones.append(19)  # add a high fifth shimmer

    left = _build_channel(root, tones, cfg, n, _rng(f"{seed}|{mood}|L"), sr)
    right = _build_channel(root, tones, cfg, n, _rng(f"{seed}|{mood}|R"), sr)
    stereo = np.stack([left, right], axis=1)

    # fade in/out
    fn = min(int(fade * sr), n // 2)
    if fn > 0:
        env = np.ones(n)
        env[:fn] = np.linspace(0, 1, fn)
        env[-fn:] = np.linspace(1, 0, fn)
        stereo *= env[:, None]

    # RMS normalize to target
    rms = np.sqrt(np.mean(stereo ** 2)) + 1e-9
    stereo *= (10 ** (target_dbfs / 20.0)) / rms
    stereo = np.clip(stereo, -1.0, 1.0)

    sf.write(out_wav, stereo.astype(np.float32), sr)
    return out_wav


if __name__ == "__main__":
    import sys
    generate(sys.argv[1] if len(sys.argv) > 1 else "music.wav",
             float(sys.argv[2]) if len(sys.argv) > 2 else 20.0,
             seed=sys.argv[3] if len(sys.argv) > 3 else "demo",
             mood=sys.argv[4] if len(sys.argv) > 4 else "dark")
    print("wrote", sys.argv[1] if len(sys.argv) > 1 else "music.wav")

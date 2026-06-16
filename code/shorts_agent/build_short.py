#!/usr/bin/env python3
"""
Build a voice-over YouTube Short from an HTML deck.

Per-short project folder (e.g. code/shorts_agent/shorts/<slug>/):
    index.html / style.css / script.js  – the click-through deck
    narration.txt   – one line per deck step (blank / #-comment lines ignored)
    voiceover.json  – optional config (voice, speed, music, timing)

Outputs (written into the same project folder):
    final.mp4       – the finished vertical short (1080x1920)
    _work/          – intermediate audio/video artifacts

Run from the repo root with the TTS venv active:
    source .venv-tts/bin/activate
    python code/shorts_agent/build_short.py code/shorts_agent/shorts/<slug>
"""
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vo_tts
import vo_music
import numpy as np
import soundfile as sf

W, H = 1080, 1920
SR = 44100
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
MUSIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "music")

# Strip leading/trailing silence from a TTS clip while keeping a small natural
# pad (so words aren't clipped). Trailing silence is removed by reversing the
# stream, trimming its (now leading) silence, then reversing back.
_TRIM_AF = (
    "silenceremove=start_periods=1:start_threshold=-45dB:"
    "start_silence=0.05:detection=peak,"
    "areverse,"
    "silenceremove=start_periods=1:start_threshold=-45dB:"
    "start_silence=0.10:detection=peak,"
    "areverse"
)

DEFAULTS = {
    "voice": None,  # None = Chatterbox built-in narrator; or path to .wav for cloning
    "speed": 1.1,
    "lang": "en-us",  # ignored by Chatterbox (English-only model); kept for compatibility
    "exaggeration": 0.35,  # 0.3-0.4 calm/measured, 0.5 animated
    "cfg_weight": 0.5,  # steady pacing
    "intro": 0.5,
    "gap": 0.35,
    "outro": 1.6,
    "subtitles": True,
    # Trim baked-in leading/trailing silence from each TTS clip so speech butts
    # right up to the (now small) inter-scene gap. Chatterbox pads clips with
    # silence that stacked on top of `gap`, making pauses between sentences feel
    # long. Set false to keep raw clips.
    "trim_silence": True,
    # Background music. Default = real public-domain (CC0) lowkey-upbeat tracks from
    # the music/ library, rotated per video (track chosen deterministically from the
    # slug). Set "source": "procedural" to use the old seeded synth instead, or pin a
    # specific file with "track": "<name>.mp3". Disable with "enabled": false.
    "music": {
        "enabled": True,
        "source": "library",   # "library" (file rotation) or "procedural" (synth)
        "track": None,          # pin a specific library file; None = auto-rotate
        "mood": "dark",         # only used when source == "procedural"
        "seed": None,           # rotation/synth seed; None = slug
        "gain_db": -26.0,       # music level under the voice
        "fade": 1.5,            # fade in/out seconds
    },
}

# ── Karaoke subtitle overlay (post-processed, burned in with ffmpeg) ───────────
# Subtitles are NOT injected into the HTML. After the silent video is recorded we
# build an ASS file with short caption chunks (≈1–2 lines visible at a time, like
# withsubtitles.com) timed to each TTS clip, then burn it in during the final
# encode. Within each chunk, words flip white→gold one-by-one (karaoke \k tags).
# Position: near the bottom edge, full width minus a small uniform margin (we no
# longer reserve the YouTube action-rail / title area — use the whole frame).
_SUB_FONT = "Arial"
_SUB_FONTSIZE = 58
_SUB_PRIMARY = "&H0000D7FF"    # gold #FFD700 once "sung"
_SUB_SECONDARY = "&H00FFFFFF"  # white before highlight
_SUB_OUTLINE = "&H00000000"    # black stroke
_SUB_MARGIN_L = 50             # uniform edge margin
_SUB_MARGIN_R = 50             # uniform edge margin (full width, no right rail)
_SUB_MARGIN_V = 180            # one line height above the previous bottom placement
_SUB_MAX_WORDS = 5
_SUB_MAX_CHARS = 26


def _ass_time(t):
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _chunk_words(words, max_words=_SUB_MAX_WORDS, max_chars=_SUB_MAX_CHARS):
    """Group words into short caption chunks (≈1 line each)."""
    chunks, cur, cur_len = [], [], 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and (len(cur) >= max_words or cur_len + add > max_chars):
            chunks.append(cur)
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        chunks.append(cur)
    return chunks


def _karaoke_text(words, chunk_cs):
    """Distribute chunk_cs centiseconds across words by length; emit \\k tags."""
    weights = [max(1, len(w)) for w in words]
    total_w = sum(weights)
    durs, acc = [], 0
    for i, wt in enumerate(weights):
        if i == len(weights) - 1:
            durs.append(max(1, chunk_cs - acc))
        else:
            d = max(1, int(round(chunk_cs * wt / total_w)))
            durs.append(d)
            acc += d
    return "".join(f"{{\\k{d}}}{w} " for d, w in zip(durs, words)).rstrip()


def generate_ass(clips, cfg, lines, work):
    """Build an ASS subtitle file timed to the voice track. Returns its path."""
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{_SUB_FONT},{_SUB_FONTSIZE},{_SUB_PRIMARY},"
        f"{_SUB_SECONDARY},{_SUB_OUTLINE},&H00000000,1,0,0,0,100,100,0,0,1,3,2,"
        f"2,{_SUB_MARGIN_L},{_SUB_MARGIN_R},{_SUB_MARGIN_V},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    events = []
    t = cfg["intro"]
    for i, (_wav, d) in enumerate(clips):
        scene_start = t
        t += d + cfg["gap"]
        if i >= len(lines):
            continue
        words = lines[i].split()
        if not words:
            continue
        chunks = _chunk_words(words)
        n_words = len(words)
        cs = scene_start
        for chunk in chunks:
            chunk_dur = d * len(chunk) / n_words
            ce = cs + chunk_dur
            text = _karaoke_text(chunk, int(round(chunk_dur * 100)))
            events.append(
                f"Dialogue: 0,{_ass_time(cs)},{_ass_time(ce)},Default,,0,0,0,,{text}"
            )
            cs = ce
    path = os.path.join(work, "subs.ass")
    with open(path, "w") as f:
        f.write(header + "\n".join(events) + "\n")
    return path


def run(cmd):
    print("·", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def wav_duration(path):
    with contextlib.closing(wave.open(path, "r")) as w:
        return w.getnframes() / float(w.getframerate())


def load_config(short_dir):
    cfg = json.loads(json.dumps(DEFAULTS))
    path = os.path.join(short_dir, "voiceover.json")
    if os.path.isfile(path):
        user = json.load(open(path))
        for k, v in user.items():
            if k == "music" and isinstance(v, dict):
                cfg["music"].update(v)
            else:
                cfg[k] = v
    return cfg


def read_lines(short_dir):
    path = os.path.join(short_dir, "narration.txt")
    out = []
    for raw in open(path):
        s = raw.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def synth(lines, cfg, work):
    """One wav per step at 44.1k stereo. Returns [(wav, duration)]."""
    clips = []
    for i, text in enumerate(lines, 1):
        raw = os.path.join(work, f"seg_{i:02d}_raw.wav")
        wav = os.path.join(work, f"seg_{i:02d}.wav")
        vo_tts.synth(
            text, raw,
            voice=cfg["voice"],
            speed=cfg["speed"],
            lang=cfg["lang"],
            exaggeration=cfg.get("exaggeration"),
            cfg_weight=cfg.get("cfg_weight")
        )
        enc = [FFMPEG, "-y", "-loglevel", "error", "-i", raw]
        if cfg.get("trim_silence", True):
            enc += ["-af", _TRIM_AF]
        enc += ["-ar", str(SR), "-ac", "2", wav]
        run(enc)
        dur = wav_duration(wav)
        clips.append((wav, dur))
        print(f"  seg {i:02d}: {dur:5.2f}s  {text[:54]}")
    return clips


def make_silence(path, seconds):
    run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"anullsrc=r={SR}:cl=stereo", "-t", f"{seconds:.3f}", path])


def build_voice_track(clips, cfg, work):
    parts = []
    intro = os.path.join(work, "sil_intro.wav")
    make_silence(intro, cfg["intro"])
    parts.append(intro)
    for i, (wav, _d) in enumerate(clips, 1):
        parts.append(wav)
        gap = os.path.join(work, f"sil_gap_{i:02d}.wav")
        make_silence(gap, cfg["gap"])
        parts.append(gap)
    outro = os.path.join(work, "sil_outro.wav")
    make_silence(outro, cfg["outro"])
    parts.append(outro)

    listfile = os.path.join(work, "concat.txt")
    with open(listfile, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    track = os.path.join(work, "voice_full.wav")
    run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", listfile, "-c", "copy", track])
    return track, wav_duration(track)


def list_music_library():
    """Return sorted list of playable track paths in the music/ library."""
    if not os.path.isdir(MUSIC_DIR):
        return []
    tracks = [f for f in os.listdir(MUSIC_DIR)
              if f.lower().endswith((".mp3", ".wav", ".ogg", ".m4a", ".flac"))]
    return [os.path.join(MUSIC_DIR, f) for f in sorted(tracks)]


def pick_music_track(music, seed):
    """Resolve which library track to use.

    Priority: explicit `track` in config → otherwise rotate deterministically from
    the seed (slug) so each video gets a stable, well-spread choice.
    """
    track = music.get("track")
    if track:
        cand = track if os.path.isabs(track) else os.path.join(MUSIC_DIR, track)
        if os.path.isfile(cand):
            return cand
        print(f"  ! music track not found: {track} — falling back to rotation")
    library = list_music_library()
    if not library:
        return None
    key = str(music.get("seed") or seed or "default")
    idx = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % len(library)
    return library[idx]


def prepare_music_file(track, total, out_wav, fade=1.5):
    """Loop/trim `track` to `total` seconds with fade in/out → stereo wav at SR."""
    fade = max(0.0, float(fade))
    fade_out_start = max(0.0, total - fade)
    af = (f"afade=t=in:st=0:d={fade:.3f},"
          f"afade=t=out:st={fade_out_start:.3f}:d={fade:.3f}")
    run([FFMPEG, "-y", "-loglevel", "error",
         "-stream_loop", "-1", "-i", track,
         "-t", f"{total:.3f}", "-af", af,
         "-ar", str(SR), "-ac", "2", out_wav])
    return out_wav


def mix_music(voice_wav, cfg, seed, total, work):
    """Mix the voice track with background music. Returns mixed wav path."""
    music = cfg["music"]
    if not music.get("enabled", True):
        return voice_wav
    music_wav = os.path.join(work, "music.wav")
    source = music.get("source", "library")
    if source == "procedural":
        vo_music.generate(music_wav, total + 0.2,
                          seed=music.get("seed") or seed,
                          mood=music.get("mood", "dark"))
    else:
        track = pick_music_track(music, seed)
        if not track:
            print("  ! music library empty — skipping background music")
            return voice_wav
        print(f"  ♪ background music: {os.path.basename(track)}")
        prepare_music_file(track, total + 0.2, music_wav,
                           fade=music.get("fade", 1.5))
    v, _ = sf.read(voice_wav, dtype="float32", always_2d=True)
    m, _ = sf.read(music_wav, dtype="float32", always_2d=True)
    if m.shape[1] == 1 and v.shape[1] == 2:
        m = np.repeat(m, 2, axis=1)
    n = min(len(v), len(m))
    gain = 10 ** (float(music.get("gain_db", -26.0)) / 20.0)
    mixed = v[:n] + m[:n] * gain
    peak = float(np.max(np.abs(mixed))) or 1.0
    if peak > 0.99:
        mixed *= 0.99 / peak
    out = os.path.join(work, "mixed.wav")
    sf.write(out, mixed, SR)
    return out


def record(clips, cfg, index_html, work):
    from playwright.sync_api import sync_playwright
    holds = [d + cfg["gap"] for _w, d in clips]
    video_dir = os.path.join(work, "video")
    os.makedirs(video_dir, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--force-color-profile=srgb"])
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            device_scale_factor=2,
            record_video_dir=video_dir,
            record_video_size={"width": W, "height": H},
        )
        page = ctx.new_page()
        page.goto("file://" + index_html)
        page.add_style_tag(content=(
            ".hud{display:none!important;}"
            "#stage{border-radius:0!important;box-shadow:none!important;}"
            "body{background:#0d1117!important;}"
        ))
        page.wait_for_timeout(int(cfg["intro"] * 1000))
        for hold in holds:
            page.keyboard.press("ArrowRight")
            page.wait_for_timeout(int(hold * 1000))
        page.wait_for_timeout(int(cfg["outro"] * 1000))
        video = page.video
        ctx.close()
        browser.close()
        raw = video.path()
    out = os.path.join(work, "screen.webm")
    shutil.move(raw, out)
    return out


def mux(video, audio, final_mp4, ass_path=None):
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-i", video, "-i", audio]
    if ass_path:
        cmd += ["-vf", f"ass={ass_path}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
            "-crf", "18", "-r", "30",
            "-c:a", "aac", "-b:a", "192k", "-shortest", final_mp4]
    run(cmd)


def safe_overlay(index_html, work, short_dir):
    """Render one frame with the safe-zone guide visible for visual QA."""
    from playwright.sync_api import sync_playwright
    out = os.path.join(short_dir, "safezone_preview.png")
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--force-color-profile=srgb"])
        ctx = browser.new_context(viewport={"width": W, "height": H},
                                  device_scale_factor=2)
        page = ctx.new_page()
        page.goto("file://" + index_html)
        page.add_style_tag(content=".safe-guide{display:block!important;}")
        page.wait_for_timeout(400)
        page.screenshot(path=out)
        browser.close()
    print(f"safe-zone preview: {out}")
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: build_short.py <path-to-short-dir> [--safe-preview]")
    short_dir = os.path.abspath(sys.argv[1])
    index_html = os.path.join(short_dir, "index.html")
    if not os.path.isfile(index_html):
        sys.exit(f"no index.html in {short_dir}")

    cfg = load_config(short_dir)
    lines = read_lines(short_dir)
    seed = os.path.basename(short_dir.rstrip("/"))
    print(f"short: {seed}\nvoice: {cfg['voice']}  music: {cfg['music']}")
    print(f"narration lines: {len(lines)}")

    work = os.path.join(short_dir, "_work")
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work)

    if "--safe-preview" in sys.argv:
        safe_overlay(index_html, work, short_dir)

    clips = synth(lines, cfg, work)
    voice_track, total = build_voice_track(clips, cfg, work)
    audio = mix_music(voice_track, cfg, seed, total, work)
    video = record(clips, cfg, index_html, work)

    ass_path = None
    if cfg.get("subtitles", True) and lines:
        ass_path = generate_ass(clips, cfg, lines, work)

    final_mp4 = os.path.join(short_dir, "final.mp4")
    mux(video, audio, final_mp4, ass_path=ass_path)
    print(f"\n✅ {final_mp4}")
    run([FFPROBE, "-v", "error", "-show_entries",
         "format=duration:stream=width,height,codec_name",
         "-of", "default=noprint_wrappers=1", final_mp4])


if __name__ == "__main__":
    main()

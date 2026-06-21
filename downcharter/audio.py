"""audio.py — Optional audio analysis to refine section energy.

The MIDI gives perfect timing/sections/instruments; the audio adds the one thing
that's missing: the real VOLUME DYNAMICS (RMS). It distinguishes a quiet verse
from a loud chorus better than note density. Used only to refine `Section.energy`.

Lightweight dependencies: numpy + soundfile (libsndfile 1.0.29+ reads
.ogg/.opus/.wav/.mp3 without ffmpeg; .opus is Ogg Opus, read via the same OGG
container). Unencrypted .mogg (Rock Band): skip the header and read the embedded
OGG stream. Everything is optional — with no audio (or no libs), the MIDI-only
pipeline stays intact.
"""
from __future__ import annotations
import os
import struct
import io

from .midi_utils import tick_to_ms

_AUDIO_EXTS = (".ogg", ".opus", ".wav", ".flac", ".mp3", ".mogg")


def available() -> bool:
    """True if the audio libs are installed."""
    try:
        import numpy  # noqa: F401
        import soundfile  # noqa: F401
        return True
    except ImportError:
        return False


# Stems that do NOT represent the band (don't add to the energy mix).
_NON_BAND_STEMS = ("crowd", "click", "guide")

# Filename keywords that mark an ISOLATED vocal stem.
_VOCAL_STEM_KEYS = ("vocal", "vox")


def find_vocal_stems(folder: str) -> list[str]:
    """Isolated vocal stem files in the folder (filename contains 'vocal'/'vox').
    Empty if the song only ships a multichannel .mogg (no separated voice), in
    which case the talky-sustain confirmation must fall back to geometry."""
    if not os.path.isdir(folder):
        return []
    out = []
    for f in os.listdir(folder):
        low = f.lower()
        if (low.endswith(_AUDIO_EXTS) and not low.endswith(".mogg")
                and any(k in low for k in _VOCAL_STEM_KEYS)
                and "harm" not in low):   # keep only the lead vocal stem
            out.append(os.path.join(folder, f))
    return sorted(out)


def voice_activity(paths, hop_s: float = 0.05):
    """RMS envelope of the vocal stem(s) plus a 'voice-present' threshold.
    Returns (env, hop_s, thr) or None on failure. The threshold is relative:
    floor (20th pct) + 12% of the dynamic range, so 'voice present' adapts to the
    stem's own noise floor. Used to confirm a singer is actually holding a note."""
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        return None
    try:
        import numpy as np
        mono, sr = load_mono_mix(paths)
        env = rms_envelope(mono, sr, hop_s)
        if len(env) < 4:
            return None
        peak = float(env.max())
        floor = float(np.percentile(env, 20))
        if peak <= 0:
            return None
        thr = floor + 0.12 * (peak - floor)
        return env, hop_s, thr
    except Exception:
        return None


def voice_offset_s(va, start_s: float, ceil_s: float,
                   gap_s: float = 0.15) -> float | None:
    """Second at which the voice falls silent after `start_s` (silence sustained
    for at least `gap_s`), searching up to `ceil_s`. Returns None if the voice
    persists all the way to the ceiling (genuine sustain). `va` is the tuple from
    `voice_activity`."""
    if va is None:
        return None
    env, hop_s, thr = va
    k = max(0, int(start_s / hop_s))
    below = 0.0
    while k < len(env) and k * hop_s < ceil_s:
        if env[k] <= thr:
            below += hop_s
            if below >= gap_s:
                return k * hop_s - below + hop_s   # start of the silent run
        else:
            below = 0.0
        k += 1
    return None


def find_song_audio(folder: str) -> list[str]:
    """Song audio in the folder (YARG layout). Returns a LIST:
      - 1 multichannel .mogg (already contains every stem) → [that one], OR
      - several separate .ogg/.opus/.wav stems → all of them (summed), except crowd/click.
    Empty list if there is no audio."""
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(_AUDIO_EXTS)]
    if not files:
        return []
    moggs = [f for f in files if f.lower().endswith(".mogg")]
    stems = [f for f in files if not f.lower().endswith(".mogg")
             and not any(k in os.path.basename(f).lower() for k in _NON_BAND_STEMS)]
    # Prefer separate .ogg stems (more informative); otherwise the multichannel .mogg.
    if stems:
        return sorted(stems)
    return moggs[:1]


def load_mono(path: str):
    """Decode the audio to mono float32. Handles .mogg (header strip)."""
    import numpy as np
    import soundfile as sf

    if path.lower().endswith(".mogg"):
        raw = open(path, "rb").read()
        version = struct.unpack("<I", raw[:4])[0]
        if version != 0x0A:
            raise ValueError(f"encrypted mogg (version {version}) not supported")
        offset = struct.unpack("<I", raw[4:8])[0]
        data, sr = sf.read(io.BytesIO(raw[offset:]), dtype="float32", always_2d=True)
    else:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def load_mono_mix(paths: list[str]):
    """Mix (sum) several stems into a single mono. Stems with different sample
    rates are resampled (linearly) to the sr of the first one. Aligns to the
    shortest stem. A single multichannel .mogg is already summed by load_mono."""
    import numpy as np
    if not paths:
        raise ValueError("no stems")
    base_sr = None
    mixes: list = []
    for p in paths:
        mono, sr = load_mono(p)
        if base_sr is None:
            base_sr = sr
        elif sr != base_sr and len(mono):
            # simple linear resampling to the base sr
            n = int(len(mono) * base_sr / sr)
            mono = np.interp(np.linspace(0, len(mono), n, endpoint=False),
                             np.arange(len(mono)), mono).astype("float32")
        mixes.append(mono)
    n = min(len(m) for m in mixes)
    mix = np.zeros(n, dtype="float32")
    for m in mixes:
        mix += m[:n]
    return mix, base_sr


def audio_duration_seconds(paths) -> float | None:
    """Duration (in seconds) of the song audio, without decoding the whole file.
    For several stems returns the LONGEST (the song ends when the last stem ends).
    Handles .mogg (header strip). Returns None if it can't be read / no libs."""
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        return None
    try:
        import soundfile as sf
    except Exception:
        return None
    best: float | None = None
    for p in paths:
        try:
            if p.lower().endswith(".mogg"):
                raw = open(p, "rb").read()
                version = struct.unpack("<I", raw[:4])[0]
                if version != 0x0A:
                    continue
                offset = struct.unpack("<I", raw[4:8])[0]
                info = sf.info(io.BytesIO(raw[offset:]))
            else:
                info = sf.info(p)
            dur = float(info.frames) / float(info.samplerate)
            if best is None or dur > best:
                best = dur
        except Exception:
            continue
    return best


def rms_envelope(mono, sr: int, hop_s: float = 0.1):
    """RMS envelope (one sample per `hop_s` seconds)."""
    import numpy as np
    hop = max(1, int(sr * hop_s))
    n = max(1, (len(mono)) // hop)
    env = np.empty(n, dtype="float32")
    for i in range(n):
        seg = mono[i * hop:(i + 1) * hop]
        env[i] = np.sqrt(np.mean(seg * seg)) if len(seg) else 0.0
    return env


def _stft_mag(mono, sr: int, hop: int = 1024, win: int = 2048):
    """Vectorized STFT magnitude (frames × bins). Returns (mag, hop)."""
    import numpy as np
    if len(mono) < win:
        return np.zeros((0, win // 2 + 1), dtype="float32"), hop
    n = 1 + (len(mono) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = mono[idx] * np.hanning(win).astype("float32")
    return np.abs(np.fft.rfft(frames, axis=1)).astype("float32"), hop


def percussive_onset_ticks(paths, tempo_map, tpb: int,
                           hop: int = 1024, win: int = 2048,
                           thr: float = 0.06) -> list[int] | None:
    """Pseudo-drums from the AUDIO (spectral flux): percussive onsets of the full
    mix, in TICKS. For charts without PART DRUMS — feeds keyframes/pyro/sync.
    Validated: ~99% recall of the real hits. Returns None on failure."""
    if isinstance(paths, str):
        paths = [paths]
    try:
        import numpy as np
        mono, sr = load_mono_mix(paths)
        mag, hop = _stft_mag(mono, sr, hop, win)
        if mag.shape[0] < 4:
            return None
        flux = np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)
        flux /= (flux.max() + 1e-9)
    except Exception:
        return None
    times: list[float] = []
    for i in range(2, len(flux) - 2):
        f = flux[i]
        if f > thr and f >= flux[i - 1] and f > flux[i + 1] and f > flux[i - 2]:
            times.append((i * hop + win / 2) / sr * 1000.0)   # ms
    return [_ms_to_tick(ms, tempo_map, tpb) for ms in times]


# Frequency bands → per-instrument activity proxy (no ML separation).
_BANDS = {"bass": (40, 250), "drums": (3000, 12000), "lead": (300, 3000)}


def band_activity_ticks(paths, tempo_map, tpb: int, band: str,
                        hop: int = 2048, win: int = 4096,
                        thr_pct: float = 60.0) -> list[int] | None:
    """Pseudo-onsets of ONE absent instrument, via the energy in its frequency
    band. 'bass' (lows), 'drums' (high transients), 'lead' (mids). Returns ticks
    where the band is active (above the thr_pct percentile). Approximate."""
    if band not in _BANDS:
        return None
    if isinstance(paths, str):
        paths = [paths]
    try:
        import numpy as np
        mono, sr = load_mono_mix(paths)
        mag, hop = _stft_mag(mono, sr, hop, win)
        if mag.shape[0] < 4:
            return None
        freqs = np.fft.rfftfreq(win, 1.0 / sr)
        lo, hi = _BANDS[band]
        sel = (freqs >= lo) & (freqs < hi)
        env = mag[:, sel].sum(axis=1)
        env /= (env.max() + 1e-9)
        thr = np.percentile(env, thr_pct)
    except Exception:
        return None
    out: list[int] = []
    for i in range(len(env)):
        if env[i] >= thr:
            ms = (i * hop + win / 2) / sr * 1000.0
            out.append(_ms_to_tick(ms, tempo_map, tpb))
    return out


def _ms_to_tick(ms: float, tempo_map, tpb: int) -> int:
    """Inverse of tick_to_ms: absolute ms → absolute tick (walks the tempo_map)."""
    from .midi_utils import DEFAULT_TEMPO
    acc_ms = 0.0
    prev_t, prev_u = 0, DEFAULT_TEMPO
    for mt, mu in tempo_map:
        seg_ms = (mt - prev_t) / tpb * (prev_u / 1000.0)
        if acc_ms + seg_ms >= ms:
            break
        acc_ms += seg_ms
        prev_t, prev_u = mt, mu
    return int(prev_t + (ms - acc_ms) * 1000.0 / prev_u * tpb)


def section_energy_tiers(paths, sections, tempo_map, tpb: int,
                         hop_s: float = 0.1) -> list[str] | None:
    """Loudness tier ('calm'/'mid'/'high') per section, via thirds of the mean RMS.
    `paths` may be a single path or a list of stems (summed into the mix).
    Returns None if the audio can't be read. Direct alignment: the RB MIDI is
    authored over the audio (same timeline) → tick→ms via the tempo_map."""
    if not sections:
        return None
    if isinstance(paths, str):
        paths = [paths]
    try:
        import numpy as np
        mono, sr = load_mono_mix(paths)
        env = rms_envelope(mono, sr, hop_s)
    except Exception:
        return None
    if len(env) == 0:
        return None

    means: list[float] = []
    for s in sections:
        a = tick_to_ms(s.start, tempo_map, tpb) / 1000.0 / hop_s
        b = tick_to_ms(s.end, tempo_map, tpb) / 1000.0 / hop_s
        i0 = max(0, int(a))
        i1 = min(len(env), max(i0 + 1, int(b)))
        seg = env[i0:i1]
        means.append(float(np.mean(seg)) if len(seg) else 0.0)

    # Thirds of the distribution (same density-driven logic as the sections).
    s = sorted(means)
    lo = s[len(s) // 3]
    hi = s[2 * len(s) // 3]
    out: list[str] = []
    for v in means:
        out.append("calm" if v <= lo else ("mid" if v < hi else "high"))
    return out

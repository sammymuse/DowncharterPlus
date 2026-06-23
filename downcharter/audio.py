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


def flux_accents(paths, tempo_map, tpb: int, top_pct: float = 90.0,
                 min_gap_s: float = 0.40, hop: int = 1024,
                 win: int = 2048) -> list[int] | None:
    """STRONG spectral-flux transients (chorus hits, crashes, synth stabs) — a
    SELECTIVE subset of percussive_onset_ticks meant to PUNCTUATE the lightshow
    (snap light changes / pyro to real musical hits), NOT to reproduce every drum
    hit. Song-relative: keeps only local flux maxima above the `top_pct` percentile
    of all peak heights, spaced >= min_gap_s apart. No absolute threshold — a quiet
    song and a loud one both surface their own strongest ~10% of transients.
    Catches AUDIO-only hits the MIDI drums miss (electronic stabs, orchestral crashes).
    Returns ticks (ascending) or None on failure / no audio."""
    if isinstance(paths, str):
        paths = [paths]
    try:
        import numpy as np
        mono, sr = load_mono_mix(paths)
        mag, hop = _stft_mag(mono, sr, hop, win)
        if mag.shape[0] < 5:
            return None
        flux = np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)
        flux /= (flux.max() + 1e-9)
        peaks = [(i, float(flux[i])) for i in range(2, len(flux) - 2)
                 if flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]
                 and flux[i] > flux[i - 2] and flux[i] >= flux[i + 2]]
        if not peaks:
            return None
        thr = float(np.percentile([f for _, f in peaks], top_pct))
        min_gap = max(1, int(min_gap_s * sr / hop))
        out: list[int] = []
        last = -10 ** 9
        for i, f in peaks:
            if f >= thr and i - last >= min_gap:
                out.append(_ms_to_tick((i * hop + win / 2) / sr * 1000.0,
                                       tempo_map, tpb))
                last = i
        return out or None
    except Exception:
        return None


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


def _rank01(values):
    """Song-relative rank of each value in [0, 1] (min→0, max→1). Ties broken by
    order; a flat distribution maps everything to 0.5. Keeps every audio cue on a
    common, scale-free axis so loudness/flux/brightness can be blended fairly.
    NOTE: rank FLATTENS magnitude (it only asks 'how many frames are below this?'),
    so a song with lots of quiet material inflates a merely-moderate section just for
    sitting above the floor. For the intensity envelope use _scale01 instead, which
    preserves magnitude. _rank01 is kept for the MIDI cues (note-count ordering)."""
    import numpy as np
    v = np.asarray(values, dtype="float64")
    n = len(v)
    if n <= 1 or float(v.max() - v.min()) < 1e-12:
        return np.full(n, 0.5)
    order = v.argsort().argsort().astype("float64")
    return order / (n - 1)


def _scale01(values):
    """Song-relative MAGNITUDE scaling to [0, 1]: (x − p10) / (p90 − p10), clipped.
    Still song-relative (no absolute dB — robust floor/ceiling are the song's own
    p10/p90), but unlike _rank01 it preserves MAGNITUDE: a section is judged against
    the song's loud ceiling, not against how many frames it beats. A merely-moderate
    verse that sits well below the choruses/breakdown maps LOW (calm) instead of being
    inflated to mid just for being above the quiet parts. A flat song → all 0.5."""
    import numpy as np
    v = np.asarray(values, dtype="float64")
    n = len(v)
    if n == 0:
        return v
    lo = float(np.percentile(v, 10))
    hi = float(np.percentile(v, 90))
    if hi - lo < 1e-12:
        return np.full(n, 0.5)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


# Empirical FEEL weights — derived from feel_study.py over the 20 venue learn
# songs (HIGH = pyro/strobe/intense/keyframe vs CALM = blackout). Each cue is the
# song-relative percentile rank; the weight is its discrimination gap (HIGH−CALM
# median), normalized. Intensity is FEEL, not amplitude: loudness/flux lead, but
# heaviness (low-end), wall/distortion (flatness) and transient hardness (density)
# add real signal — and BRIGHTNESS IS INVERTED (high intensity is DARKER/heavier,
# gap −16.5; the old code added it positively, pulling the wrong way).
_FEEL_W = {"loud": 0.34, "flux": 0.27, "flat": 0.11,
           "low": 0.11, "dens": 0.06, "bright": -0.11}


def _feel_frames(mono, sr):
    """Per-STFT-frame FEEL cues for the intensity score. Returns (feats, stft_s)
    where feats maps name→array: loud, flux, bright, low (=<150 Hz power ratio),
    dens (smoothed transient density), flat (spectral flatness)."""
    import numpy as np
    mag, hop = _stft_mag(mono, sr)
    if mag.shape[0] < 4:
        return None, None
    P = mag ** 2
    win = (mag.shape[1] - 1) * 2
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    tot = P.sum(axis=1) + 1e-9
    loud = np.sqrt(P.mean(axis=1))
    flux = np.concatenate([[0.0], np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)])
    bright = (P * freqs[None, :]).sum(axis=1) / tot
    low = P[:, freqs < 150].sum(axis=1) / tot
    gm = np.exp(np.log(P + 1e-12).mean(axis=1))
    flat = gm / (P.mean(axis=1) + 1e-12)
    stft_s = hop / sr
    thr = np.percentile(flux, 70)
    hot = (flux > thr).astype("float32")
    k = max(1, int(0.5 / stft_s))                 # ~0.5 s smoothing
    dens = np.convolve(hot, np.ones(k) / k, mode="same")
    return ({"loud": loud, "flux": flux, "bright": bright,
             "low": low, "dens": dens, "flat": flat}, stft_s)


def feel_envelope(paths, hop_s: float = 0.1):
    """Continuous FEEL intensity envelope in [0, 1], one value per STFT frame.
    Each cue is percentile-ranked song-relative, blended with the empirical
    _FEEL_W weights (brightness inverted). This is the song's REAL intensity shape
    — used both for per-section means and for sub-section (within-section) dynamics.
    Returns (env, stft_s) or (None, None) on failure."""
    if isinstance(paths, str):
        paths = [paths]
    try:
        import numpy as np
        mono, sr = load_mono_mix(paths)
        feats, stft_s = _feel_frames(mono, sr)
    except Exception:
        return None, None
    if feats is None:
        return None, None
    import numpy as np
    n = len(feats["loud"])
    comp = np.zeros(n, dtype="float64")
    for k, w in _FEEL_W.items():
        r = _scale01(feats[k])     # song-relative MAGNITUDE (not rank — keeps feel)
        comp += abs(w) * (1.0 - r if w < 0 else r)
    return comp, stft_s


def section_energy_scores(paths, sections, tempo_map, tpb: int,
                          hop_s: float = 0.1) -> list[float] | None:
    """Composite FEEL energy score per section in [0, 1] — the mean of the
    continuous feel_envelope over each section. See _FEEL_W / feel_envelope:
    loudness+flux lead, heaviness/wall/hardness add the 'feel', brightness is
    inverted (heavy = dark). Returns None on failure."""
    if not sections:
        return None
    env, stft_s = feel_envelope(paths, hop_s)
    if env is None:
        return None
    import numpy as np
    out: list[float] = []
    for s in sections:
        a_s = tick_to_ms(s.start, tempo_map, tpb) / 1000.0
        b_s = tick_to_ms(s.end, tempo_map, tpb) / 1000.0
        j0 = max(0, int(a_s / stft_s))
        j1 = min(len(env), max(j0 + 1, int(b_s / stft_s)))
        out.append(float(np.mean(env[j0:j1])) if j1 > j0 else 0.0)
    return out


def section_energy_tiers(paths, sections, tempo_map, tpb: int,
                         hop_s: float = 0.1) -> list[str] | None:
    """Energy tier ('calm'/'mid'/'high') per section from the composite FEEL score
    (see section_energy_scores / _FEEL_W). FIXED thresholds on the song-relative
    score (NOT forced thirds): 'high' is rare and a near-flat song reads all-'mid'.
    Returns None if the audio can't be read. Kept for callers that want tiers directly."""
    scores = section_energy_scores(paths, sections, tempo_map, tpb, hop_s)
    if scores is None:
        return None
    return ["calm" if v < 0.45 else ("mid" if v < 0.62 else "high") for v in scores]


def section_brightness_tiers(paths, sections, tempo_map, tpb: int,
                             hop_s: float = 0.1) -> list[str] | None:
    """Color-temperature tier per section from the audio TIMBRE (mean spectral
    centroid), song-relative: the BRIGHTEST third → 'warm', the DARKEST third →
    'cool', the middle → None (neutral). Mapping inverted after the venue_audio_study
    ground-truth pass (the 20 official venues put 'warm' presets in brighter audio,
    bright_p 63.6 vs 57.1 for 'cool'). No absolute frequency — thirds of the song's
    own range. Returns None if the audio can't be read."""
    if not sections:
        return None
    if isinstance(paths, str):
        paths = [paths]
    try:
        import numpy as np
        mono, sr = load_mono_mix(paths)
        mag, hop = _stft_mag(mono, sr)
    except Exception:
        return None
    if mag.shape[0] < 4:
        return None
    win = (mag.shape[1] - 1) * 2
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    centroid = (mag * freqs[None, :]).sum(axis=1) / (mag.sum(axis=1) + 1e-9)
    stft_s = hop / sr
    bright: list[float] = []
    for s in sections:
        a_s = tick_to_ms(s.start, tempo_map, tpb) / 1000.0
        b_s = tick_to_ms(s.end, tempo_map, tpb) / 1000.0
        j0 = max(0, int(a_s / stft_s))
        j1 = min(mag.shape[0], max(j0 + 1, int(b_s / stft_s)))
        bright.append(float(np.mean(centroid[j0:j1])) if j1 > j0 else 0.0)
    vals = sorted(bright)
    lo = vals[len(vals) // 3]
    hi = vals[2 * len(vals) // 3]
    return ["cool" if v <= lo else ("warm" if v > hi else None) for v in bright]


def flux_strobe_spans(paths, tempo_map, tpb: int, hi_pct: float = 88.0,
                      min_beats: float = 1.75, hop: int = 1024,
                      win: int = 2048) -> list[tuple[int, int]] | None:
    """Audio strobe spans: stretches where the spectral flux stays SUSTAINED above the
    song's own `hi_pct` percentile (a tremolo / blast 'wall'). Catches audio-driven
    walls (electronic, shoegaze, orchestral crescendos) that the MIDI drums don't flag.
    The CONTINUITY requirement (>= min_beats above threshold) is the natural gate: a
    calm song's loud frames are isolated strums that never sustain, so it produces no
    span — no absolute level needed. Returns (start_tick, end_tick) spans or None."""
    if isinstance(paths, str):
        paths = [paths]
    try:
        import numpy as np
        mono, sr = load_mono_mix(paths)
        mag, hop = _stft_mag(mono, sr, hop, win)
        if mag.shape[0] < 8:
            return None
        flux = np.concatenate([[0.0], np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)])
        w = max(1, int(0.15 * sr / hop))                  # ~0.15 s smoothing
        sm = np.convolve(flux, np.ones(w) / w, mode="same")
        thr = float(np.percentile(sm, hi_pct))
        active = sm >= thr
    except Exception:
        return None
    spans: list[tuple[int, int]] = []
    n = len(active)
    i = 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            a = _ms_to_tick((i * hop + win / 2) / sr * 1000.0, tempo_map, tpb)
            b = _ms_to_tick((j * hop + win / 2) / sr * 1000.0, tempo_map, tpb)
            spans.append((a, b))
            i = j
        else:
            i += 1
    # Merge runs separated by < 1/2 beat, then keep only the sustained ones.
    bridge = tpb // 2
    merged: list[tuple[int, int]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] < bridge:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    min_ticks = int(min_beats * tpb)
    return [(a, b) for a, b in merged if b - a >= min_ticks] or None


def calm_blackout_ticks(paths, sections, tempo_map, tpb: int,
                        sub_beats: int = 2, min_beats: float = 4.0,
                        hop_s: float = 0.1) -> list[int] | None:
    """Blackout anchors = the START of each contiguous LOW-energy region of the song.
    Ground-truth driven: the venue_audio_study over the 20 official venues showed
    blackout lighting sits in CALM states (loud_p median 36, 46% in the bottom third)
    and NOT at a sharp intensity fall (drop ratio ~1.0). So instead of find_drops'
    build->drop transients, we anchor a blackout where the song's energy is genuinely
    low. Reuses energy_envelope's song-relative 'calm' tier, merges consecutive calm
    spans, and keeps regions lasting >= min_beats. Returns one tick per calm region
    (its start), or None. Song-relative: a uniformly loud song yields no calm region."""
    env = energy_envelope(paths, sections, tempo_map, tpb, sub_beats, hop_s)
    if not env:
        return None
    span_ticks = max(1, sub_beats) * tpb
    out: list[int] = []
    run_start = None
    run_len = 0
    for start, tier in env:
        if tier == "calm":
            if run_start is None:
                run_start = start
            run_len += span_ticks
        else:
            if run_start is not None and run_len >= int(min_beats * tpb):
                out.append(run_start)
            run_start, run_len = None, 0
    if run_start is not None and run_len >= int(min_beats * tpb):
        out.append(run_start)
    return out or None


def find_drops(paths, tempo_map, tpb: int, win_s: float = 0.5,
               hi_pct: float = 70.0, drop_ratio: float = 0.45,
               hop: int = 1024, win: int = 2048) -> list[int] | None:
    """SUPERSEDED for blackout placement by calm_blackout_ticks (the venue_audio_study
    showed official blackouts sit in calm states, not at sharp falls). Kept for
    reference / possible future 'drop' cues.
    Drop moments: a sharp COLLAPSE of intensity (loudness+flux) right after a
    sustained-loud stretch — the classic build->drop / breakdown entry. For each frame
    boundary, compares the mean intensity of the previous `win_s` seconds against the
    next: a drop = previous window in the song's loud range (>= hi_pct percentile) AND
    the next window falls to <= drop_ratio of it. Song-relative — a steadily quiet (or
    steadily loud) song yields no drop. Returns the drop ticks, or None."""
    if isinstance(paths, str):
        paths = [paths]
    try:
        import numpy as np
        mono, sr = load_mono_mix(paths)
        mag, hop = _stft_mag(mono, sr, hop, win)
        if mag.shape[0] < 8:
            return None
        flux = np.concatenate([[0.0], np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)])
        loud = mag.sum(axis=1)
        inten = flux / (flux.max() + 1e-9) + loud / (loud.max() + 1e-9)
        w = max(1, int(0.2 * sr / hop))                   # ~0.2 s smoothing
        inten = np.convolve(inten, np.ones(w) / w, mode="same")
        hi = float(np.percentile(inten, hi_pct))
    except Exception:
        return None
    stft_s = hop / sr
    fw = max(1, int(win_s / stft_s))
    out: list[int] = []
    i = fw
    n = len(inten)
    while i < n - fw:
        prev = float(inten[i - fw:i].mean())
        nxt = float(inten[i:i + fw].mean())
        if prev >= hi and nxt <= prev * drop_ratio:
            out.append(_ms_to_tick((i * hop + win / 2) / sr * 1000.0, tempo_map, tpb))
            i += fw                                       # skip past this drop
        else:
            i += 1
    return out or None


def energy_envelope(paths, sections, tempo_map, tpb: int, sub_beats: int = 2,
                    hop_s: float = 0.1) -> list[tuple[int, str]] | None:
    """Within-section energy envelope at SUB-SECTION resolution. Splits the whole song
    into spans of `sub_beats` beats and scores each with the composite FEEL cue
    (section_energy_scores / _FEEL_W: loud+flux lead, heaviness/wall/hardness add the
    feel, brightness inverted), then FIXED thresholds → 'calm'/'mid'/'high'. Returns a
    sorted list of (start_tick, tier) breakpoints, or None. Intensity is NOT locked to
    section boundaries: a chorus can start 'calm' and ramp to 'mid' mid-way — the
    lightshow/camera follow the real moment-to-moment feel, not one tier per section."""
    if not sections:
        return None
    from types import SimpleNamespace
    span_len = max(1, sub_beats) * tpb
    spans: list[SimpleNamespace] = []
    for s in sections:
        t = s.start
        while t < s.end:
            spans.append(SimpleNamespace(start=t, end=min(t + span_len, s.end)))
            t += span_len
    if len(spans) < 3:
        return None
    scores = section_energy_scores(paths, spans, tempo_map, tpb, hop_s)
    if scores is None:
        return None
    return [(spans[i].start,
             "calm" if scores[i] < 0.45 else ("mid" if scores[i] < 0.62 else "high"))
            for i in range(len(spans))]

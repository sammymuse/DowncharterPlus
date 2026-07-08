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
import sys
import struct
import io

import numpy as np

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

# Default vocal channel indices inside a 12-channel .mogg (RB3 standard layout).
# Layout: drum=(0,1), bass=(2,3), guitar=(4,5), vocals=(6,7), keys=(8,9), crowd=(10,11)
_MOGG_VOCAL_CHANNELS = (6, 7)


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
            # Exclude our own cache files — resolve_vocal_audio manages those
            if f in _VOCAL_CACHE_NAMES:
                continue
            out.append(os.path.join(folder, f))
    return sorted(out)


def find_vocal_audio(folder: str) -> list[str] | None:
    """Deprecated — use ``resolve_vocal_audio`` instead, which also handles
    ``.mogg`` extraction and MDX-NET separation as fallback."""
    stems = find_vocal_stems(folder)
    if stems:
        return stems
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.lower().endswith(".mogg"):
                return None
    return []


# ── Vocal separation via MDX-NET (fallback when no isolated stems) ─────

_VOCAL_CACHE_MOGG = ".downcharter_vocals_mogg.wav"
_VOCAL_CACHE_MDX = ".downcharter_vocals_mdx.wav"
# Cache filenames — must follow the constants above (used by find_song_audio
# to exclude cache files from the full-mix stem list).
_VOCAL_CACHE_NAMES = (_VOCAL_CACHE_MOGG, _VOCAL_CACHE_MDX)


def _vocal_source_from_path(p: str) -> str:
    """Return 'mdx', 'mogg', or 'stems' based on cache file name."""
    bn = os.path.basename(p)
    if bn == _VOCAL_CACHE_MDX:
        return "mdx"
    if bn == _VOCAL_CACHE_MOGG:
        return "mogg"
    return "stems"


def _clean_vocal_cache(mid_path: str, cache_path: str | None = None) -> None:
    """Remove the vocal-separation cache file for this song.

    If ``cache_path`` is given, only that file is removed (avoids race
    conditions with concurrent processing).  Otherwise scans the song
    folder for ``.downcharter_vocals_*.wav`` as fallback.

    The cache is ~9 MB per song and is only needed during this song's
    processing step (voice activity, syllable trimming).  Deleting after
    each song keeps 1000-song batches lean.
    """
    if cache_path is not None:
        try:
            os.remove(cache_path)
        except Exception as exc:
            sys.stderr.write(f"  [cache] cleanup warning: {exc}\n")
        return
    # Fallback: scan the folder (legacy / unknown paths)
    folder = os.path.dirname(os.path.abspath(mid_path))
    for f in os.listdir(folder):
        if f.startswith(".downcharter_vocals_") and f.endswith(".wav"):
            try:
                os.remove(os.path.join(folder, f))
            except Exception as exc:
                sys.stderr.write(f"  [cache] cleanup warning: {exc}\n")


def _cache_is_fresh(cache_path: str, source_paths: list[str]) -> bool:
    """True if the cache file exists and is newer than all sources."""
    if not os.path.isfile(cache_path):
        return False
    try:
        ctime = os.path.getmtime(cache_path)
        return all(os.path.isfile(p) and os.path.getmtime(p) <= ctime
                   for p in source_paths)
    except Exception:
        return False


def _atomic_write_cache(cache_path: str, mono, sr: int) -> bool:
    """Write ``mono`` to ``cache_path`` atomically via a temp file.
    Returns True on success, False on any I/O or encoding error.
    The temp file is cleaned up automatically (even on failure)."""
    import soundfile as sf
    import tempfile
    folder = os.path.dirname(cache_path) or "."
    try:
        tmp = tempfile.NamedTemporaryFile(dir=folder, suffix=".wav",
                                          delete=False)
        tmp.close()
        sf.write(tmp.name, mono, sr)
        # Validate the written file before committing
        sf.info(tmp.name)
        os.replace(tmp.name, cache_path)
        return True
    except Exception:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        return False


def _resample_np(src: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear resampling via ``np.interp`` — no scipy/librosa needed.

    NOTE: linear interpolation without an anti-aliasing lowpass filter may
    cause aliasing on downsampling.  For the MDX-NET model this is
    acceptable because the input is a full-band mix and the model's
    internal STFT provides frequency selectivity.  If scipy is available,
    ``scipy.signal.resample_poly`` would be a higher-quality alternative.
    """
    if src_sr == dst_sr or len(src) == 0:
        return src
    src = np.asarray(src, dtype=np.float32)
    # Anti-aliasing lowpass for downsampling (src_sr/dst_sr > 1.1)
    # Simple half-band FIR via numpy convolution.
    ratio = src_sr / dst_sr
    if ratio > 1.1 and len(src) > 50:
        # Cutoff at 0.9 * dst_sr/2 (Nyquist of destination)
        cutoff = int(round(min(len(src), 512) * 0.9 / ratio))
        if cutoff >= 3:
            kernel = np.hanning(cutoff * 2 - 1).astype(np.float32)
            kernel = kernel / kernel.sum()
            src = np.convolve(src, kernel, mode="same")
    n = max(1, int(round(len(src) * dst_sr / src_sr)))
    x_old = np.arange(len(src), dtype=np.float32)
    x_new = np.linspace(0, len(src) - 1, n, dtype=np.float32)
    return np.interp(x_new, x_old, src).astype(np.float32)


def _try_mdx_separation(
    folder: str, model_path: str | None, log_fn=None
) -> list[str] | None:
    """Attempt MDX-NET vocal separation on the full mix of ``folder``.
    
    Returns a single-element list (path to cached vocal WAV) or ``None``.
    """
    try:
        from .separate import separate_vocals
    except ImportError:
        return None
    mdx_cache = os.path.join(folder, _VOCAL_CACHE_MDX)
    try:
        import numpy as np
        mix_paths = find_song_audio(folder)
        if not mix_paths:
            return None
        if _cache_is_fresh(mdx_cache, mix_paths):
            return [mdx_cache]

        # Log start — visible in terminal (GUI captures process_folder's log_fn
        # for the per-song "vocal source" line after processing).
        song = os.path.basename(folder.strip("/\\")) or folder
        msg = f"  [mdx] separating vocals: {song}...\n"
        if log_fn:
            log_fn(msg, "info")
        else:
            sys.stderr.write(msg)

        if len(mix_paths) == 1:
            mono, file_sr = load_mono(mix_paths[0])
        else:
            mono, file_sr = load_mono_mix(mix_paths)
        if mono is None or not file_sr:
            return None
        # MDX-NET model requires 44100 Hz — resample if needed
        model_sr = 44100
        if file_sr != model_sr:
            mono = _resample_np(mono, file_sr, model_sr)
        stereo = np.stack([mono, mono], axis=0)  # (2, N) — shape expected by separate_vocals
        del mono
        vocals = separate_vocals(stereo, sr=model_sr, model_path=model_path)
        del stereo
        if vocals is None:
            msg = f"  [mdx]  failed (model or onnx issue)\n"
            if log_fn:
                log_fn(msg, "err")
            else:
                sys.stderr.write(msg)
            return None
        mono_vocals = vocals.mean(axis=0, dtype=np.float32)
        # Resample back to the original file rate for caching
        if file_sr != model_sr:
            mono_vocals = _resample_np(mono_vocals, model_sr, file_sr)
        if _atomic_write_cache(mdx_cache, mono_vocals, file_sr):
            size_mb = os.path.getsize(mdx_cache) / 2**20
            from . import separate as _sep_mod
            prov = getattr(_sep_mod, "last_provider", None)
            prov_txt = f" · {prov}" if prov else ""
            msg = f"  [mdx]  done ({size_mb:.1f} MB cached{prov_txt})\n"
            if log_fn:
                log_fn(msg, "ok")
            else:
                sys.stderr.write(msg)
            return [mdx_cache]
    except Exception:
        import traceback
        msg = f"  [mdx]  failed: {traceback.format_exc()}\n"
        if log_fn:
            log_fn(msg, "err")
        else:
            sys.stderr.write(msg)
        return None


def resolve_vocal_audio(
    folder: str,
    model_path: str | None = None,
    allow_separation: bool = True,
    log_fn=None,
) -> list[str] | None:
    """Find or create vocal audio for a song folder.

    Returns a single-element list (path to a vocal audio file) for use with
    ``voice_activity()``, or ``None`` if no vocal audio can be found.

    Priority:
    1. Isolated vocal stems (``Vocal.ogg`` / ``vocals.wav``) → returned directly
    2. A ``.mogg`` → extract vocal channels → cache to ``.downcharter_vocals_mogg.wav``
    3. MDX-NET separation on the full mix (only when ``allow_separation=True``)
       → cache to ``.downcharter_vocals_mdx.wav``
    4. Nothing works → ``None``

    Each source type has its own cache file, invalidated independently when
    the corresponding source file changes.  Cache files can be safely deleted.

    Args:
        folder: Song folder to search for vocal audio.
        model_path: Optional path to the MDX-NET ONNX model.
        allow_separation: If True (default), try MDX-NET separation as last
            resort when no stems or .mogg are available.
        log_fn: Optional callback ``log_fn(msg, tag)`` for GUI logging.
    """
    # Priority 1: already-separated stems
    stems = find_vocal_stems(folder)
    if stems:
        return stems

    # Priority 2: .mogg channel extraction
    if os.path.isdir(folder):
        mogg_cache = os.path.join(folder, _VOCAL_CACHE_MOGG)
        for f in os.listdir(folder):
            if not f.lower().endswith(".mogg"):
                continue
            mogg_path = os.path.join(folder, f)
            if _cache_is_fresh(mogg_cache, [mogg_path]):
                return [mogg_cache]
            try:
                result = load_vocal_from_mogg(mogg_path)
            except Exception:
                result = None
            if result is not None:
                mono, _sr = result
                if _atomic_write_cache(mogg_cache, mono, _sr):
                    return [mogg_cache]
            # Try the next .mogg if the first one failed

    # Priority 3: MDX-NET separation (only when allow_separation=True)
    if allow_separation:
        result = _try_mdx_separation(folder, model_path, log_fn=log_fn)
        if result is not None:
            return result

    return None


def load_vocal_from_mogg(
    mogg_path: str,
    vocal_channels: tuple[int, int] = _MOGG_VOCAL_CHANNELS,
) -> tuple:
    """Extract ONLY the vocal channels from an unencrypted .mogg file.

    Returns ``(mono_float32, sample_rate)`` or ``None`` on failure
    (encrypted mogg, missing channels, or I/O error).

    The standard RB3 12-channel layout places vocals at channels 6-7
    (stereo).  The caller can override ``vocal_channels`` for non-standard
    layouts.  If the file has fewer channels than required, falls back to
    an all-channel mean (so it never silently returns silence)."""
    try:
        with open(mogg_path, "rb") as f:
            header = f.read(8)
            version = struct.unpack("<I", header[:4])[0]
            if version != 0x0A:
                return None  # encrypted — not supported
            offset = struct.unpack("<I", header[4:8])[0]
            f.seek(offset)
            buf = io.BytesIO(f.read())
    except Exception:
        return None
    try:
        data, sr = _read_all_or_blocks(buf)
    except Exception:
        return None
    # Select requested channels, or fall back to all-channel mean.
    mc = data.shape[1]
    lo, hi = vocal_channels
    if hi < mc:
        mono = data[:, lo:hi + 1].mean(axis=1)
    else:
        mono = data.mean(axis=1)
    return mono.astype("float32"), sr


def voice_activity_from_mogg(
    mogg_path: str,
    vocal_channels: tuple[int, int] = _MOGG_VOCAL_CHANNELS,
    hop_s: float = 0.05,
):
    """Voice activity envelope from the vocal channels of a ``.mogg`` file.

    Returns ``(env, hop_s, thr)`` — same three-tuple format as
    :func:`voice_activity` — or ``None`` if the mogg can't be read /
    is encrypted / has no audible content on the vocal channels."""
    result = load_vocal_from_mogg(mogg_path, vocal_channels)
    if result is None:
        return None
    mono, sr = result
    try:
        import numpy as np
        env = rms_envelope(mono, sr, hop_s)
        if len(env) < 4:
            return None
        peak = float(env.max())
        floor = float(np.percentile(env, 20))
        if peak <= 0:
            return None
        thr = floor + 0.08 * (peak - floor)
        return env, hop_s, thr
    except Exception:
        return None


def voice_activity(paths, hop_s: float = 0.05):
    """RMS envelope of the vocal stem(s) plus a 'voice-present' threshold.
    Returns (env, hop_s, thr) or None on failure. The threshold is relative:
    floor (20th pct) + 8% of the dynamic range, so 'voice present' adapts to the
    stem's own noise floor. Used to confirm a singer is actually holding a note.
    Kept low (8%, was 12%) because a SUSTAINED sung vowel modulates in amplitude
    with vibrato — its troughs dip well below a 12% line while the singer is still
    clearly holding the note — and a high threshold clipped those sustains short."""
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
        thr = floor + 0.08 * (peak - floor)
        return env, hop_s, thr
    except Exception:
        return None


def voice_active_spans(va, min_dur: float = 0.3,
                       merge_gap: float = 0.25) -> list[tuple[float, float]]:
    """Continuous singing spans (start_s, end_s), straight from the vocal stem's
    RMS envelope (`va` from `voice_activity`) — independent of any charted note
    or lyric event. Silent gaps < `merge_gap` apart are bridged (so a vibrato
    dip doesn't fragment a held note); runs shorter than `min_dur` are dropped
    as noise."""
    if va is None:
        return []
    env, hop_s, thr = va
    runs: list[list[float]] = []
    for i, v in enumerate(env):
        if v <= thr:
            continue
        t = i * hop_s
        if runs and t - runs[-1][1] <= merge_gap:
            runs[-1][1] = t + hop_s
        else:
            runs.append([t, t + hop_s])
    return [(r[0], r[1]) for r in runs if r[1] - r[0] >= min_dur]


def voice_active_ticks(va, tempo_map, tpb: int, min_dur: float = 0.3,
                       merge_gap: float = 0.25, spacing_s: float = 0.5) -> list[int]:
    """Pseudo-onset ticks marking continuous singing (see `voice_active_spans`) —
    independent of any charted note or lyric event. Fills animation-onset gaps
    where the lyrics track has no event (e.g. an unscripted scream/ad-lib) but
    the singer is audibly still performing; without this, idle-gap detection
    (`venue.build_animations`) reads those stretches as silence and the
    vocalist's body freezes in an idle pose while still singing.

    Emits one tick every `spacing_s` seconds inside each continuous active run."""
    runs = voice_active_spans(va, min_dur, merge_gap)
    out: list[int] = []
    for start_s, end_s in runs:
        t = start_s
        while t < end_s:
            out.append(_ms_to_tick(t * 1000.0, tempo_map, tpb))
            t += spacing_s
    return out


def voice_offset_s(va, start_s: float, ceil_s: float,
                   gap_s: float = 0.25) -> float | None:
    """Second at which the voice falls silent after `start_s` (silence sustained
    for at least `gap_s`), searching up to `ceil_s`. Returns None if the voice
    persists all the way to the ceiling (genuine sustain). `va` is the tuple from
    `voice_activity`.

    `gap_s` is 0.25s (was 0.15) so a single vibrato trough — the brief amplitude
    dip inside a held vowel, ~0.1s at 5-6 Hz — can NOT masquerade as the end of the
    syllable. Real inter-phrase silence runs much longer (>0.4s), so genuine cuts
    still register; only the mid-note vibrato dips are now ignored."""
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


def syllable_gain(va, start_s: float, end_s: float,
                  floor: float = 0.55, ceiling: float = 1.0) -> float:
    """Loudness of a syllable's window [start_s, end_s] as a viseme-weight gain in
    [floor, ceiling]. Uses the vocal-stem RMS envelope from `voice_activity`: the
    louder the singer holds the vowel, the wider the mouth opens. A `ceiling` of
    1.0 keeps the mouth at baseline weight at peak loudness; only quieter syllables
    close down more. Song-relative (peak of the whole stem), never absolute dB.
    Returns 1.0 if no audio (no scaling); callers already clamp the scaled weight
    to [0, 255]."""
    if va is None:
        return 1.0
    try:
        env, hop_s, _thr = va
        a = max(0, int(start_s / hop_s))
        b = max(a + 1, int(end_s / hop_s))
        seg = env[a:b]
        peak = float(env.max())
        if len(seg) == 0 or peak <= 0:
            return 1.0
        amp = float(seg.max()) / peak
        amp = max(0.0, min(1.0, amp))
        return floor + (ceiling - floor) * amp
    except Exception:
        return 1.0


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
             and not any(k in os.path.basename(f).lower() for k in _NON_BAND_STEMS)
             and os.path.basename(f) not in _VOCAL_CACHE_NAMES]
    # Prefer separate .ogg stems (more informative); otherwise the multichannel .mogg.
    if stems:
        return sorted(stems)
    return moggs[:1]


def _find_ffmpeg():
    """Locate ffmpeg binary. Returns path or None."""
    import shutil
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    # Common Windows locations + user's shared build
    for candidate in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser(
            r"~\Downloads\ffmpeg-master-latest-win64-gpl-shared"
            r"\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _read_all_or_blocks(src):
    """Read every frame of `src` (a path or BytesIO) to a 2D float32 array.

    Tries libsndfile first. If that fails (malformed OGG/OPUS pages), tries
    ffmpeg as fallback — ffmpeg handles malformed pages gracefully without
    inserting silence. Only falls back to the old seek-based chunked reading
    (which substitutes silence for bad pages) as a last resort.

    Returns (data2d, sr)."""
    import numpy as np
    import soundfile as sf
    try:
        return sf.read(src, dtype="float32", always_2d=True)
    except Exception:
        pass

    # If src is a file path (not BytesIO), try ffmpeg
    if isinstance(src, str) and os.path.isfile(src):
        ff = _find_ffmpeg()
        if ff:
            try:
                return _decode_with_ffmpeg(src, ff)
            except Exception:
                pass

    # Last resort: seek-based chunked reading (may insert silence for bad pages)
    if hasattr(src, "seek"):
        src.seek(0)
    with sf.SoundFile(src) as f:
        sr, ch, total = f.samplerate, f.channels, len(f)
        chunk = max(1, sr)                    # 1 s
        out, pos, good = [], 0, 0
        while pos < total:
            n = min(chunk, total - pos)
            try:
                f.seek(pos)
                d = f.read(n, dtype="float32", always_2d=True)
                if len(d) < n:                # short read near a bad page
                    d = np.vstack([d, np.zeros((n - len(d), ch), "float32")])
                good += 1
            except Exception:
                d = np.zeros((n, ch), "float32")     # skip bad page, keep timing
            out.append(d)
            pos += n
    if not good:
        raise
    return np.concatenate(out, axis=0), sr


def _decode_with_ffmpeg(src, ffmpeg_path):
    """Decode audio via ffmpeg to WAV in memory, then read with soundfile.

    ffmpeg handles malformed OGG/OPUS pages that trip libsndfile, avoiding
    the silence-gap fallback of _read_all_or_blocks."""
    import subprocess
    import tempfile
    import soundfile as sf
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            [ffmpeg_path, "-y", "-i", src,
             "-f", "wav", "-acodec", "pcm_f32le",
             "-ar", "48000", "-ac", "2", tmp.name],
            capture_output=True, check=True)
        data, sr = sf.read(tmp.name, dtype="float32", always_2d=True)
        return data, sr
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def load_mono(path: str):
    """Decode the audio to mono float32. Handles .mogg (header strip)."""
    if path.lower().endswith(".mogg"):
        with open(path, "rb") as f:
            header = f.read(8)
            version = struct.unpack("<I", header[:4])[0]
            if version != 0x0A:
                raise ValueError(f"encrypted mogg (version {version}) not supported")
            offset = struct.unpack("<I", header[4:8])[0]
            f.seek(offset)
            data, sr = _read_all_or_blocks(io.BytesIO(f.read()))
    else:
        data, sr = _read_all_or_blocks(path)
    return data.mean(axis=1), sr


def load_mono_mix(paths: list[str]):
    """Mix (sum) several stems into a single mono. Stems with different sample
    rates are resampled (linearly) to the sr of the first one. Aligns to the
    shortest stem. A single multichannel .mogg is already summed by load_mono."""
    import numpy as np
    if not paths:
        raise ValueError("no stems")
    base_sr = None
    mix = None
    for p in paths:
        mono, sr = load_mono(p)
        if base_sr is None:
            base_sr = sr
        elif sr != base_sr and len(mono):
            mono = _resample_np(mono, sr, base_sr)
        if mix is None:
            mix = mono.copy()
        else:
            n = min(len(mix), len(mono))
            mix[:n] += mono[:n]
    if mix is None:
        return None, None
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
    # Ensure C-contiguous so stride arithmetic is valid
    if not mono.flags["C_CONTIGUOUS"]:
        mono = np.ascontiguousarray(mono, dtype=np.float32)
    n = 1 + (len(mono) - win) // hop
    strides = (mono.strides[0] * hop, mono.strides[0])
    shape = (n, win)
    frames = np.lib.stride_tricks.as_strided(mono, shape=shape, strides=strides, writeable=False)
    frames = frames * np.hanning(win).astype(np.float32)
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


# Empirical FEEL weights — calibrated against 100 official venue learn songs
# (feel_calibrate.py grid search maximizing 3-way tier agreement).
# loud leads (0.40), brightness inverted and weighted (-0.20) — high intensity
# is DARKER. flux (0.20) captures transients. low/dens/flat at 0.10 each add
# heaviness/hardness/wall. Σ|w| ≈ 1.0 for stable threshold calibration.
_FEEL_W = {"loud": 0.40, "flux": 0.20, "flat": 0.10,
           "low": 0.10, "dens": 0.10, "bright": -0.20}


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
    # COVERAGE GUARD: a malformed .opus/.ogg may only decode partway (libsndfile stops
    # at the first bad page). A half-decoded envelope would score every uncovered
    # section ~0 → wrongly 'calm'. If the audio doesn't cover ≥80% of the song, bail
    # to None so the whole song uses the (consistent) MIDI-only fallback instead.
    song_end_s = tick_to_ms(sections[-1].end, tempo_map, tpb) / 1000.0
    if song_end_s > 0 and len(env) * stft_s < 0.80 * song_end_s:
        return None
    out: list[float] = []
    for s in sections:
        a_s = tick_to_ms(s.start, tempo_map, tpb) / 1000.0
        b_s = tick_to_ms(s.end, tempo_map, tpb) / 1000.0
        j0 = max(0, int(a_s / stft_s))
        j1 = min(len(env), max(j0 + 1, int(b_s / stft_s)))
        out.append(float(np.mean(env[j0:j1])) if j1 > j0 else 0.0)
    return out


def section_energy_subspans(paths, sections, tempo_map, tpb: int,
                            hop_s: float = 0.1, smooth_s: float = 1.5,
                            min_span_s: float = 3.0, heavy_gate: float = 999.0,
                            onsets: list[int] | None = None,
                            subspan_alpha: float = 0.6,
                            calm_thresh: float = 0.495,
                            high_thresh: float = 0.595):
    """Per-section SUB-SPANS of energy tier, following the music WITHIN a section.
    Instead of one mean tier per section (which washes out a chorus that starts calm
    and builds, and collapses a compressed song to all-'mid'), this segments the
    per-frame feel_envelope into ['calm'/'mid'/'high'] runs. Two safeguards keep it
    from flickering: the envelope is smoothed over `smooth_s`, and any run shorter
    than `min_span_s` is merged into its nearest-tier neighbour. The 3.0s floor is the
    official venues' p10 mood-span length (median 16s) — they hold a mood far longer
    than the audio flickers, so anything shorter would over-animate. Because the
    envelope is MAGNITUDE-scaled song-relative (_scale01: p90→1), the song's own
    loudest ~10%+ of frames always reach 'high' — so even a less-dynamic / compressed
    song gets its peaks back here without any structural/kind promotion.

    HEAVINESS GATE: loudness alone can't tell a *heavy* breakdown (headbang) from a
    loud *sung* chorus — both can be equally loud. A 'high' span is demoted to 'mid'
    (→ [play], not [intense]) when its song-relative HEAVINESS — bass energy + spectral
    flatness + darkness — is below `heavy_gate` (0.5 = the song's own midpoint, NOT
    restrictive: only the lighter half of the loud material loses the headbang). The
    heaviest passages (real breakdowns) keep [intense]. NB: the 20 official venues do
    NOT gate intense this way (they headbang loud choruses freely); this is a more
    contained aesthetic, by design, for sung/emotional material.
    Returns list[list[(start_tick, end_tick, tier)]] (one list per section, covering
    the whole section, runs contiguous), or None on audio failure."""
    if not sections:
        return None
    if isinstance(paths, str):
        paths = [paths]
    try:
        import numpy as np
        mono, sr = load_mono_mix(paths)
        feats, stft_s = _feel_frames(mono, sr)
    except Exception:
        return None
    if feats is None:
        return None
    import numpy as np
    # COVERAGE GUARD (see section_energy_scores): a half-decoded malformed file would
    # leave the song's tail without sub-spans → bail so it stays MIDI-only & consistent.
    song_end_s = tick_to_ms(sections[-1].end, tempo_map, tpb) / 1000.0
    if song_end_s > 0 and len(feats["loud"]) * stft_s < 0.80 * song_end_s:
        return None
    # Composite FEEL envelope (same blend as feel_envelope) + a song-relative
    # HEAVINESS envelope (low-end + flatness + darkness), both from one STFT pass.
    env = np.zeros(len(feats["loud"]), dtype="float64")
    for k, wt in _FEEL_W.items():
        r = _scale01(feats[k])
        env += abs(wt) * (1.0 - r if wt < 0 else r)
    heavy = (_scale01(feats["low"]) + _scale01(feats["flat"])
             + (1.0 - _scale01(feats["bright"]))) / 3.0
    w = max(1, int(round(smooth_s / stft_s)))
    if w > 1:
        heavy = np.convolve(heavy, np.ones(w) / w, mode="same")
    if w > 1:
        env = np.convolve(env, np.ones(w) / w, mode="same")
    # ── MIDI onset density blend ─────────────────────────────────────────
    # When onset ticks are provided, compute per-frame onset density
    # (onsets in a 2-beat window around each frame) and blend with the
    # audio envelope: combined = alpha * env + (1-alpha) * midi_density.
    # This lets MIDI fill in the mid zone that audio can't separate.
    if onsets and len(onsets) > 1:
        # Convert onset ticks to seconds for alignment with frames
        onset_s = np.array([tick_to_ms(t, tempo_map, tpb) / 1000.0
                            for t in onsets])
        n_frames = len(env)
        # 2-beat window in seconds (average ~0.5s at 120 BPM)
        avg_beat_s = (tick_to_ms(sections[-1].end, tempo_map, tpb)
                      - tick_to_ms(sections[0].start, tempo_map, tpb)) / 1000.0
        avg_beats_total = max(1, (sections[-1].end - sections[0].start) / tpb)
        beat_s = avg_beat_s / avg_beats_total if avg_beats_total > 0 else 0.5
        win_s = 2.0 * beat_s
        # Vectorized: compute density for each frame using searchsorted on arrays
        frame_times = np.arange(n_frames) * stft_s
        # For each frame, count onsets in [t - win_s/2, t + win_s/2]
        lo = frame_times - win_s / 2
        hi = frame_times + win_s / 2
        # searchsorted gives indices; difference counts onsets in window
        idx_lo = np.searchsorted(onset_s, lo, side="left")
        idx_hi = np.searchsorted(onset_s, hi, side="right")
        midi_env = (idx_hi - idx_lo).astype("float64")
        # Normalize song-relative (rank01)
        order = np.argsort(midi_env)
        rank = np.empty_like(order, dtype="float64")
        rank[order] = np.linspace(0, 1, len(order))
        midi_env = rank
        # Smooth MIDI density to match audio envelope smoothness — raw
        # onset counts oscillate frame-to-frame (0–N per 2-beat window),
        # creating micro-runs that the min_span merge absorbs entirely.
        w_midi = max(1, int(round(smooth_s * 2 / stft_s)))
        if w_midi > 1:
            midi_env = np.convolve(midi_env, np.ones(w_midi) / w_midi, mode="same")
        # Blend
        env = subspan_alpha * env + (1.0 - subspan_alpha) * midi_env
    tier = np.where(env < calm_thresh, 0, np.where(env < high_thresh, 1, 2)).astype(int)
    min_frames = max(1, int(round(min_span_s / stft_s)))
    names = ("calm", "mid", "high")
    out: list[list[tuple[int, int, str]]] = []
    for s in sections:
        a_s = tick_to_ms(s.start, tempo_map, tpb) / 1000.0
        b_s = tick_to_ms(s.end, tempo_map, tpb) / 1000.0
        j0 = max(0, int(a_s / stft_s))
        j1 = min(len(env), max(j0 + 1, int(b_s / stft_s)))
        seg = tier[j0:j1]
        if len(seg) == 0:
            out.append([(s.start, s.end, "calm")])
            continue
        # contiguous runs of equal tier: [start_frame, end_frame, tier]
        runs: list[list[int]] = []
        cs = 0
        for k in range(1, len(seg) + 1):
            if k == len(seg) or seg[k] != seg[cs]:
                runs.append([cs, k, int(seg[cs])])
                cs = k
        # merge any run shorter than min_frames into the closest-tier neighbour
        while len(runs) > 1:
            i = min(range(len(runs)), key=lambda r: runs[r][1] - runs[r][0])
            if runs[i][1] - runs[i][0] >= min_frames:
                break
            left = runs[i - 1] if i > 0 else None
            right = runs[i + 1] if i < len(runs) - 1 else None
            if left is None:
                tgt = i + 1
            elif right is None:
                tgt = i - 1
            else:
                dl, dr = abs(left[2] - runs[i][2]), abs(right[2] - runs[i][2])
                if dl != dr:
                    tgt = i - 1 if dl < dr else i + 1
                else:  # tie → merge into the larger neighbour
                    tgt = i - 1 if (left[1] - left[0]) >= (right[1] - right[0]) else i + 1
            lo = min(runs[i][0], runs[tgt][0])
            hi = max(runs[i][1], runs[tgt][1])
            runs[tgt] = [lo, hi, runs[tgt][2]]
            del runs[i]
        # Heaviness gate: a 'high' run that isn't heavy enough (sung loud chorus, not a
        # breakdown) drops to 'mid' → [play] instead of [intense].
        if heavy_gate < 1.0:
            for r in runs:
                if r[2] == 2:
                    hh = float(np.mean(heavy[j0 + r[0]:j0 + r[1]]))
                    if hh < heavy_gate:
                        r[2] = 1
        # collapse any adjacent equal-tier runs left after merging/gating
        collapsed: list[list[int]] = []
        for r in runs:
            if collapsed and collapsed[-1][2] == r[2]:
                collapsed[-1][1] = r[1]
            else:
                collapsed.append(r)
        spans: list[tuple[int, int, str]] = []
        for ri, (f0, f1, tr) in enumerate(collapsed):
            start_tick = s.start if ri == 0 else _ms_to_tick((j0 + f0) * stft_s * 1000.0, tempo_map, tpb)
            end_tick = s.end if ri == len(collapsed) - 1 else _ms_to_tick((j0 + f1) * stft_s * 1000.0, tempo_map, tpb)
            spans.append((start_tick, end_tick, names[tr]))
        out.append(spans)
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
    return ["calm" if v < 0.40 else ("mid" if v < 0.45 else "high") for v in scores]


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

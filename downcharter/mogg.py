"""mogg.py — build a native Rock Band .mogg from separate YARG/CH stems.

A YARG/Clone-Hero song folder ships its audio as separate stems
(drums.ogg, bass.ogg, guitar.ogg, vocals.ogg, song.ogg, …) OR as a single
full-mix file (song.mp3, song.wav, …). Any of .ogg/.opus/.wav/.flac/.mp3 work
(libsndfile decodes them all without ffmpeg); a pre-built .mogg is reused
verbatim upstream so it never reaches this builder. Rock Band wants a
single multichannel Ogg Vorbis with an 8-byte header — the ".mogg". This module
decodes the stems, interleaves them into one N-channel OGG (libsndfile/soundfile
handle multichannel Vorbis), and prepends the unencrypted mogg header:

    u32le version = 0x0A      (0x0A = unencrypted; RPCS3 plays these directly)
    u32le ogg_offset = 8      (the OGG stream starts right after the header)
    <ogg vorbis bytes …>

The returned channel layout (instrument → channel indices) is what the dta
generator needs to write the (tracks ...) / (pans ...) / (vols ...) lists.
"""
from __future__ import annotations
import os
import io
import struct

# Map filename keywords → Rock Band instrument track names, in RB channel order.
# The first matching keyword wins; anything unmatched becomes part of the
# backing "song" track. Order here is the order channels are laid into the mogg.
_STEM_ORDER = [
    ("drum",   ("drums", "drum")),
    ("bass",   ("bass",)),
    ("guitar", ("guitar", "rhythm")),
    ("keys",   ("keys", "keyboard")),
    ("vocals", ("vocals", "vocal", "vox", "harm")),
    ("song",   ("song", "backing", "rhythm_track", "crowd")),
]

# Audio stem extensions we can decode (libsndfile reads all of these without
# ffmpeg). A single full-mix file (e.g. song.mp3) is enough — stems are optional.
_AUDIO_EXT = (".ogg", ".opus", ".wav", ".flac", ".mp3")
_NON_AUDIO = (".mogg",)

# Rock Band 3's audio engine assumes 44.1 kHz moggs; any other rate crashes the
# game at song load. We always encode the mogg at this rate.
_RB3_SAMPLE_RATE = 44100


def _classify(filename: str) -> str | None:
    low = os.path.basename(filename).lower()
    for track, keys in _STEM_ORDER:
        if any(k in low for k in keys):
            return track
    return "song"  # unknown stem → fold into backing track


def collect_stems(folder: str) -> list[tuple[str, str]]:
    """Return [(track_name, path)] for every audio stem under `folder`, ordered by
    the RB track order in _STEM_ORDER. Excludes any existing .mogg.

    Walks recursively so audio living in a subfolder is found — matching the
    recursive source discovery in ps3build (_find_one). The .mogg-output dir is
    skipped so a freshly written package never feeds back in as a stem source."""
    if not os.path.isdir(folder):
        return []
    found: list[tuple[str, str]] = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            low = f.lower()
            if low.endswith(_AUDIO_EXT) and not low.endswith(_NON_AUDIO):
                found.append((_classify(f), os.path.join(root, f)))
    order = {name: i for i, (name, _) in enumerate(_STEM_ORDER)}
    found.sort(key=lambda tp: (order.get(tp[0], 99), os.path.basename(tp[1]).lower()))
    return found


def _decode(path: str):
    """Decode a stem to (data2d float32 [frames, ch], samplerate).

    Uses audio._read_all_or_blocks, the SEEK-based reader: some encoders emit
    .opus/.ogg that libsndfile opens fine but trips on ('Supported file format
    but file is malformed') during a sequential read at a few bad pages. The
    fallback reads 1 s at a time, substituting silence for any bad page, so the
    whole song is recovered with its timeline intact (a plain sf.read would stop
    dead at the first bad page)."""
    from . import audio as _audio
    return _audio._read_all_or_blocks(path)


def _resample(data, sr_from: int, sr_to: int):
    import numpy as np
    if sr_from == sr_to or len(data) == 0:
        return data
    n = int(round(len(data) * sr_to / sr_from))
    src_idx = np.arange(len(data))
    dst_idx = np.linspace(0, len(data), n, endpoint=False)
    out = np.empty((n, data.shape[1]), dtype="float32")
    for c in range(data.shape[1]):
        out[:, c] = np.interp(dst_idx, src_idx, data[:, c]).astype("float32")
    return out


def build_mogg_from_stems(folder: str, out_path: str, log_fn=None,
                          pad_seconds: float = 0.0):
    """Build `out_path` (.mogg) from the stems in `folder`.

    Returns a list describing the channel layout:
        [(track_name, [channel_index, ...]), ...]
    in the order the channels appear in the mogg. Raises if there are no stems.

    `pad_seconds` prepends that much silence to every channel so the audio stays
    in sync with a MIDI that was lead-in-padded for RB3 (convert.pad_start /
    Onyx magmaPad). 0.0 = no padding (the normal case)."""
    import numpy as np
    import soundfile as sf

    log = log_fn or (lambda *a, **k: None)
    stems = collect_stems(folder)
    if not stems:
        raise FileNotFoundError(
            "no audio (.ogg/.opus/.wav/.flac/.mp3 or a .mogg) found to build a .mogg")

    decoded = []
    src_sr = None
    failed: list[str] = []
    for track, path in stems:
        try:
            data, sr = _decode(path)
        except Exception as e:
            # A single corrupt/truncated stem must not abort the whole song:
            # skip it with a warning and build from whatever decodes.
            failed.append(os.path.basename(path))
            log(f"    ⚠ skipped unreadable audio "
                f"{os.path.basename(path)} ({e})\n", "warn")
            continue
        if src_sr is None:
            src_sr = sr
        decoded.append((track, data, sr))
    if not decoded:
        listed = ", ".join(failed) if failed else "none"
        raise ValueError(f"no audio could be decoded (malformed/unreadable: {listed})")

    # Rock Band 3 requires 44.1 kHz moggs — its audio engine assumes 44100 and
    # crashes at song LOAD on any other rate (e.g. a 48 kHz source stem). Always
    # resample the output to 44100, regardless of the source rate. (A 44.1 kHz
    # source is a no-op.)
    base_sr = _RB3_SAMPLE_RATE
    if src_sr != base_sr:
        log(f"    ◇ mogg: resampling {src_sr} Hz → {base_sr} Hz (RB3 requires "
            f"44.1 kHz)\n", "info")
    # Resample everything to RB3's 44.1 kHz, pad to the longest length.
    resampled = [(t, _resample(d, sr, base_sr)) for (t, d, sr) in decoded]
    # Lead-in silence (kept in lockstep with a lead-in-padded MIDI).
    pad_frames = int(round(max(0.0, pad_seconds) * base_sr))
    if pad_frames:
        resampled = [(t, np.concatenate(
            [np.zeros((pad_frames, d.shape[1] if d.ndim == 2 else 1), "float32"),
             d if d.ndim == 2 else d.reshape(-1, 1)], axis=0)) for (t, d) in resampled]
        log(f"    ◇ mogg: prepended {pad_seconds:.3f}s lead-in silence\n", "info")
    max_len = max((len(d) for _, d in resampled), default=0)

    layout: list[tuple[str, list[int]]] = []
    cols: list = []  # individual channel columns (1D arrays length max_len)
    ch = 0
    for track, data in resampled:
        nch = data.shape[1] if data.ndim == 2 else 1
        idxs = []
        for c in range(nch):
            col = np.zeros(max_len, dtype="float32")
            col[: len(data)] = data[:, c]
            cols.append(col)
            idxs.append(ch)
            ch += 1
        layout.append((track, idxs))

    interleaved = np.stack(cols, axis=1) if cols else np.zeros((0, 0), "float32")
    total_ch = interleaved.shape[1] if interleaved.ndim == 2 else 0

    # Encode to a multichannel OGG. libsndfile's Vorbis encoder crashes (stack
    # overflow) when handed one giant array, so write in 1-second blocks. Encode
    # to a temp .ogg, then prepend the 8-byte mogg header into out_path.
    tmp_ogg = out_path + ".tmp.ogg"
    chunk = max(1, base_sr)
    with sf.SoundFile(tmp_ogg, mode="w", samplerate=base_sr, channels=total_ch,
                      format="OGG", subtype="VORBIS") as f:
        pos = 0
        while pos < max_len:
            n = min(chunk, max_len - pos)
            f.write(interleaved[pos:pos + n])
            pos += n

    header = struct.pack("<II", 0x0A, 8)  # version 0x0A (unencrypted), offset 8
    with open(tmp_ogg, "rb") as src, open(out_path, "wb") as dst:
        dst.write(header)
        while True:
            block = src.read(1 << 20)
            if not block:
                break
            dst.write(block)
    try:
        os.remove(tmp_ogg)
    except OSError:
        pass

    log(f"    ◇ mogg: built {total_ch} channels @ {base_sr} Hz "
        f"({len(decoded)} stem(s))\n", "info")
    return layout

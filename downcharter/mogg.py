"""mogg.py — build a native Rock Band .mogg from separate YARG/CH stems.

A YARG/Clone-Hero song folder ships its audio as separate stems
(drums.ogg, bass.ogg, guitar.ogg, vocals.ogg, song.ogg, …). Rock Band wants a
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

_NON_AUDIO = (".mogg",)


def _classify(filename: str) -> str | None:
    low = os.path.basename(filename).lower()
    for track, keys in _STEM_ORDER:
        if any(k in low for k in keys):
            return track
    return "song"  # unknown stem → fold into backing track


def collect_stems(folder: str) -> list[tuple[str, str]]:
    """Return [(track_name, path)] for every audio stem in `folder`, ordered by
    the RB track order in _STEM_ORDER. Excludes any existing .mogg."""
    if not os.path.isdir(folder):
        return []
    found: list[tuple[str, str]] = []
    for f in sorted(os.listdir(folder)):
        low = f.lower()
        if low.endswith((".ogg", ".opus", ".wav", ".flac")) and not low.endswith(_NON_AUDIO):
            found.append((_classify(f), os.path.join(folder, f)))
    order = {name: i for i, (name, _) in enumerate(_STEM_ORDER)}
    found.sort(key=lambda tp: (order.get(tp[0], 99), os.path.basename(tp[1]).lower()))
    return found


def _decode(path: str):
    """Decode a stem to (data2d float32 [frames, ch], samplerate)."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data, sr


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


def build_mogg_from_stems(folder: str, out_path: str, log_fn=None):
    """Build `out_path` (.mogg) from the stems in `folder`.

    Returns a list describing the channel layout:
        [(track_name, [channel_index, ...]), ...]
    in the order the channels appear in the mogg. Raises if there are no stems.
    """
    import numpy as np
    import soundfile as sf

    log = log_fn or (lambda *a, **k: None)
    stems = collect_stems(folder)
    if not stems:
        raise FileNotFoundError("no audio stems (.ogg/.wav) found to build a .mogg")

    decoded = []
    base_sr = None
    for track, path in stems:
        data, sr = _decode(path)
        if base_sr is None:
            base_sr = sr
        decoded.append((track, data, sr))

    # Resample everything to the first stem's rate, pad to the longest length.
    resampled = [(t, _resample(d, sr, base_sr)) for (t, d, sr) in decoded]
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
        f"({len(stems)} stems)\n", "info")
    return layout

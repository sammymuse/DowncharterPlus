"""milo.py — native .milo builder (Phase A of the native-package pipeline).

Today Downcharter only emits a notes.mid and relies on Onyx to compile the
LIPSYNC1 track into a .milo and pack a CON/PS3 package. The problem: we don't
control that step, so we can't *guarantee* our lipsync reaches the game (two
packs built around our Phase-2 lipsync change came out byte-identical, i.e.
Onyx re-packed an old milo). This module builds the milo ourselves so the
lipsync we generate is guaranteed to be in the file the game loads.

The milo carries a single `song.lipsync` (CharLipSync) inside a `MagmaLipsync1`
ObjectDir, wrapped in an uncompressed MILO_A header. Everything here was
reverse-engineered and byte-verified against the real Onyx-built milos we have
locally (`midis/Converted/.../gen/*.milo_ps3`):

  * Milo header (Compression.hs `addMiloHeader`, uncompressed MILO_A):
      u32le 0xCABEDEAF · u32le 0x810 header size · u32le blockCount ·
      u32le maxBlockSize · u32le each block size · zero-pad to 0x810 · body.
    We emit a single block = the whole body (the multi-block split only matters
    above maxBlockSize and isn't needed here).
  * Milo dir body (Dir.hs `magmaMiloDir`, MagmaLipsync1): version 28, type
    "ObjectDir", name "lipsync", one entry ("CharLipSync","song.lipsync"), the
    7 hardcoded dummy matrices, then the files section
    `<unknown bytes> ADDEADDE <song.lipsync> ADDEADDE`.
    The dir header up to the first ADDEADDE barrier is constant for every
    single-LIPSYNC1 milo, so it's embedded verbatim (`_DIR_PREFIX`) — confirmed
    identical across our local samples and platform-independent (a PS3
    .milo_ps3 and an Xbox .milo_xbox share the same body; only the outer
    CON/STFS wrapper differs).
  * CharLipSync `song.lipsync`: same big-endian format as `lipsync._serialize`,
    BUT the viseme table is DYNAMIC — Onyx lists only the visemes actually used,
    alphabetically sorted, and the per-frame events index into that table (our
    local Elegy milo had 28 visemes incl. `Though`; Cryogen had 26 without it).
    `build_song_lipsync` reproduces that exactly via the dense-state path in
    `lipsync.frames_from_spans`.
"""
from __future__ import annotations
import math
import struct

from . import lipsync as _lip

# Files-section barrier between objects in a milo dir body.
_BARRIER = b"\xad\xde\xad\xde"

# MILO_A uncompressed header constants.
_MILO_MAGIC = 0xCABEDEAF
_MILO_HEADER_SIZE = 0x810

# The MagmaLipsync1 ObjectDir header tail (U3, U4, empty subname, 7 dummy
# matrices, trailing bookkeeping). Extracted byte-by-byte from the Onyx-built
# milo blob at offset 71 (after the first entry). Constant regardless of how
# many CharLipSync entries are listed — only U1 + U2 + the entry list change.
#
# We assemble the full header for 1 entry, then slice off the tail (offset 71+)
# so the hex string is written once and sliced, avoiding subtle string-pasting
# length bugs.
_FULL_DIR_SINGLE = bytes.fromhex(
    "0000001c000000094f626a656374446972000000076c697073796e63000000040000001500000001"
    "0000000b436861724c697053796e630000000c736f6e672e6c697073796e630000001b0000000200"
    "0000000000000000000000000000073f3504f3bf3504f3000000003f13cd3a3f13cd3abf13cd3a3e"
    "d105eb3ed105eb3f5105ebc3ddb3d7c3ddb3d743ddb3d700000000bf800000000000003f80000000"
    "0000000000000000000000000000003f800000c44000000000000000000000000000003f80000000"
    "000000bf800000000000000000000000000000000000003f8000004440000000000000000000003f"
    "80000000000000000000000000000000000000bf800000000000003f800000000000000000000000"
    "000000444000003f800000000000000000000000000000000000003f80000000000000bf80000000"
    "0000000000000000000000c44000003f8000000000000000000000000000003f8000000000000000"
    "000000000000003f80000000000000c440000000000000bf800000000000000000000000000000bf"
    "8000000000000000000000000000003f800000000000004440000000000000000000000100000000"
    "00000000000000000000000000000000000000000000"
)
_DIR_TAIL = _FULL_DIR_SINGLE[71:]  # U3 + U4 + subname + matrices + trailing


def _build_dir_prefix(entry_names: list[str]) -> bytes:
    """Build the ObjectDir header bytes for N `entry_names`.

    Entry name convention (RB3 multi-entry milos, verified against 100+ official
    samples): single-entry = ``["song.lipsync"]``; multi-entry =
    ``["part2.lipsync", ..., "song.lipsync"]`` — part entries FIRST, song LAST.

    U1/U2 scale with entry count (verified: N=1→U1=4/U2=21, N=2→U1=6/U2=35,
    N=3→U1=8/U2=49):
        U1 = 2 + 2*N
        U2 = 7 + 14*N

    Structure:
        u32 version = 28
        u32 type_name_len + "ObjectDir"
        u32 name_len + "lipsync"
        u32 U1 = 2 + 2*N
        u32 U2 = 7 + 14*N
        u32 entry_count (= N)
        for each entry: u32 type_len + "CharLipSync" + u32 name_len + name
        _DIR_TAIL (= U3, U4, subname, 7 matrices, trailing)
    """
    n = len(entry_names)
    u1 = 2 + 2 * n
    u2 = 7 + 14 * n
    out = bytearray()
    # ObjectDir header fields are BIG-ENDIAN (>I) — verified against 100+ official
    # milos (PS3 and Xbox). The outer MILO_A wrapper uses native-endian for the
    # header fields, but the dir body is pure big-endian.
    out += struct.pack(">I", 28)               # version
    out += struct.pack(">I", 9)                # type name length
    out += b"ObjectDir"                        # type name
    out += struct.pack(">I", 7)               # name length
    out += b"lipsync"                          # name
    out += struct.pack(">I", u1)               # U1
    out += struct.pack(">I", u2)               # U2
    out += struct.pack(">I", n)                # entry_count

    for name in entry_names:
        out += struct.pack(">I", 11)           # type length
        out += b"CharLipSync"                  # type
        out += struct.pack(">I", len(name))    # name length
        out += name.encode("ascii")            # name

    out += _DIR_TAIL
    return bytes(out)


# Singleton dir prefix for the common single-entry case (N=1).
# Uses the same _build_dir_prefix path as multi-entry, so behaviour is
# consistent; byte-identical to the original hardcoded _FULL_DIR_SINGLE.
_DIR_PREFIX = _build_dir_prefix(["song.lipsync"])


def add_milo_header(body: bytes) -> bytes:
    """Wrap a milo dir body in the uncompressed MILO_A header (single block)."""
    out = bytearray()
    out += struct.pack("<I", _MILO_MAGIC)
    out += struct.pack("<I", _MILO_HEADER_SIZE)
    out += struct.pack("<I", 1)              # blockCount = 1 (whole body)
    out += struct.pack("<I", len(body))      # maxBlockSize = the one block
    out += struct.pack("<I", len(body))      # block 0 size
    out += b"\x00" * (_MILO_HEADER_SIZE - len(out))   # zero-pad to 0x810
    return bytes(out) + body


def build_milo(lipsync_or_list: bytes | list[bytes]) -> bytes:
    """Assemble a complete .milo (== .milo_ps3 == .milo_xbox body).

    Accepts a single ``bytes`` (single ``song.lipsync`` entry) or a ``list[bytes]``
    (multi-entry for PART VOCALS + HARM2 + HARM3: entries are
    ``["part2.lipsync", "song.lipsync"]`` for 2 tracks or
    ``["part2.lipsync", "part3.lipsync", "song.lipsync"]`` for 3 — part entries FIRST,
    song LAST, matching the official RB3 milo convention verified against 100+ real
    files).

    The body is platform-independent (PS3 and Xbox share the same milo body)."""
    if isinstance(lipsync_or_list, bytes):
        lipsync_list: list[bytes] = [lipsync_or_list]
    else:
        lipsync_list = lipsync_or_list

    n = len(lipsync_list)
    if n == 1:
        names = ["song.lipsync"]
    else:
        # part entries first, song.lipsync last — verified against official milos
        names = [f"part{k}.lipsync" for k in range(2, n + 1)]
        names.append("song.lipsync")

    body = bytearray()
    body += _build_dir_prefix(names)
    # N entries → N+2 barriers total (leading + N intermediate + trailing).
    # Barrier layout: [leading, blob0_end, blob1_end, ..., blob{N-1}_end, trailing]
    # so blob i is body[barriers[i+1]+4 : barriers[i+2]].
    # The leading barrier (barriers[0]) marks end-of-header; the formula
    # i1=barriers[index*2+1], i2=barriers[index*2+2] then gives the correct range.
    body += _BARRIER  # leading (barriers[0])
    for lb in lipsync_list:
        body += lb + _BARRIER  # blob + intermediate barrier
    return add_milo_header(bytes(body))


def _serialize_lipsync(frames: dict[int, dict], n_frames: int) -> bytes:
    """CharLipSync bytes with a DYNAMIC viseme table (matching Onyx).

    Only the visemes actually used (weight>0 in some frame) are listed, sorted
    alphabetically — which is exactly the order Onyx emits (e.g. Bump_hi,
    Bump_lo, Cage_hi, …). Per-frame events are delta-encoded (each frame lists
    only the visemes that CHANGED; the game holds the previous value) and index
    into this table. Big-endian; same envelope as `lipsync._serialize`."""
    used: set[str] = set()
    for fr in frames.values():
        for name, w in fr.items():
            if w > 0 and name in _lip._VIDX:
                used.add(name)
    visemes = sorted(used)
    vidx = {n: i for i, n in enumerate(visemes)}

    body = bytearray()
    prev: dict[int, int] = {}
    for fr in range(n_frames):
        cur: dict[int, int] = {}
        for name, w in frames.get(fr, {}).items():
            idx = vidx.get(name)
            if idx is not None and w > 0:
                cur[idx] = w & 0xFF
        events: list[tuple[int, int]] = []
        for idx, w in cur.items():
            if prev.get(idx, 0) != w:
                events.append((idx, w))
        for idx in prev:
            if idx not in cur:
                events.append((idx, 0))
        events.sort()
        body.append(len(events) & 0xFF)
        for idx, w in events:
            body += struct.pack("BB", idx & 0xFF, w & 0xFF)
        prev = cur

    out = bytearray()
    out += struct.pack(">I", 1)              # version (RB3/Magma v2)
    out += struct.pack(">I", 2)              # subversion
    out += struct.pack(">I", 0)              # DTA import string (empty)
    out += struct.pack("B", 0)               # dtb flag
    out += struct.pack(">I", 0)              # skip
    out += struct.pack(">I", len(visemes))   # viseme count
    for name in visemes:
        b = name.encode("ascii")
        out += struct.pack(">I", len(b)) + b
    out += struct.pack(">I", n_frames)       # keyframe count
    out += struct.pack(">I", len(body))      # following size
    out += body
    out += struct.pack(">I", 0)              # trailing
    return bytes(out)


def build_song_lipsync(spans, song_len_s: float, lang: str = "en",
                        phrase_ends: list[float] | None = None,
                        vocal_notes: list[tuple[float, int]] | None = None,
                        facial_seed: int | None = None) -> bytes:
    """CharLipSync bytes for the milo, from the same audio-guided syllable spans
    (`lip_spans` = [(start_s, end_s, text, gain)]) used for the LIPSYNC1 MIDI track.

    Uses the sparse keyframes path (`lipsync_keyframes_from_spans`) which correctly
    sustains vowels at full weight via 'hold' graph tokens — unlike the dense
    `frames_from_spans` path whose 3-point control (attack → release → end) decays
    the mouth throughout the syllable. The sparse keyframes are converted to dense
    30fps frames via `_facial_frames_from_keyframes` (graph-aware interpolation:
    'hold' = flat plateau, 'linear'/'ease' = smooth transitions) before serialising.

    When phrase_ends/vocal_notes are provided, facial animation keyframes
    (Blink, Squint, Eyebrows) are also embedded in the milo so they reach
    the game — not just the MIDI LIPSYNC1 track."""
    keyframes = _lip.lipsync_keyframes_from_spans(
        spans,
        phrase_ends=phrase_ends,
        song_len_s=song_len_s,
        vocal_notes=vocal_notes,
        facial_seed=facial_seed,
    )
    n_frames = max(1, int(math.ceil(song_len_s * _lip.FPS)) + 1)
    frames = _lip._facial_frames_from_keyframes(keyframes, n_frames)
    return _serialize_lipsync(frames, n_frames)


def build_milo_from_spans(spans, song_len_s: float, lang: str = "en",
                           phrase_ends: list[float] | None = None,
                           vocal_notes: list[tuple[float, int]] | None = None,
                           facial_seed: int | None = None) -> bytes:
    """Convenience: spans → complete .milo ready to write to disk."""
    return build_milo(build_song_lipsync(
        spans, song_len_s, lang,
        phrase_ends=phrase_ends,
        vocal_notes=vocal_notes,
        facial_seed=facial_seed,
    ))


def build_multi_lipsync(spans_list: list, song_len_s: float, lang: str = "en",
                         phrase_ends: list[float] | None = None,
                         vocal_notes: list[tuple[float, int]] | None = None,
                         facial_seed: int | None = None) -> list[bytes]:
    """Build N lipsync byte blobs from N span lists (lead + HARM1 + HARM2 + HARM3).

    Each list may be empty (a track with no lyrics produces no lipsync entry).
    Returns entries in the order expected by build_milo for the RB3 multi-entry
    convention: harmonies FIRST, lead LAST (so build_milo names them
    part2.lipsync / part3.lipsync / song.lipsync)."""
    parts: list[bytes] = []
    lead: bytes | None = None
    for i, spans in enumerate(spans_list):
        if spans:
            # Vary facial_seed per entry so each vocalist blinks/pairs differently
            seed = (facial_seed + i) if facial_seed is not None else None
            blob = build_song_lipsync(
                spans, song_len_s, lang,
                phrase_ends=phrase_ends,
                vocal_notes=vocal_notes,
                facial_seed=seed,
            )
            if i == 0:
                lead = blob   # first vocal track = lead / PART VOCALS
            else:
                parts.append(blob)
    if lead is not None:
        parts.append(lead)   # lead goes LAST → named "song.lipsync" by build_milo
    return parts


# ───────────────────────────── validation reader ─────────────────────────────
def parse_song_lipsync(milo_bytes: bytes, index: int = 0) -> dict:
    """Parse the Nth lipsync entry from a .milo back to
    {visemes, n_frames, lipsync_bytes, frames}.

    For a multi-entry milo the entries are (in order):
    ``part2.lipsync``, ``part3.lipsync``, …, ``song.lipsync`` (RB3 convention).
    ``index=0`` reads the first entry; ``index=1`` the second, etc.
    Used for round-trip validation gate."""
    body = milo_bytes[_MILO_HEADER_SIZE:]
    # Collect all ADDEADDE barrier positions
    barriers: list[int] = []
    pos = 0
    while True:
        p = body.find(_BARRIER, pos)
        if p == -1:
            break
        barriers.append(p)
        pos = p + 4
    if index + 1 >= len(barriers):
        raise ValueError(
            f"lipsync entry {index} not found "
            f"(milo has {len(barriers) - 1} entries)")
    i1 = barriers[index]
    i2 = barriers[index + 1]
    lip = body[i1 + 4:i2]
    o = 0

    def u32() -> int:
        nonlocal o
        v = struct.unpack(">I", lip[o:o + 4])[0]
        o += 4
        return v

    u32()                       # version
    u32()                       # subversion
    dta_len = u32()             # DTA import string length…
    o += dta_len                # …skip its bytes (NB: not `o += u32()` — the
    o += 1                      # nonlocal advance would be clobbered). dtb flag
    u32()                       # skip
    nvis = u32()
    visemes = []
    for _ in range(nvis):
        n = u32()
        visemes.append(lip[o:o + n].decode("latin1"))
        o += n
    n_frames = u32()
    u32()                       # following size
    frames: dict[int, dict] = {}
    state: dict[int, int] = {}
    for fr in range(n_frames):
        cnt = lip[o]
        o += 1
        for _ in range(cnt):
            idx, w = lip[o], lip[o + 1]
            o += 2
            if w:
                state[idx] = w
            else:
                state.pop(idx, None)
        if state:
            frames[fr] = {visemes[i]: w for i, w in state.items()}
    return {"visemes": visemes, "n_frames": n_frames,
            "lipsync_bytes": bytes(lip), "frames": frames}

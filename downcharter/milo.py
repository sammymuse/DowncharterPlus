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

# The MagmaLipsync1 ObjectDir header, from milo version (28) up to — but not
# including — the first ADDEADDE barrier of the files section. Constant for
# every single-`song.lipsync` milo; copied verbatim (byte-verified) from the
# Onyx-built milos. Contains: ver 28, "ObjectDir", name "lipsync", U1=4,
# U2=0x15, entry ("CharLipSync","song.lipsync"), U3=0x1B, U4=2, empty subname,
# the 7 dummy matrices, and the trailing dir bookkeeping + 13 unknown bytes.
_DIR_PREFIX = bytes.fromhex(
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


def build_milo(lipsync_bytes: bytes) -> bytes:
    """Assemble a complete .milo (== .milo_ps3 == .milo_xbox body) carrying one
    `song.lipsync` (CharLipSync). The body is platform-independent."""
    body = _DIR_PREFIX + _BARRIER + lipsync_bytes + _BARRIER
    return add_milo_header(body)


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

    Reuses `lipsync.frames_from_spans` (the dense 30 fps state path) so the milo's
    inherently dense per-frame viseme states come straight from what we already
    compute — no new lipsync logic.

    When phrase_ends/vocal_notes are provided, facial animation keyframes
    (Blink, Squint, Eyebrows) are also embedded in the milo so they reach
    the game — not just the MIDI LIPSYNC1 track."""
    frames, n_frames = _lip.frames_from_spans(
        spans, song_len_s, lang,
        phrase_ends=phrase_ends,
        vocal_notes=vocal_notes,
        facial_seed=facial_seed,
    )
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


# ───────────────────────────── validation reader ─────────────────────────────
def parse_song_lipsync(milo_bytes: bytes) -> dict:
    """Parse a .milo back to {visemes, n_frames, lipsync_bytes, frames} — a small
    reader for the round-trip validation gate (diff our output vs. a real milo)."""
    body = milo_bytes[_MILO_HEADER_SIZE:]
    i1 = body.find(_BARRIER)
    i2 = body.find(_BARRIER, i1 + 4)
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

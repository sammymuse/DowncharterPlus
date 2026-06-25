"""ps3build.py — assemble the native RB3 PS3 outputs (Phase B).

Goal of the whole effort: stop depending on Onyx to compile our LIPSYNC into a
.milo and pack the song. Onyx re-packed stale milos (two packs built around our
Phase-2 lipsync change came out byte-identical), so our lipsync wasn't reaching
the game. By building the milo ourselves (downcharter/milo.py) we guarantee the
lipsync we generate is in the file the game loads.

This module wires that milo into the file system. It is being rolled out in two
steps, smallest-blast-radius first:

  1. `write_milo_sidecar` (LIVE): during normal processing, write a
     `<song>.milo_ps3` next to the processed notes.mid, built from the same
     audio-guided syllable spans as the LIPSYNC1 track. A PS3 .milo_ps3 and an
     Xbox .milo_xbox have identical bodies, so the same bytes serve both — we
     also drop a `.milo_xbox` copy. This lets us A/B test the milo IN-GAME by
     swapping it into a known-good RPCS3 pack's `gen/` folder, BEFORE investing
     in from-scratch dta/mogg/folder generation. (Decided test-first: confirm
     the milo carries our lipsync in-game first.)

  2. `build_ps3_song` (TODO, pending step 1's in-game validation + the
     unencrypted .mid / .mogg-version questions): lay out the full unencrypted
     PS3 song folder — `<ID>/songs/<id>/<id>.mid` (plain), `<id>.mogg` (reuse the
     source mogg verbatim, per the decision to keep it as-is), `gen/<id>.milo_ps3`,
     and a generated `songs/songs.dta`. The Xbox-360 CON/STFS packer is a later
     follow-up (downcharter/stfs.py) reusing the same milo + dta + mogg.
"""
from __future__ import annotations
import os

from . import milo as _milo


def write_milo_sidecar(dst_mid_path: str, spans, song_len_s: float,
                       lang: str = "en") -> list[str]:
    """Build the .milo from the lipsync spans and write it next to the MIDI.

    `spans` = [(start_s, end_s, text, gain)] (the same audio-guided syllable
    spans that drive the LIPSYNC1 track). Writes `<base>.milo_ps3` and an
    identical `<base>.milo_xbox` (the body is platform-independent; only the
    outer CON/STFS wrapper differs). Returns the paths written (empty if there's
    nothing to build). Never raises into the caller's pipeline."""
    if not spans or song_len_s <= 0:
        return []
    milo_bytes = _milo.build_milo_from_spans(spans, song_len_s, lang)
    base = os.path.splitext(dst_mid_path)[0]
    written: list[str] = []
    for ext in (".milo_ps3", ".milo_xbox"):
        path = base + ext
        with open(path, "wb") as f:
            f.write(milo_bytes)
        written.append(path)
    return written

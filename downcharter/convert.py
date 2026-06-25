"""convert.py — the native package-conversion pipeline (separate from the normal
per-folder MIDI processing).

This drives the "Convert" GUI tab: it takes an already-built RB3 song folder and
re-assembles it natively as a clean RPCS3 PS3 song folder, with our own milo
(downcharter/milo.py) so the lipsync we generate is guaranteed in the file the
game loads — no Onyx, no stale milos. Xbox CON (.con) and YARG .sng are planned
follow-ups; this first cut handles only the PS3 folder, per the rollout decision.

Bass-pedal variants (RB3 doesn't read YARG-style Expert+ 2x automatically):
  * "2x": force every Expert+ double-kick (note 95) down to a normal kick
    (note 96) so the doubles ALWAYS play, regardless of the in-game 2x toggle.
  * "1x": remove the Expert+ double-kicks entirely, leaving a chart that's
    playable with a single pedal.
  * "both": emit two folders, one of each.

The note-95 markers come from Downcharter's Expert+ pass (processor
`_apply_expert_plus`), so the source notes.mid is expected to already carry them
where fast double-bass was detected.
"""
from __future__ import annotations
import mido

from .constants import DRUM_KICK_EXPERT, DRUM_KICK_2X


def _is_drums_track(track: mido.MidiTrack) -> bool:
    return "DRUM" in (track.name or "").strip().upper()


def apply_pedal_variant(mid: mido.MidiFile, mode: str) -> tuple[mido.MidiFile, dict]:
    """Return a NEW MidiFile with PART DRUMS kicks adjusted for `mode`.

    mode == "2x": every note-95 (Expert+ 2x-kick) becomes note-96 (normal kick),
                  so the doubles play with no in-game toggle. (No-op for songs
                  with no 95 markers — they already play as-is.)
    mode == "1x": every note-95 note_on/note_off is dropped, removing the
                  double-kicks so the chart is single-pedal playable.

    Returns (new_mid, stats) where stats = {"converted": n, "removed": n}.
    Never mutates the input file.
    """
    if mode not in ("1x", "2x"):
        raise ValueError(f"pedal mode must be '1x' or '2x', got {mode!r}")

    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    converted = 0
    removed = 0

    for track in mid.tracks:
        new_tr = mido.MidiTrack()
        new_tr.name = track.name
        if not _is_drums_track(track):
            for msg in track:
                new_tr.append(msg.copy())
            out.tracks.append(new_tr)
            continue

        # Drums track: walk messages, carrying delta time across dropped events
        # (1x mode) so timing of surviving messages is preserved.
        pending_delta = 0
        for msg in track:
            delta = msg.time + pending_delta
            is_note = msg.type in ("note_on", "note_off")
            if is_note and msg.note == DRUM_KICK_2X:
                if mode == "2x":
                    new_tr.append(msg.copy(note=DRUM_KICK_EXPERT, time=delta))
                    pending_delta = 0
                    if msg.type == "note_on" and msg.velocity > 0:
                        converted += 1
                else:  # 1x → drop this note, push its delta onto the next msg
                    pending_delta = delta
                    if msg.type == "note_on" and msg.velocity > 0:
                        removed += 1
            else:
                new_tr.append(msg.copy(time=delta))
                pending_delta = 0
        out.tracks.append(new_tr)

    return out, {"converted": converted, "removed": removed}

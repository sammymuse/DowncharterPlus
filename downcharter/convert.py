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


# Open (no-fret) strum markers. YARG/Clone-Hero charts use the note one below the
# green gem of each difficulty (the "ENHANCED_OPENS" extension). Rock Band 3 has
# no open-strum lane and silently IGNORES these notes, so the chart plays with
# gaps. We remap each open onto its difficulty's GREEN gem so it stays playable.
#   Easy 59→60, Medium 71→72, Hard 83→84, Expert 95→96
OPEN_TO_GREEN = {59: 60, 71: 72, 83: 84, 95: 96}

# Tracks that use the 5-fret open convention (NOT drums — there note 95 is the
# 2x-kick, handled separately by apply_pedal_variant).
_FRET_TRACK_KEYS = ("GUITAR", "BASS", "RHYTHM")


def _is_fret_track(track: mido.MidiTrack) -> bool:
    nm = (track.name or "").strip().upper()
    return "DRUM" not in nm and any(k in nm for k in _FRET_TRACK_KEYS)


def convert_open_notes(mid: mido.MidiFile) -> tuple[mido.MidiFile, dict]:
    """Return a NEW MidiFile with open-strum notes on 5-fret tracks remapped to
    the green gem of their difficulty (RB3 ignores open notes otherwise).

    Returns (new_mid, {"converted": n}). Never mutates the input.
    """
    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    converted = 0
    for track in mid.tracks:
        new_tr = mido.MidiTrack()
        new_tr.name = track.name
        fret = _is_fret_track(track)
        for msg in track:
            if (fret and msg.type in ("note_on", "note_off")
                    and msg.note in OPEN_TO_GREEN):
                new_tr.append(msg.copy(note=OPEN_TO_GREEN[msg.note]))
                if msg.type == "note_on" and msg.velocity > 0:
                    converted += 1
            else:
                new_tr.append(msg.copy())
        out.tracks.append(new_tr)
    return out, {"converted": converted}


# ── drum limb animations ───────────────────────────────────────────────────────
# RB3 animates the drummer from dedicated animation notes (24-51 on PART DRUMS).
# YARG auto-animates from the chart, but RB3 needs them authored, so a YARG chart
# with no animation notes leaves the drummer idle. We synthesise them from the
# Expert gems using the classic rock sticking: right hand leads the cymbals/ride,
# left hand plays the snare; pro-tom markers (110/111/112) turn the
# yellow/blue/green lanes into toms.
_ANIM_KICK      = 24
_ANIM_SNARE_LH  = 26   # hard
_ANIM_HIHAT_RH  = 31
_ANIM_RIDE_RH   = 42
_ANIM_CRASH_RH  = 36   # crash 1, right hand, hard
_ANIM_TOM1_RH   = 47   # yellow tom
_ANIM_TOM2_RH   = 49   # blue tom
_ANIM_FLOOR_RH  = 51   # green/floor tom
_TOM_MARKERS    = (110, 111, 112)


def generate_drum_animations(mid: mido.MidiFile) -> tuple[mido.MidiFile, dict]:
    """Return a NEW MidiFile with drummer limb-animation notes (24-51) synthesised
    on PART DRUMS from its Expert gems. No-op for a drums track that is already
    animated. Returns (new_mid, {"added": n}). Never mutates the input."""
    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    tpb = mid.ticks_per_beat
    anim_len = max(1, tpb // 8)
    added = 0

    for track in mid.tracks:
        if not _is_drums_track(track):
            new_tr = mido.MidiTrack()
            for m in track:
                new_tr.append(m.copy())
            out.tracks.append(new_tr)
            continue

        # Absolute-tick view of the track.
        abs_msgs, t = [], 0
        already_animated = False
        for m in track:
            t += m.time
            abs_msgs.append((t, m))
            if m.type == "note_on" and 24 <= m.note <= 51:
                already_animated = True

        if already_animated:
            new_tr = mido.MidiTrack()
            for m in track:
                new_tr.append(m.copy())
            out.tracks.append(new_tr)
            continue

        # Pro-tom marker spans → know whether a lane is a tom at a given tick.
        spans = {n: [] for n in _TOM_MARKERS}
        opening: dict[int, int] = {}
        for tick, m in abs_msgs:
            if m.type in ("note_on", "note_off") and m.note in _TOM_MARKERS:
                if m.type == "note_on" and m.velocity > 0:
                    opening[m.note] = tick
                elif m.note in opening:
                    spans[m.note].append((opening.pop(m.note), tick))

        def _is_tom(marker: int, tick: int) -> bool:
            return any(a <= tick < b for a, b in spans[marker])

        # Map each Expert gem to one animation note.
        anim: list[tuple[int, int]] = []
        for tick, m in abs_msgs:
            if not (m.type == "note_on" and m.velocity > 0):
                continue
            n = m.note
            if n in (DRUM_KICK_EXPERT, DRUM_KICK_2X):
                note = _ANIM_KICK
            elif n == 97:                                   # red → snare
                note = _ANIM_SNARE_LH
            elif n == 98:                                   # yellow → tom1 / hihat
                note = _ANIM_TOM1_RH if _is_tom(110, tick) else _ANIM_HIHAT_RH
            elif n == 99:                                   # blue → tom2 / ride
                note = _ANIM_TOM2_RH if _is_tom(111, tick) else _ANIM_RIDE_RH
            elif n in (100, 101):                           # green → floor / crash
                note = _ANIM_FLOOR_RH if _is_tom(112, tick) else _ANIM_CRASH_RH
            else:
                continue
            anim.append((tick, note))

        # Merge gems + animation pairs, drop the old end-of-track, re-time.
        merged = [(tick, m) for tick, m in abs_msgs if m.type != "end_of_track"]
        for tick, note in anim:
            merged.append((tick, mido.Message("note_on", note=note, velocity=96, time=0)))
            merged.append((tick + anim_len, mido.Message("note_off", note=note, velocity=0, time=0)))
            added += 1
        merged.sort(key=lambda tm: tm[0])

        new_tr = mido.MidiTrack()
        last = 0
        for tick, m in merged:
            new_tr.append(m.copy(time=tick - last))
            last = tick
        new_tr.append(mido.MetaMessage("end_of_track", time=0))
        out.tracks.append(new_tr)

    return out, {"added": added}


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

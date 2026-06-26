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
from .midi_utils import build_tempo_map, tick_to_ms


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
# YARG auto-animates from the chart, but RB3 needs them authored, so a chart with
# no animation notes leaves the drummer idle. We synthesise them from the Expert
# Pro-drum gems with proper LEFT/RIGHT-HAND sticking — a faithful port of Onyx's
# `autoDrumAnimation`/`autoSticking` (mtolly/onyx, Onyx/MIDI/Track/Drums.hs):
#
#   * Each gem maps to an "anim pad" (snare, hihat, ride, crashes, toms), with
#     pro-tom markers (110/111/112) turning the yellow/blue/green lanes into toms.
#   * Simultaneous hits split between the hands (lower kit position → LH, higher
#     → RH); special chords (two cymbals → both crashes; red+yellow-tom → snare
#     flam) match Onyx.
#   * Single hits within 0.25 s of each other form a "phrase" whose sticking
#     alternates hands, leading with the hand that keeps the drummer from crossing
#     over on the next hit (double-strokes are delayed to the latest moment).
#
# The full RB3 animation note map (per Onyx `parseDrumAnimation`):
#   24 kick(RF) · 26/27 snare LH/RH · 30/31 hihat LH/RH · 34/36 crash1 LH/RH ·
#   38 crash2 RH · 42/43 ride RH/LH · 44 crash2 LH · 46/47 tom1 LH/RH ·
#   48/49 tom2 LH/RH · 50/51 floortom LH/RH
_TOM_MARKERS = (110, 111, 112)

# Anim pads, ordered left→right across the kit (used for hand assignment and the
# "normal direction" of motion). Mirrors Onyx's `AnimPad` Ord.
_SNARE, _HIHAT, _CRASH1, _TOM1, _TOM2, _FLOOR, _CRASH2, _RIDE = range(8)
_LH, _RH = "LH", "RH"

# (pad, hand) → RB3 animation note. Hard hits only (we don't author ghost notes).
_ANIM_NOTE = {
    (_SNARE, _LH): 26,  (_SNARE, _RH): 27,
    (_HIHAT, _LH): 30,  (_HIHAT, _RH): 31,
    (_CRASH1, _LH): 34, (_CRASH1, _RH): 36,
    (_TOM1, _LH): 46,   (_TOM1, _RH): 47,
    (_TOM2, _LH): 48,   (_TOM2, _RH): 49,
    (_FLOOR, _LH): 50,  (_FLOOR, _RH): 51,
    (_CRASH2, _LH): 44, (_CRASH2, _RH): 38,
    (_RIDE, _LH): 43,   (_RIDE, _RH): 42,
}
_ANIM_KICK = 24


def _flip(hand: str) -> str:
    return _RH if hand == _LH else _LH


def _normal_direction(x: int, y: int):
    """Natural hand when moving from pad x to pad y (None = no preference).
    Hihat↔snare and snare↔tom1 are centred, so neither implies a direction."""
    if (x, y) in ((_HIHAT, _SNARE), (_SNARE, _HIHAT), (_SNARE, _TOM1), (_TOM1, _SNARE)):
        return None
    if x < y:
        return _RH
    if x > y:
        return _LH
    return None


def _auto_sticking(pads: list[int]) -> list[str]:
    """Assign LH/RH to a phrase of single hits (Onyx `autoSticking`)."""
    out: list[str] = []
    prev = None                       # None | (hand, pad)
    for i, x in enumerate(pads):
        rest = pads[i + 1:]
        if prev is None:
            # Look ahead: first non-None direction decides the starting hand so the
            # run lands correctly; flip if an even number of "free" moves precede it.
            dirs = [_normal_direction(a, b) for a, b in zip(pads[i:], rest)]
            n, h = 0, None
            for d in dirs:
                if d is None:
                    n += 1
                else:
                    h = d
                    break
            if h is None:
                hand = _RH
            else:
                hand = _flip(h) if n % 2 == 0 else h
        else:
            prev_hand, prev_pad = prev
            if x == prev_pad:
                # Same pad: keep the hand (double stroke) only if that sets us up to
                # NOT cross over on the next hit; otherwise alternate.
                if rest and _flip(prev_hand) == _normal_direction(x, rest[0]):
                    hand = prev_hand
                else:
                    hand = _flip(prev_hand)
            else:
                hand = _flip(prev_hand)   # moving pads always switches hands
        out.append(hand)
        prev = (hand, x)
    return out


def generate_drum_animations(mid: mido.MidiFile) -> tuple[mido.MidiFile, dict]:
    """Return a NEW MidiFile with drummer limb-animation notes (24-51) synthesised
    on PART DRUMS from its Expert gems, with left/right-hand sticking. No-op for a
    drums track that is already animated. Returns (new_mid, {"added": n}). Never
    mutates the input."""
    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    tpb = mid.ticks_per_beat
    anim_len = max(1, tpb // 8)
    tempo_map = build_tempo_map(mid)
    close_ms = 250.0                  # Onyx closeTime = 0.25 s
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

        # Collect Expert gems per tick (one set of lanes per onset).
        gems_at: dict[int, set] = {}
        for tick, m in abs_msgs:
            if not (m.type == "note_on" and m.velocity > 0):
                continue
            n = m.note
            if n in (DRUM_KICK_EXPERT, DRUM_KICK_2X):
                lane = "kick"
            elif n == 97:
                lane = "red"
            elif n == 98:
                lane = "yellow_tom" if _is_tom(110, tick) else "yellow_cym"
            elif n == 99:
                lane = "blue_tom" if _is_tom(111, tick) else "blue_cym"
            elif n in (100, 101):
                lane = "green_tom" if _is_tom(112, tick) else "green_cym"
            else:
                continue
            gems_at.setdefault(tick, set()).add(lane)

        # Per-onset → list of anim pads (Onyx `autoDrumAnimation`). Kicks are
        # emitted directly as note 24; everything else feeds the sticking pass.
        _LANE_PAD = {
            "red": _SNARE, "yellow_cym": _HIHAT, "blue_cym": _RIDE,
            "green_cym": _CRASH2, "yellow_tom": _TOM1, "blue_tom": _TOM2,
            "green_tom": _FLOOR,
        }
        kicks: list[int] = []
        events: list[tuple] = []      # ("pair", tick, lo, hi) | ("single", tick, pad)
        for tick in sorted(gems_at):
            lanes = gems_at[tick]
            if "kick" in lanes:
                kicks.append(tick)
            # Special chords (match Onyx ordering).
            if {"yellow_cym", "green_cym"} <= lanes \
                    or {"blue_cym", "green_cym"} <= lanes:
                pads = [_CRASH1, _CRASH2]
            elif {"red", "yellow_tom"} <= lanes:
                pads = [_SNARE, _SNARE]
            else:
                pads = sorted(_LANE_PAD[l] for l in lanes if l in _LANE_PAD)
            if not pads:
                continue
            if len(pads) >= 2:
                events.append(("pair", tick, min(pads), max(pads)))
            else:
                events.append(("single", tick, pads[0]))

        # Walk the event stream; flush single-hit phrases through _auto_sticking.
        anim: list[tuple[int, int]] = []   # (tick, note)

        def _emit(tick: int, pad: int, hand: str) -> None:
            anim.append((tick, _ANIM_NOTE[(pad, hand)]))

        buffer: list[tuple[int, int]] = []   # (tick, pad)

        def _flush() -> None:
            if not buffer:
                return
            hands = _auto_sticking([p for _, p in buffer])
            for (btick, pad), hand in zip(buffer, hands):
                _emit(btick, pad, hand)
            buffer.clear()

        prev_tick = None
        for ev in events:
            if ev[0] == "pair":
                _flush()
                _, tick, lo, hi = ev
                _emit(tick, lo, _LH)
                _emit(tick, hi, _RH)
                prev_tick = tick
            else:
                _, tick, pad = ev
                if buffer and prev_tick is not None and \
                        (tick_to_ms(tick, tempo_map, tpb)
                         - tick_to_ms(prev_tick, tempo_map, tpb)) <= close_ms:
                    buffer.append((tick, pad))
                else:
                    _flush()
                    buffer.append((tick, pad))
                prev_tick = tick
        _flush()

        for tick in kicks:
            anim.append((tick, _ANIM_KICK))

        # Merge gems + animation note pairs, drop the old end-of-track, re-time.
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


def count_double_kicks(mid: mido.MidiFile) -> int:
    """Number of Expert+ 2x-kick markers (note-95 note_ons) on the drums track(s).

    Zero means there is nothing for the "2x" variant to convert: the chart is
    already single-pedal, so a 2x build would be byte-for-byte the 1x build.
    Callers use this to decide whether the "2x" name/label is warranted."""
    n = 0
    for track in mid.tracks:
        if not _is_drums_track(track):
            continue
        for msg in track:
            if (msg.type == "note_on" and msg.velocity > 0
                    and msg.note == DRUM_KICK_2X):
                n += 1
    return n

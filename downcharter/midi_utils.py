"""
midi_utils.py — Tempo and MIDI event helpers
"""
from __future__ import annotations
import mido
from dataclasses import dataclass, field
from typing import Iterator

# ── Charset for MIDI text events (lyrics/sections) ────────────────────────────
# mido writes/reads meta text as latin-1 by default → BLOWS UP on lyrics with
# characters outside latin-1 (e.g. Cyrillic in UTF-8 .charts). We force UTF-8 on
# write (the modern standard that CH/YARG/RB read) and UTF-8-with-latin-1-fallback
# on read (doesn't regress old latin-1 MIDIs). Global, idempotent patch.
def _install_utf8_text_codec() -> None:
    from mido.midifiles import meta as _meta
    if getattr(_meta, "_dc_utf8_patched", False):
        return

    def _encode(string):
        return list(bytearray(string.encode("utf-8")))

    def _decode(data):
        b = bytes(bytearray(data))
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return b.decode("latin-1", errors="replace")

    _meta.encode_string = _encode
    _meta.decode_string = _decode
    _meta._dc_utf8_patched = True

_install_utf8_text_codec()

DEFAULT_TEMPO = 500_000   # 120 BPM em µs/beat


# ── Tempo map ─────────────────────────────────────────────────────────────────

def build_time_sig_map(mid: mido.MidiFile) -> list[tuple[int, int, int]]:
    """
    Returns a list of (abs_tick, numerator, denominator) for time signatures.
    Defaults to 4/4 if none exists.
    """
    changes: dict[int, tuple[int, int]] = {}
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == "time_signature":
                changes[t] = (msg.numerator, msg.denominator)
    result = [(t, n, d) for t, (n, d) in sorted(changes.items())]
    if not result or result[0][0] > 0:
        result.insert(0, (0, 4, 4))
    return result


def measure_ticks_at(abs_tick: int, time_sig_map: list, tpb: int) -> int:
    """Measure length in ticks at the given position."""
    num, den = 4, 4
    for t, n, d in time_sig_map:
        if t <= abs_tick:
            num, den = n, d
        else:
            break
    return int(tpb * num * 4 / den)


def build_tempo_map(mid: mido.MidiFile) -> list[tuple[int, int]]:
    """
    Returns a sorted list of (abs_tick, tempo_us) from all tracks.
    Guarantees it starts at tick=0.
    """
    changes: dict[int, int] = {}
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == "set_tempo":
                changes[t] = msg.tempo
    result = sorted(changes.items())
    if not result or result[0][0] > 0:
        result.insert(0, (0, DEFAULT_TEMPO))
    return result


def tick_to_ms(abs_tick: int, tempo_map: list[tuple[int, int]], tpb: int) -> float:
    """Convert absolute tick → milliseconds."""
    ms = 0.0
    prev_t, prev_u = 0, DEFAULT_TEMPO
    for mt, mu in tempo_map:
        if mt >= abs_tick:
            break
        ms += (min(mt, abs_tick) - prev_t) / tpb * (prev_u / 1000.0)
        prev_t, prev_u = mt, mu
    ms += (abs_tick - prev_t) / tpb * (prev_u / 1000.0)
    return ms


def ms_to_ticks(ms: float, abs_tick_ref: int,
                tempo_map: list[tuple[int, int]], tpb: int) -> int:
    """
    Converts a duration in ms starting at abs_tick_ref into ticks.
    Used to compute sustain lengths.
    """
    # Find the tempo in effect at abs_tick_ref
    tempo = DEFAULT_TEMPO
    for mt, mu in tempo_map:
        if mt <= abs_tick_ref:
            tempo = mu
        else:
            break
    ticks_per_ms = tpb / (tempo / 1000.0)
    return int(ms * ticks_per_ms)


def ms_to_abs_tick(ms_target: float, tempo_map: list[tuple[int, int]], tpb: int) -> int:
    """Inverse of tick_to_ms: absolute milliseconds → absolute tick."""
    ms = 0.0
    prev_t, prev_u = 0, DEFAULT_TEMPO
    for mt, mu in tempo_map:
        seg_ms = (mt - prev_t) / tpb * (prev_u / 1000.0)
        if ms + seg_ms >= ms_target:
            break
        ms += seg_ms
        prev_t, prev_u = mt, mu
    rem_ticks = (ms_target - ms) / (prev_u / 1000.0) * tpb
    return int(round(prev_t + rem_ticks))


def bpm_at(abs_tick: int, tempo_map: list[tuple[int, int]]) -> float:
    """BPM in effect at a given tick."""
    tempo = DEFAULT_TEMPO
    for mt, mu in tempo_map:
        if mt <= abs_tick:
            tempo = mu
        else:
            break
    return 60_000_000 / tempo


# ── Absolute events ───────────────────────────────────────────────────────────

@dataclass
class AbsEvent:
    """MIDI message with absolute time (ticks)."""
    abs_tick: int
    msg: mido.Message

    def copy(self, **kwargs) -> "AbsEvent":
        return AbsEvent(
            abs_tick=kwargs.get("abs_tick", self.abs_tick),
            msg=kwargs.get("msg", self.msg.copy()),
        )


def to_abs(track: mido.MidiTrack) -> list[AbsEvent]:
    """Convert a MIDI track into a list of AbsEvent."""
    t = 0
    result = []
    for msg in track:
        t += msg.time
        result.append(AbsEvent(abs_tick=t, msg=msg))
    return result


def to_track(events: list[AbsEvent]) -> mido.MidiTrack:
    """Convert a list of AbsEvent into a MidiTrack with correct delta times."""
    track = mido.MidiTrack()
    prev = 0
    for ev in sorted(events, key=lambda e: e.abs_tick):
        track.append(ev.msg.copy(time=ev.abs_tick - prev))
        prev = ev.abs_tick
    return track


# ── Note pairing: group note_on + note_off ───────────────────────────────────

@dataclass
class Note:
    """A MIDI note with start, end and duration in ticks."""
    note:     int
    velocity: int
    start:    int   # abs_tick of the note_on
    end:      int   # abs_tick of the note_off  (exclusive)
    channel:  int = 0

    @property
    def duration(self) -> int:
        return self.end - self.start

    def with_note(self, new_note: int) -> "Note":
        return Note(new_note, self.velocity, self.start, self.end, self.channel)

    def with_end(self, new_end: int) -> "Note":
        return Note(self.note, self.velocity, self.start, new_end, self.channel)


def pair_notes(events: list[AbsEvent]) -> tuple[list[Note], list[AbsEvent]]:
    """
    Splits events into:
      - notes: list of Note (note_on paired with note_off)
      - others: non-note events (meta, CC, sysex, set_tempo, etc.)
    """
    notes: list[Note] = []
    others: list[AbsEvent] = []
    # Stack of open note_ons: note → list of (abs_tick, velocity, channel)
    open_ons: dict[int, list[tuple[int, int, int]]] = {}

    for ev in events:
        msg = ev.msg
        if msg.type == "note_on" and msg.velocity > 0:
            open_ons.setdefault(msg.note, []).append(
                (ev.abs_tick, msg.velocity, msg.channel)
            )
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            stack = open_ons.get(msg.note, [])
            if stack:
                start, vel, ch = stack.pop(0)
                notes.append(Note(msg.note, vel, start, ev.abs_tick, ch))
            # else: note_off with no note_on — ignore
        else:
            others.append(ev)

    # note_ons with no note_off: close at the last tick + 1
    for note_val, stack in open_ons.items():
        for start, vel, ch in stack:
            last = max((e.abs_tick for e in events), default=start)
            notes.append(Note(note_val, vel, start, last + 1, ch))

    notes.sort(key=lambda n: (n.start, n.note))
    return notes, others


def notes_to_events(notes: list[Note]) -> list[AbsEvent]:
    """Convert a list of Note into AbsEvents (note_on + note_off)."""
    events = []
    for n in notes:
        events.append(AbsEvent(n.start,
                               mido.Message("note_on",  note=n.note,
                                            velocity=n.velocity, channel=n.channel, time=0)))
        events.append(AbsEvent(n.end,
                               mido.Message("note_off", note=n.note,
                                            velocity=0, channel=n.channel, time=0)))
    return events

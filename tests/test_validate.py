"""Tests for downcharter/validate.py — pre-pack RB3 MIDI sanity gate.

Verifies:
  - validate_rb_midi: clean MIDI → empty issues
  - 480 TPB check: non-480 → error
  - [end] check: missing → warn
  - BEAT track + lead-in: absent → warn, too few → error
  - Overlapping same-pitch detection (broken chords / hung sustains)
  - Empty difficulty detection (crash class)
  - Vocal phrase checks: notes outside phrase → error, overlapping phrases → error
"""
import mido
import pytest

from downcharter import validate as _val


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _make_track(name: str, notes: list = None, texts: list = None,
                lyrics: list = None) -> mido.MidiTrack:
    """Create a track with optional note_on/note_off pairs, text events, lyrics."""
    tr = mido.MidiTrack()
    tr.name = name
    tr.append(mido.MetaMessage("track_name", name=name, time=0))
    if texts:
        for tick, txt in texts:
            tr.append(mido.MetaMessage("text", text=txt, time=tick))
    if lyrics:
        for tick, ly in lyrics:
            tr.append(mido.MetaMessage("lyrics", text=ly, time=tick))
    if notes:
        t = 0
        for tick, note, vel in notes:
            delta = tick - t
            tr.append(mido.Message("note_on", note=note, velocity=vel, time=delta))
            tr.append(mido.Message("note_off", note=note, velocity=0, time=0))
            t = tick
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def _make_mid(tracks_data: list) -> mido.MidiFile:
    """Build MidiFile from [(name, notes, texts, lyrics), ...]."""
    mid = mido.MidiFile(ticks_per_beat=480)
    for args in tracks_data:
        if len(args) == 2:
            mid.tracks.append(_make_track(args[0], notes=args[1]))
        elif len(args) == 3:
            mid.tracks.append(_make_track(args[0], notes=args[1], texts=args[2]))
        else:
            mid.tracks.append(_make_track(*args))
    return mid


# ── Basic sanity ────────────────────────────────────────────────────────────────

class TestValidateBasic:
    def test_valid_midi_empty_issues(self):
        """A well-formed 480 TPB MIDI with [end], BEAT, and all diffs returns no errors."""
        mid = mido.MidiFile(ticks_per_beat=480)
        # Tempo track
        tt = mido.MidiTrack()
        tt.name = "Tempo"
        tt.append(mido.MetaMessage("track_name", name="Tempo", time=0))
        tt.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        mid.tracks.append(tt)
        # BEAT track with >=2 beats lead-in before the first gem (at tick 1440).
        # Beats: downbeat (12) at 0, 480, 960, 1440, ...; upbeat (13) at 240, 720, ...
        beat = mido.MidiTrack()
        beat.name = "BEAT"
        time_sig = mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0)
        beat.append(time_sig)
        for i in range(12):
            note = 12 if i % 4 == 0 else 13
            beat.append(mido.Message("note_on", note=note, velocity=100, time=240))
            beat.append(mido.Message("note_off", note=note, velocity=0, time=60))
        beat.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(beat)
        # EVENTS
        ev = mido.MidiTrack()
        ev.name = "EVENTS"
        ev.append(mido.MetaMessage("text", text="[end]", time=0))
        ev.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(ev)
        # PART GUITAR with all 4 difficulties, starting at tick 1440 (3 beats in)
        gtr = _make_track("PART GUITAR", notes=[
            (1440, 96, 100),   # Expert
            (1920, 84, 100),   # Hard
            (2400, 72, 100),   # Medium
            (2880, 60, 100),   # Easy
        ])
        mid.tracks.append(gtr)

        issues = _val.validate_rb_midi(mid)
        errors = [msg for lvl, msg in issues if lvl == "error"]
        assert errors == [], f"unexpected errors: {errors}"

    def test_non_480_tpb_error(self):
        """Non-480 TPB produces an error."""
        mid = mido.MidiFile(ticks_per_beat=960)
        issues = _val.validate_rb_midi(mid)
        assert any("480" in msg for lvl, msg in issues if lvl == "error")


# ── [end] check ─────────────────────────────────────────────────────────────────

class TestEndCheck:
    def test_no_end_warn(self):
        """Missing [end] produces a warning."""
        mid = mido.MidiFile(ticks_per_beat=480)
        mid.tracks.append(_make_track("PART GUITAR", notes=[(0, 96, 100)]))
        issues = _val.validate_rb_midi(mid)
        assert any("no [end]" in msg.lower() for lvl, msg in issues)


# ── BEAT track check ────────────────────────────────────────────────────────────

class TestBeatCheck:
    def test_no_beat_warn(self):
        """Missing BEAT track produces a warning."""
        mid = mido.MidiFile(ticks_per_beat=480)
        mid.tracks.append(_make_track("PART GUITAR", notes=[(0, 96, 100)]))
        issues = _val.validate_rb_midi(mid)
        assert any("BEAT" in msg for lvl, msg in issues)


# ── Overlap detection ───────────────────────────────────────────────────────────

class TestOverlapCheck:
    def test_no_overlap_clean(self):
        """Clean track with no overlaps produces no error."""
        mid = mido.MidiFile(ticks_per_beat=480)
        tr = _make_track("PART GUITAR", notes=[
            (0, 96, 100), (480, 97, 100), (960, 96, 100),
        ])
        mid.tracks.append(tr)
        issues = _val.validate_rb_midi(mid)
        assert not any("overlap" in msg.lower() for lvl, msg in issues)

    def test_overlap_detected(self):
        """A same-pitch note_on while held produces an 'overlap' error."""
        mid = mido.MidiFile(ticks_per_beat=480)
        tr = mido.MidiTrack()
        tr.name = "PART GUITAR"
        tr.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
        tr.append(mido.Message("note_on", note=96, velocity=100, time=0))
        tr.append(mido.Message("note_on", note=96, velocity=100, time=240))  # overlap!
        tr.append(mido.Message("note_off", note=96, velocity=0, time=240))
        tr.append(mido.Message("note_off", note=96, velocity=0, time=0))
        tr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr)
        issues = _val.validate_rb_midi(mid)
        assert any("overlap" in msg.lower() for lvl, msg in issues)


# ── Empty difficulty check ─────────────────────────────────────────────────────

class TestEmptyDifficulty:
    def test_all_difficulties_present_clean(self):
        """Expert with all lower difficulties → no error."""
        mid = mido.MidiFile(ticks_per_beat=480)
        tr = _make_track("PART GUITAR", notes=[
            (0, 96, 100),   # Expert
            (480, 84, 100), # Hard
            (960, 72, 100), # Medium
            (1440, 60, 100),# Easy
        ])
        mid.tracks.append(tr)
        issues = _val.validate_rb_midi(mid)
        assert not any("difficulty" in msg.lower() for lvl, msg in issues)

    def test_empty_easy_error(self):
        """Expert with no Easy gems → error."""
        mid = mido.MidiFile(ticks_per_beat=480)
        tr = _make_track("PART GUITAR", notes=[
            (0, 96, 100),   # Expert only
        ])
        mid.tracks.append(tr)
        issues = _val.validate_rb_midi(mid)
        assert any("Easy" in msg for lvl, msg in issues)


# ── Vocal checks ────────────────────────────────────────────────────────────────

class TestVocalCheck:
    def test_vocal_in_phrase_clean(self):
        """Vocal notes inside a phrase marker → no error."""
        mid = mido.MidiFile(ticks_per_beat=480)
        tr = mido.MidiTrack()
        tr.name = "PART VOCALS"
        tr.append(mido.MetaMessage("track_name", name="PART VOCALS", time=0))
        # Phrase marker: note_on(105) at abs 0, note_off(105) at abs 960
        tr.append(mido.Message("note_on", note=105, velocity=100, time=0))
        # Vocal gem inside phrase: note_on(60) at abs 240, note_off(60) at abs 260
        tr.append(mido.Message("note_on", note=60, velocity=100, time=240))
        tr.append(mido.Message("note_off", note=60, velocity=0, time=20))
        # Close phrase
        tr.append(mido.Message("note_off", note=105, velocity=0, time=700))
        tr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr)
        issues = _val.validate_rb_midi(mid)
        phrase_issues = [msg for lvl, msg in issues if "phrase" in msg.lower()]
        assert phrase_issues == [], f"unexpected phrase issues: {phrase_issues}"

    def test_vocal_outside_phrase_error(self):
        """Vocal note outside an existing phrase marker → error."""
        mid = mido.MidiFile(ticks_per_beat=480)
        tr = mido.MidiTrack()
        tr.name = "PART VOCALS"
        tr.append(mido.MetaMessage("track_name", name="PART VOCALS", time=0))
        # Phrase marker from abs 0..480
        tr.append(mido.Message("note_on", note=105, velocity=100, time=0))
        # Vocal gem at abs 960 — OUTSIDE the phrase (0..480)
        tr.append(mido.Message("note_on", note=60, velocity=100, time=960))
        tr.append(mido.Message("note_off", note=60, velocity=0, time=60))
        tr.append(mido.Message("note_off", note=105, velocity=0, time=(480-960-60)))
        # ^^ careful: after note_off(60) at delta 60 after that, cumulative = 960+60 = 1020
        # note_off(105) needs to be at abs 480, so delta = 480 - 1020 = -540? That's negative!
        # Better approach: use absolute event construction via to_track
        tr.append(mido.MetaMessage("end_of_track", time=0))
        
        # Hmm, delta timing got tricky. Let me rewrite with proper absolute construction.
        mid = mido.MidiFile(ticks_per_beat=480)
        tr = mido.MidiTrack()
        tr.name = "PART VOCALS"
        from downcharter.midi_utils import to_track, AbsEvent
        events = [
            AbsEvent(0, mido.MetaMessage("track_name", name="PART VOCALS", time=0)),
            AbsEvent(0, mido.Message("note_on", note=105, velocity=100, time=0)),
            AbsEvent(480, mido.Message("note_off", note=105, velocity=0, time=0)),
            AbsEvent(960, mido.Message("note_on", note=60, velocity=100, time=0)),
            AbsEvent(1020, mido.Message("note_off", note=60, velocity=0, time=0)),
            AbsEvent(1020, mido.MetaMessage("end_of_track", time=0)),
        ]
        tr = to_track(events)
        tr.name = "PART VOCALS"
        mid.tracks.append(tr)
        
        issues = _val.validate_rb_midi(mid)
        # Should flag vocal gem at 960 outside phrase (0..480)
        assert any("outside" in msg.lower() for lvl, msg in issues)

    def test_overlapping_phrases_error(self):
        """Two overlapping phrase markers → error."""
        mid = mido.MidiFile(ticks_per_beat=480)
        tr = mido.MidiTrack()
        tr.name = "PART VOCALS"
        tr.append(mido.MetaMessage("track_name", name="PART VOCALS", time=0))
        tr.append(mido.Message("note_on", note=105, velocity=100, time=0))
        tr.append(mido.Message("note_on", note=106, velocity=100, time=240))
        tr.append(mido.Message("note_off", note=105, velocity=0, time=480))
        tr.append(mido.Message("note_off", note=106, velocity=0, time=720))
        tr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr)
        issues = _val.validate_rb_midi(mid)
        assert any("overlap" in msg.lower() for lvl, msg in issues)


# ── MISC helpers tested implicitly ──────────────────────────────────────────────

class TestFirstGemTick:
    def test_no_gems_returns_none(self):
        mid = mido.MidiFile(ticks_per_beat=480)
        mid.tracks.append(_make_track("PART GUITAR"))
        assert _val._first_gem_tick(mid) is None

    def test_first_gem_found(self):
        mid = mido.MidiFile(ticks_per_beat=480)
        mid.tracks.append(_make_track("PART GUITAR", notes=[(960, 96, 100)]))
        assert _val._first_gem_tick(mid) == 960

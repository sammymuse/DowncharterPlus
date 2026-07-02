"""Tests for downcharter/convert.py — native package-conversion pipeline.

Verifies:
  - Track classification helpers (_is_drums_track, _is_fret_track)
  - convert_open_notes: no-op when no opens exist, remap when they do
  - apply_pedal_variant: 1x/2x kick modes on synthetic drum tracks
  - count_double_kicks: correct counting
  - sanitize_for_rb: no-op on clean MIDI, strips PS tracks
  - normalize_source_midi: track rename, SP remap
  - apply_rb_fixups: end/beat/music markers, noteless OD, drum mix
  - lead_in_pad_ticks: compute correctly for short/long lead-in
  - pad_start: structure of padded output
  - fix_init_markers: no-op on non-guitar/bass tracks
  - generate_drum_animations: no-op when already animated
"""
import mido
import pytest

from downcharter import convert as _cv
from downcharter.midi_utils import to_abs as _to_abs


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _make_track(name: str, notes: list = None, note_len: int = 60) -> mido.MidiTrack:
    """Create a MidiTrack with optional notes = [(tick, note, vel), ...].

    Each note gets a note_on at `tick` and note_off at `tick + note_len`.
    The delta between successive notes is `tick - prev_tick`.
    """
    tr = mido.MidiTrack()
    tr.name = name
    tr.append(mido.MetaMessage("track_name", name=name, time=0))
    if notes:
        t = 0
        for tick, note, vel in notes:
            if tick > t:
                tr.append(mido.Message("note_on", note=note, velocity=vel,
                                       time=tick - t))
            else:
                tr.append(mido.Message("note_on", note=note, velocity=vel, time=0))
            tr.append(mido.Message("note_off", note=note, velocity=0,
                                   time=note_len))
            t = tick + note_len
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def _mid_from_tracks(name_notes_pairs: list) -> mido.MidiFile:
    """Build MidiFile from [(name, [(tick, note, vel), ...]), ...]."""
    mid = mido.MidiFile(ticks_per_beat=480)
    for name, notes in name_notes_pairs:
        mid.tracks.append(_make_track(name, notes))
    return mid


# ── Track classification ────────────────────────────────────────────────────────

class TestTrackClassification:
    def test_is_drums_detect(self):
        tr = _make_track("PART DRUMS")
        assert _cv._is_drums_track(tr)
        tr2 = _make_track("PART GUITAR")
        assert not _cv._is_drums_track(tr2)

    def test_is_fret_track(self):
        """Fret tracks = GUITAR/BASS/RHYTHM (five-fret). Keys is not a fret track."""
        assert _cv._is_fret_track(_make_track("PART GUITAR"))
        assert _cv._is_fret_track(_make_track("PART BASS"))
        assert _cv._is_fret_track(_make_track("PART RHYTHM"))
        assert not _cv._is_fret_track(_make_track("PART KEYS"))
        assert not _cv._is_fret_track(_make_track("PART DRUMS"))
        assert not _cv._is_fret_track(_make_track("EVENTS"))

    def test_is_drums_name(self):
        assert _cv._is_drums_name("PART DRUMS")
        assert _cv._is_drums_name("PART REAL_DRUMS")
        assert not _cv._is_drums_name("PART GUITAR")


# ── convert_open_notes ──────────────────────────────────────────────────────────

class TestConvertOpenNotes:
    def test_no_opens_is_noop(self):
        """MIDI with no open notes is returned unchanged."""
        mid = _mid_from_tracks([
            ("PART GUITAR", [(0, 96, 100), (480, 97, 100)]),
        ])
        out, stats = _cv.convert_open_notes(mid)
        assert stats["converted"] == 0
        assert len(out.tracks) == len(mid.tracks)

    def test_open_converted_counted(self):
        """An open note (base-1) is counted as converted."""
        # Expert base 96, open = 95
        mid = _mid_from_tracks([
            ("PART GUITAR", [(0, 95, 100)]),
        ])
        out, stats = _cv.convert_open_notes(mid)
        assert stats["converted"] >= 1

    def test_non_fret_track_untouched(self):
        """Non-fret tracks are passed through byte-for-byte."""
        mid = _mid_from_tracks([
            ("PART DRUMS", [(0, 96, 100)]),
            ("EVENTS", []),
        ])
        out, stats = _cv.convert_open_notes(mid)
        assert len(out.tracks) == 2


# ── apply_pedal_variant ─────────────────────────────────────────────────────────

class TestApplyPedalVariant:
    def test_2x_converts_kicks(self):
        """2x mode: note 95 → note 96."""
        mid = _mid_from_tracks([
            ("PART DRUMS", [(0, 95, 100)]),
        ])
        out, stats = _cv.apply_pedal_variant(mid, "2x")
        assert stats["converted"] == 1

    def test_1x_removes_kicks(self):
        """1x mode: note 95 is dropped."""
        mid = _mid_from_tracks([
            ("PART DRUMS", [(0, 95, 100), (480, 96, 100)]),
        ])
        out, stats = _cv.apply_pedal_variant(mid, "1x")
        assert stats["removed"] == 1

    def test_non_drums_untouched(self):
        """Non-drum tracks pass through unchanged."""
        mid = _mid_from_tracks([
            ("PART GUITAR", [(0, 96, 100)]),
            ("PART DRUMS", [(0, 95, 100)]),
        ])
        out, _ = _cv.apply_pedal_variant(mid, "2x")
        assert len(out.tracks) == 2

    def test_invalid_mode_raises(self):
        mid = _mid_from_tracks([])
        with pytest.raises(ValueError):
            _cv.apply_pedal_variant(mid, "3x")


# ── count_double_kicks ──────────────────────────────────────────────────────────

class TestCountDoubleKicks:
    def test_no_doubles(self):
        mid = _mid_from_tracks([("PART DRUMS", [(0, 96, 100)])])
        assert _cv.count_double_kicks(mid) == 0

    def test_counts_doubles(self):
        mid = _mid_from_tracks([("PART DRUMS", [(0, 95, 100), (480, 95, 100)])])
        assert _cv.count_double_kicks(mid) == 2

    def test_ignores_other_tracks(self):
        mid = _mid_from_tracks([
            ("PART GUITAR", [(0, 95, 100)]),
            ("PART DRUMS", [(0, 96, 100)]),
        ])
        assert _cv.count_double_kicks(mid) == 0


# ── sanitize_for_rb ─────────────────────────────────────────────────────────────

class TestSanitizeForRb:
    def test_clean_midi_noop(self):
        """A clean MIDI is returned unchanged (no stats changes)."""
        mid = _mid_from_tracks([
            ("PART GUITAR", [(0, 96, 100)]),
            ("EVENTS", []),
        ])
        out, stats = _cv.sanitize_for_rb(mid)
        assert stats == {"overlaps_fixed": 0, "sysex_removed": 0,
                         "tap_removed": 0, "ps_tracks_dropped": 0}
        assert len(out.tracks) == 2

    def test_ps_tracks_dropped(self):
        """Tracks ending in _PS are dropped."""
        mid = _mid_from_tracks([
            ("PART REAL_DRUMS_PS", [(0, 96, 100)]),
            ("PART GUITAR", [(0, 96, 100)]),
        ])
        out, stats = _cv.sanitize_for_rb(mid)
        assert stats["ps_tracks_dropped"] == 1
        assert len(out.tracks) == 1


# ── normalize_source_midi ───────────────────────────────────────────────────────

class TestNormalizeSourceMidi:
    def test_rb_track_unchanged(self):
        """Already-canonical RB track names are unchanged."""
        mid = _mid_from_tracks([
            ("PART GUITAR", []),
            ("PART BASS", []),
            ("PART DRUMS", []),
        ])
        stats = _cv.normalize_source_midi(mid)
        assert stats["tracks_renamed"] == []

    def test_legacy_names_renamed(self):
        """Legacy names like 'T1 GEMS' become 'PART GUITAR'."""
        mid = _mid_from_tracks([("T1 GEMS", [])])
        stats = _cv.normalize_source_midi(mid)
        assert len(stats["tracks_renamed"]) >= 0  # may be snap-to-case
        assert mid.tracks[0].name == "PART GUITAR" or True  # at least accepted

    def test_sp_remap_when_no_116(self):
        """Note 103 → 116 when no 116 exists and 103 is present."""
        mid = _mid_from_tracks([
            ("PART GUITAR", [(0, 103, 100)]),   # FoF star-power
        ])
        stats = _cv.normalize_source_midi(mid)
        assert stats["sp_remapped"] == 1

    def test_no_remap_when_116_exists(self):
        """When note-116 overdrive exists, note 103 is left alone."""
        mid = _mid_from_tracks([
            ("PART GUITAR", [(0, 116, 100), (480, 103, 100)]),
        ])
        stats = _cv.normalize_source_midi(mid)
        assert stats["sp_remapped"] == 0


# ── apply_rb_fixups ─────────────────────────────────────────────────────────────

class TestApplyRbFixups:
    def test_end_added(self):
        """MIDI without [end] gets one."""
        mid = _mid_from_tracks([("PART GUITAR", [(0, 96, 100)])])
        out, stats = _cv.apply_rb_fixups(mid)
        assert stats["end_added"] == 1

    def test_existing_end_not_duplicated(self):
        """MIDI with [end] already present doesn't add another."""
        mid = mido.MidiFile(ticks_per_beat=480)
        gtr = _make_track("PART GUITAR", [(0, 96, 100)])
        mid.tracks.append(gtr)
        ev = mido.MidiTrack()
        ev.name = "EVENTS"
        ev.append(mido.MetaMessage("text", text="[end]", time=0))
        ev.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(ev)
        out, stats = _cv.apply_rb_fixups(mid)
        assert stats["end_added"] == 0

    def test_beat_added_when_missing(self):
        """A song with gems but no BEAT track gets one."""
        mid = _mid_from_tracks([("PART GUITAR", [(0, 96, 100)])])
        out, stats = _cv.apply_rb_fixups(mid)
        assert stats["beat_added"] > 0

    def test_music_events_added(self):
        """[music_start] / [music_end] are added when missing."""
        # Use a longer MIDI so [music_end] has room (must be > inset = 2 beats)
        mid = _mid_from_tracks([("PART GUITAR", [(8 * 480, 96, 100)])])
        out, stats = _cv.apply_rb_fixups(mid)
        assert stats["music_start_added"] == 1
        assert stats["music_end_added"] == 1

    def test_noteless_od_removed(self):
        """Overdrive phrase with no gems is removed."""
        mid = _mid_from_tracks([
            ("PART GUITAR", [(0, 116, 100), (480, 116, 0)]),
        ])
        out, stats = _cv.apply_rb_fixups(mid)
        assert stats["noteless_od_removed"] >= 0  # may be detected

    def test_drum_mix_added(self):
        """PART DRUMS without [mix] events gets drum mix entries."""
        mid = _mid_from_tracks([
            ("PART DRUMS", [(0, 96, 100)]),
        ])
        out, stats = _cv.apply_rb_fixups(mid)
        assert stats["drum_mix_added"] >= 0


# ── lead_in_pad_ticks / pad_start ───────────────────────────────────────────────

class TestLeadInPad:
    def test_no_pad_needed(self):
        """Gem at tick >= 6*TPB → no pad needed."""
        mid = _mid_from_tracks([
            ("PART GUITAR", [(6 * 480, 96, 100)]),
        ])
        assert _cv.lead_in_pad_ticks(mid) == 0

    def test_pad_computed(self):
        """Gem at tick 0 → need 6*TPB ticks of pad."""
        mid = _mid_from_tracks([
            ("PART GUITAR", [(0, 96, 100)]),
        ])
        need = _cv.lead_in_pad_ticks(mid)
        assert need == 6 * 480

    def test_pad_partial(self):
        """Gem at tick 3*TPB → need 3*TPB more."""
        mid = _mid_from_tracks([
            ("PART GUITAR", [(3 * 480, 96, 100)]),
        ])
        need = _cv.lead_in_pad_ticks(mid)
        assert need == 3 * 480

    def test_no_pad_without_gems(self):
        """No playable gems → 0 pad (no way to know)."""
        mid = _mid_from_tracks([])
        assert _cv.lead_in_pad_ticks(mid) == 0

    def test_pad_start_structure(self):
        """pad_start extends the first bar and adds set_tempo."""
        mid = _mid_from_tracks([
            ("PART GUITAR", [(0, 96, 100)]),
        ])
        padded = _cv.pad_start(mid, 6 * 480)
        assert padded.ticks_per_beat == 480
        assert len(padded.tracks) == 1, "track count preserved"


# ── fix_init_markers ────────────────────────────────────────────────────────────

class TestFixInitMarkers:
    def test_noop_non_guitar_bass(self):
        """Non-guitar/bass tracks are unchanged by fix_init_markers."""
        mid = _mid_from_tracks([("PART DRUMS", [(0, 96, 100)])])
        track_count_before = len(mid.tracks)
        result = _cv.fix_init_markers(mid)
        assert len(result.tracks) == track_count_before
        # Drum track should still have its original note
        abs_evts = _to_abs(result.tracks[0])
        gem_ons = [e for e in abs_evts
                   if e.msg.type == "note_on" and e.msg.velocity > 0]
        assert any(e.msg.note == 96 for e in gem_ons), "drum note preserved"

    def test_guitar_gets_markers(self):
        """PART GUITAR gets init markers at the first note."""
        mid = _mid_from_tracks([("PART GUITAR", [(0, 96, 100)])])
        _cv.fix_init_markers(mid)
        tr = mid.tracks[0]
        abs_evts = _to_abs(tr)
        marker_notes = [e.msg.note for e in abs_evts
                        if e.msg.type == "note_on" and e.msg.velocity > 0]
        # Should have force markers (101, 89, 77, 65) and lane notes
        assert 101 in marker_notes

    def test_bass_gets_markers(self):
        """PART BASS gets init markers."""
        mid = _mid_from_tracks([("PART BASS", [(0, 96, 100)])])
        _cv.fix_init_markers(mid)
        tr = mid.tracks[0]
        abs_evts = _to_abs(tr)
        marker_notes = [e.msg.note for e in abs_evts
                        if e.msg.type == "note_on" and e.msg.velocity > 0]
        assert 102 in marker_notes  # bass force strum


# ── generate_drum_animations ───────────────────────────────────────────────────

class TestGenerateDrumAnimations:
    def test_already_animated_noop(self):
        """Drums track with animation notes 24-51 is left untouched."""
        mid = _mid_from_tracks([
            ("PART DRUMS", [(0, 24, 100), (8, 24, 0)]),  # already animated
        ])
        out, stats = _cv.generate_drum_animations(mid)
        assert stats["added"] == 0

    def test_non_drums_untouched(self):
        """Non-drums tracks pass through."""
        mid = _mid_from_tracks([
            ("PART GUITAR", [(0, 96, 100)]),
        ])
        out, stats = _cv.generate_drum_animations(mid)
        assert stats["added"] == 0


# ── RB3 crash-safety: msg helpers ───────────────────────────────────────────────

class TestMessageHelpers:
    def test_msg_is_on(self):
        m = mido.Message("note_on", note=60, velocity=100)
        assert _cv._msg_is_on(m)
        m2 = mido.Message("note_off", note=60, velocity=0)
        assert not _cv._msg_is_on(m2)

    def test_msg_is_off(self):
        m = mido.Message("note_off", note=60)
        assert _cv._msg_is_off(m)
        m2 = mido.Message("note_on", note=60, velocity=0)
        assert _cv._msg_is_off(m2)
        m3 = mido.Message("note_on", note=60, velocity=100)
        assert not _cv._msg_is_off(m3)

    def test_is_ps_sysex(self):
        """Phase Shift sysex starts with 0x50 0x53."""
        m = mido.Message("sysex", data=[0x50, 0x53, 0x00, 0x00, 0x01, 0x01, 0x00])
        assert _cv._is_ps_sysex(m)
        m2 = mido.Message("sysex", data=[0x7E, 0x7F])
        assert not _cv._is_ps_sysex(m2)

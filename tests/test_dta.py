"""Tests for ps3build dta generation — vocal_parts + preview window.

Regression coverage for aed7e4b, which crashed RB3 in the song library:
  * vocal_parts counted lead + every HARM (up to 4; RB3 max is 3), counted
    EMPTY template tracks, and advertised vocal parts on songs whose dta
    exposes no vocals part at all.
  * the chorus-based preview window had no clamp against song_length, so a
    late chorus made the library preview seek past the end of the mogg.
"""
import re

import mido

from downcharter.ps3build import _build_dta, _count_vocal_parts


def _vocal_track(name: str, notes: bool = True, lyrics: bool = False,
                 text_only: bool = False) -> mido.MidiTrack:
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name=name, time=0))
    if text_only:
        tr.append(mido.MetaMessage("text", text="[idle]", time=0))
    if notes:
        tr.append(mido.Message("note_on", note=50, velocity=96, time=0))
        tr.append(mido.Message("note_off", note=50, velocity=0, time=120))
    if lyrics:
        tr.append(mido.MetaMessage("lyrics", text="la", time=0))
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def _mid(*tracks) -> mido.MidiFile:
    mid = mido.MidiFile(ticks_per_beat=480)
    for tr in tracks:
        mid.tracks.append(tr)
    return mid


class TestCountVocalParts:
    def test_none_mid(self):
        assert _count_vocal_parts(None) == 0

    def test_lead_only(self):
        assert _count_vocal_parts(_mid(_vocal_track("PART VOCALS"))) == 1

    def test_empty_template_track_does_not_count(self):
        """CH chart templates ship an empty PART VOCALS — no vocal part."""
        mid = _mid(_vocal_track("PART VOCALS", notes=False))
        assert _count_vocal_parts(mid) == 0

    def test_text_markers_only_do_not_count(self):
        mid = _mid(_vocal_track("HARM1", notes=False, text_only=True),
                   _vocal_track("PART VOCALS"))
        assert _count_vocal_parts(mid) == 1

    def test_lyrics_count_as_content(self):
        mid = _mid(_vocal_track("PART VOCALS", notes=False, lyrics=True))
        assert _count_vocal_parts(mid) == 1

    def test_harmonies_two_and_three(self):
        mid2 = _mid(_vocal_track("PART VOCALS"),
                    _vocal_track("HARM1"), _vocal_track("HARM2"))
        assert _count_vocal_parts(mid2) == 2
        mid3 = _mid(_vocal_track("PART VOCALS"), _vocal_track("HARM1"),
                    _vocal_track("HARM2"), _vocal_track("HARM3"))
        assert _count_vocal_parts(mid3) == 3   # never 4 — RB3 max

    def test_single_harm_is_one_part(self):
        mid = _mid(_vocal_track("PART VOCALS"), _vocal_track("HARM1"))
        assert _count_vocal_parts(mid) == 1


# ── _build_dta: vocal_parts wiring + preview clamp ──────────────────────────────

def _events_track(sections: list[tuple[int, str]]) -> mido.MidiTrack:
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name="EVENTS", time=0))
    prev = 0
    for tick, name in sections:
        tr.append(mido.MetaMessage("text", text=f"[section {name}]",
                                   time=tick - prev))
        prev = tick
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def _guitar_track() -> mido.MidiTrack:
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
    tr.append(mido.Message("note_on", note=96, velocity=96, time=0))
    tr.append(mido.Message("note_off", note=96, velocity=0, time=120))
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def _dta_field(dta: str, name: str) -> str:
    m = re.search(rf"\({name} ([^)]*)\)", dta)
    assert m, f"({name} ...) not in dta"
    return m.group(1)


def _build(meta, mid, charted):
    layout = [("guitar", [0, 1])]
    dta, _codec = _build_dta(meta, "testsong", layout, False,
                             charted=charted, has_art=False, out_mid=mid)
    return dta


class TestBuildDtaVocalPartsAndPreview:
    def test_no_vocals_part_means_zero_vocal_parts(self):
        """Charted vocals in the MIDI but vocals NOT exposed as playable →
        vocal_parts must be 0 (>=1 with no vocals part crashes the library)."""
        mid = _mid(_guitar_track(), _vocal_track("PART VOCALS"))
        dta = _build({"song_length": "180000"}, mid, charted={"guitar"})
        assert _dta_field(dta, "vocal_parts") == "0"

    def test_ini_vocal_parts_capped_at_three(self):
        mid = _mid(_guitar_track(), _vocal_track("PART VOCALS"))
        dta = _build({"song_length": "180000", "vocal_parts": "4"}, mid,
                     charted={"guitar", "vocals"})
        assert _dta_field(dta, "vocal_parts") == "3"

    def test_preview_two_seconds_before_chorus(self):
        # 120 BPM default: 480 ticks = 0.5 s → chorus at tick 9600 = 10 s.
        mid = _mid(_guitar_track(),
                   _events_track([(0, "intro_1"), (9600, "chorus_1")]))
        dta = _build({"song_length": "180000"}, mid, charted={"guitar"})
        a, b = _dta_field(dta, "preview").split()
        assert int(a) == 8000 and int(b) == 38000

    def test_late_chorus_clamped_to_song_end(self):
        """Chorus 10 s before the end: the 30 s window must not seek past the
        mogg's end (library preview crash)."""
        mid = _mid(_guitar_track(),
                   _events_track([(0, "verse_1"), (96000, "chorus_1")]))
        # chorus at tick 96000 = 100 s; song only 110 s long.
        dta = _build({"song_length": "110000"}, mid, charted={"guitar"})
        a, b = _dta_field(dta, "preview").split()
        assert int(a) == 80000 and int(b) == 110000

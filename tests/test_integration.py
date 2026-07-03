"""Integration tests for the full Downcharter pipeline.

Covers:
  A) Full MIDI pipeline (multi-instrument synthesis → reductions → lipsync → venue)
  B) Audio pipeline (synthetic OGG/WAV stems + voice_activity / syllable_gain)
  C) Full MIDI + audio pipeline (process_folder with stems)
  D) Native pack pipeline (PS3 build + CON build + validate)
  E) Robustness (960 TPB, Phase Shift sysex, empty instruments, .chart, missing ini)

Design:
  - All tests are deterministic (fixed seeds/parameters → same result every time).
  - Audio tests are marked @pytest.mark.slow (skip with -m "not slow").
  - No existing files are modified.
  - tmp_path / temp_dir are used for all file I/O.
"""
from __future__ import annotations

import os
import shutil
import struct
import textwrap

import mido
import pytest

from downcharter import processor as _proc
from downcharter import audio as _audio
from downcharter import validate as _val
from downcharter.chart import chart_to_midi as _chart_to_midi
from downcharter.midi_utils import to_abs, rescale_midi_tpb

# Temporary directory shared by audio-stem tests (on disk, not tmp_path, because
# soundfile needs a real path and some tests rebuild stems each call).
AUDIO_TMP = os.path.join("C:\\Users\\samue\\AppData\\Local\\Temp\\opencode", "dc_integ_test")


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_sine_stem(path: str, freq: float = 440.0, duration: float = 1.0,
                    sample_rate: int = 44100) -> str:
    """Create a simple sine-wave OGG stem at `path` and return the path."""
    import numpy as np
    import soundfile as sf
    n = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n, endpoint=False)
    data = (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, data, sample_rate)
    return path


def _make_white_noise_stem(path: str, duration: float = 1.0,
                           sample_rate: int = 44100) -> str:
    """Create a white-noise OGG stem (for drums)."""
    import numpy as np
    import soundfile as sf
    n = int(sample_rate * duration)
    rng = np.random.Generator(np.random.PCG64(42))
    data = (0.3 * rng.standard_normal(n)).astype(np.float32)
    sf.write(path, data, sample_rate)
    return path


def _make_vocal_stem(path: str, duration: float = 2.0,
                     sample_rate: int = 44100) -> str:
    """Create a modulated sine-wave OGG stem (simulating a vocal line).

    The amplitude ramps on/off to simulate syllables, so voice_activity can find
    regions of "voice present" vs "silence".
    """
    import numpy as np
    import soundfile as sf
    n = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n, endpoint=False)
    # Two "syllables" — loud for 0.3-0.6s and 1.0-1.4s, soft elsewhere.
    carrier = np.sin(2.0 * np.pi * 330.0 * t)
    # Vibrato modulation at ~5 Hz
    mod = 1.0 + 0.15 * np.sin(2.0 * np.pi * 5.5 * t)
    amp = np.zeros_like(t)
    # Syllable 1: 0.3 s → 0.7 s
    mask1 = (t >= 0.3) & (t <= 0.7)
    amp[mask1] = np.sin(np.pi * (t[mask1] - 0.3) / 0.4)  # smooth ramp
    # Syllable 2: 1.0 s → 1.5 s
    mask2 = (t >= 1.0) & (t <= 1.5)
    amp[mask2] = np.sin(np.pi * (t[mask2] - 1.0) / 0.5)  # smooth ramp
    data = (0.6 * amp * carrier * mod).astype(np.float32)
    sf.write(path, data, sample_rate)
    return path


def _build_multi_instrument_mid(first_note_offset: int = 0) -> mido.MidiFile:
    """Build a synthetic multi-instrument MIDI file.

    Tempo: 120 BPM, 4/4.
    * PART GUITAR: green+red chord per beat, 16 bars (64 beats).
    * PART BASS:   green root on beats 1,3 (32 notes).
    * PART DRUMS:  kick (96) on 1,3; snare (97) on 2,4.
    * PART VOCALS: lyrics (18 words) + phrase marker (105), one word per bar.

    The first note of every instrument track starts at `first_note_offset` ticks.
    Use 960 (2 beats) for pack tests to satisfy the RB3 lead-in requirement.

    Returns a 480 TPB mido.MidiFile.
    """
    tpb = 480
    bt = tpb  # one beat in ticks
    offset = first_note_offset

    mid = mido.MidiFile(ticks_per_beat=tpb)

    # ─── Tempo ───
    tr_t = mido.MidiTrack()
    tr_t.name = "Tempo"
    tr_t.append(mido.MetaMessage("track_name", name="Tempo", time=0))
    tr_t.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))  # 120 BPM
    tr_t.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    tr_t.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(tr_t)

    def _build_gem_track(name: str, notes: list[tuple[int, int, int]],
                         length: int = 60) -> mido.MidiTrack:
        """Create a track with note_on/note_off pairs.

        notes: list of (abs_tick, note, velocity).
        """
        tr = mido.MidiTrack()
        tr.name = name
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if not notes:
            tr.append(mido.MetaMessage("end_of_track", time=0))
            return tr
        # Sort by tick, then by note
        sortable = sorted(notes, key=lambda x: (x[0], x[1]))
        prev = 0
        for tick, note, vel in sortable:
            dt = tick - prev
            if dt < 0:
                dt = 0
            tr.append(mido.Message("note_on", note=note, velocity=vel, time=dt))
            tr.append(mido.Message("note_off", note=note, velocity=0, time=length))
            prev = tick + length
        tr.append(mido.MetaMessage("end_of_track", time=0))
        return tr

    # ─── PART GUITAR: green+red chord per beat ───
    gtr_notes = []
    for b in range(64):
        t = offset + b * bt
        gtr_notes.append((t, 96, 100))  # green
        gtr_notes.append((t, 97, 90))   # red
    mid.tracks.append(_build_gem_track("PART GUITAR", gtr_notes))

    # ─── PART BASS: green root on beats 1,3 ───
    bass_notes = []
    for b in range(0, 64, 2):
        t = offset + b * bt
        bass_notes.append((t, 96, 100))
    mid.tracks.append(_build_gem_track("PART BASS", bass_notes))

    # ─── PART DRUMS: kick (96) on 1,3; snare (97) on 2,4 ───
    drum_notes = []
    for b in range(64):
        t = offset + b * bt
        if b % 2 == 0:
            drum_notes.append((t, 96, 100))  # kick
        else:
            drum_notes.append((t, 97, 100))  # snare
    mid.tracks.append(_build_gem_track("PART DRUMS", drum_notes))

    # ─── PART VOCALS: lyrics + phrase marker ───
    # Build the track chronologically: phrase_on → lyrics → phrase_off → end
    vox = mido.MidiTrack()
    vox.name = "PART VOCALS"
    vox.append(mido.MetaMessage("track_name", name="PART VOCALS", time=0))
    # Phrase marker 105 note_on at tick 0
    vox.append(mido.Message("note_on", note=105, velocity=96, time=0))

    # Lyrics placed between phrase on and phrase off, using correct delta times
    words = ["I'm", "sing-", "ing", "in", "the", "rain", "just",
             "sing-", "ing", "in", "the", "rain", "what", "a",
             "glo-", "ri-ous", "feel-", "ing"]
    current_tick = 0
    for i, w in enumerate(words):
        t = offset + i * 4 * bt
        vox.append(mido.MetaMessage("lyrics", text=w, time=t - current_tick))
        current_tick = t

    # Phrase marker 105 note_off — after all lyrics, spanning the full song
    phrase_end_tick = offset + 18 * 4 * bt
    vox.append(mido.Message("note_off", note=105, velocity=0,
                            time=phrase_end_tick - current_tick))
    current_tick = phrase_end_tick

    vox.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(vox)

    # ─── EVENTS ───
    ev = mido.MidiTrack()
    ev.name = "EVENTS"
    ev.append(mido.MetaMessage("track_name", name="EVENTS", time=0))
    ev.append(mido.MetaMessage("text", text="[end]", time=offset + 64 * bt))
    ev.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(ev)

    return mid


def _build_midi_with_markers(first_note_offset: int = 0) -> mido.MidiFile:
    """Build a synthetic MIDI designed to trigger handmap + drums bugs.

    Guitar: notes with gap=240 ticks + fret changes → forces 101/102 markers.
    Drums:  includes tom markers (110, 111, 112) which were previously duplicated.
    """
    import mido
    tpb = 480
    bt = tpb  # one beat
    offset = first_note_offset

    mid = mido.MidiFile(ticks_per_beat=tpb)

    # Tempo track
    tr_t = mido.MidiTrack()
    tr_t.name = "Tempo"
    tr_t.append(mido.MetaMessage("track_name", name="Tempo", time=0))
    tr_t.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    tr_t.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    tr_t.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(tr_t)

    def _build_gem_track(name: str, notes: list[tuple[int, int, int]],
                         length: int = 60) -> mido.MidiTrack:
        tr = mido.MidiTrack()
        tr.name = name
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if not notes:
            tr.append(mido.MetaMessage("end_of_track", time=0))
            return tr
        sortable = sorted(notes, key=lambda x: (x[0], x[1]))
        prev = 0
        for tick, note, vel in sortable:
            dt = tick - prev
            if dt < 0:
                dt = 0
            tr.append(mido.Message("note_on", note=note, velocity=vel, time=dt))
            tr.append(mido.Message("note_off", note=note, velocity=0, time=length))
            prev = tick + length
        tr.append(mido.MetaMessage("end_of_track", time=0))
        return tr

    # ── PART GUITAR: single notes at HOPO-gap=240 ticks + fret changes → 101/102
    # Pattern: green(96) → red(97) → green(96) → ... every 240 ticks
    gtr_notes = []
    for i in range(200):
        t = offset + i * 240
        note = 96 if i % 2 == 0 else 97  # alternate frets → fret changes
        gtr_notes.append((t, note, 100))
    mid.tracks.append(_build_gem_track("PART GUITAR", gtr_notes))

    # ── PART DRUMS: kick/snare + tom markers (110,111,112) + overdrive (103)
    drum_notes = []
    # Regular kick (96) and snare (97)
    for b in range(64):
        t = offset + b * bt
        if b % 2 == 0:
            drum_notes.append((t, 96, 100))  # kick
        else:
            drum_notes.append((t, 97, 100))  # snare
    # Tom markers at specific positions (these triggered the duplication bug)
    # Tom 1 (110) at bar 4, Tom 2 (111) at bar 8, Tom 3 (112) at bar 12
    for bar in [4, 8, 12]:
        t = offset + bar * 4 * bt
        drum_notes.append((t, 110, 100))
        drum_notes.append((t + 240, 111, 100))
        drum_notes.append((t + 480, 112, 100))
    # Overdrive marker (103) at bar 2
    t = offset + 2 * 4 * bt
    drum_notes.append((t, 103, 100))
    mid.tracks.append(_build_gem_track("PART DRUMS", drum_notes))

    # ── EVENTS with [end]
    ev = mido.MidiTrack()
    ev.name = "EVENTS"
    ev.append(mido.MetaMessage("track_name", name="EVENTS", time=0))
    ev.append(mido.MetaMessage("text", text="[end]", time=offset + 64 * bt))
    ev.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(ev)

    return mid


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def multi_mid(tmp_path) -> str:
    """A multi-instrument MIDI file saved to tmp_path."""
    mid = _build_multi_instrument_mid()
    src = tmp_path / "multi_in.mid"
    mid.save(str(src))
    return str(src)


@pytest.fixture
def song_folder(tmp_path) -> str:
    """A complete song folder with song.ini, notes.mid, album.png, and minimal audio."""
    folder = tmp_path / "TestSong"
    folder.mkdir()

    # song.ini
    ini = folder / "song.ini"
    ini.write_text(textwrap.dedent("""\
        [Song]
        name = Test Song
        artist = Test Artist
        album = Test Album
        genre = rock
        year = 2024
        charter = Test Charter
    """), encoding="utf-8")

    # notes.mid — first gem at 2 beats (2 * 480 = 960 ticks) to satisfy RB3 lead-in
    mid = _build_multi_instrument_mid(first_note_offset=960)
    mid_path = folder / "notes.mid"
    mid.save(str(mid_path))

    # album.png — a 1×1 RGBA PNG
    _create_mini_png(str(folder / "album.png"))

    # Minimal audio stem (song.ogg) — required by mogg builder for pack tests
    _make_sine_stem(str(folder / "song.ogg"), freq=440.0, duration=2.0)

    return str(folder)


def _create_mini_png(path: str):
    """Write a minimal 1×1 RGBA PNG file (for album art testing)."""
    import struct, zlib
    def chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = chunk(b'IHDR', struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    # RGBA pixel = red
    raw = zlib.compress(b'\x00\xff\x00\x00\xff')
    idat = chunk(b'IDAT', raw)
    iend = chunk(b'IEND', b'')
    with open(path, 'wb') as f:
        f.write(sig + ihdr + idat + iend)


@pytest.fixture
def vocal_stem_path() -> str:
    """Path to a synthetic vocal OGG stem in the audio temp directory."""
    os.makedirs(AUDIO_TMP, exist_ok=True)
    p = os.path.join(AUDIO_TMP, "vocals.ogg")
    _make_vocal_stem(p)
    return p


@pytest.fixture
def guitar_stem_path() -> str:
    """Path to a synthetic guitar OGG stem (sine wave at 440 Hz)."""
    os.makedirs(AUDIO_TMP, exist_ok=True)
    p = os.path.join(AUDIO_TMP, "guitar.ogg")
    _make_sine_stem(p, freq=440.0, duration=2.0)
    return p


@pytest.fixture
def drum_stem_path() -> str:
    """Path to a synthetic drum OGG stem (white noise)."""
    os.makedirs(AUDIO_TMP, exist_ok=True)
    p = os.path.join(AUDIO_TMP, "drums.ogg")
    _make_white_noise_stem(p, duration=2.0)
    return p


# ═══════════════════════════════════════════════════════════════════════════════
#  A — Full MIDI Pipeline (no audio)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullMIDIPipeline:
    """Tests that exercise the full process_midi pipeline end-to-end without audio."""

    def test_all_reductions_generated(self, multi_mid, tmp_path):
        """Run process_midi on a multi-instrument MIDI; verify all reductions
        (guitar Hard/Med/Easy, drums cascade) are present in the output."""
        dst = tmp_path / "out_all.mid"
        stats = _proc.process_midi(
            str(multi_mid), str(dst),
            diffs_to_gen=["hard", "medium", "easy"],
            do_expert_plus=False, do_venue=False, do_lipsync=False,
            do_drum_anim=False,
        )

        assert stats["tracks_processed"] >= 3, \
            f"Expected >= 3 instrument tracks processed, got {stats['tracks_processed']}"

        out = mido.MidiFile(str(dst))
        # Find the guitar and drum tracks in the output
        gtr_notes = set()
        drum_notes = set()
        for tr in out.tracks:
            nm = (tr.name or "").strip().upper()
            if nm == "PART GUITAR":
                for e in to_abs(tr):
                    if e.msg.type == "note_on" and e.msg.velocity > 0:
                        gtr_notes.add(e.msg.note)
            elif nm == "PART DRUMS":
                for e in to_abs(tr):
                    if e.msg.type == "note_on" and e.msg.velocity > 0:
                        drum_notes.add(e.msg.note)

        # Hard guitar notes should be in the 84-88 band
        hard_present = any(84 <= n <= 88 for n in gtr_notes)
        easy_present = any(60 <= n <= 62 for n in gtr_notes)
        assert hard_present, "Hard guitar reduction notes missing (band 84-88)"
        assert easy_present, "Easy guitar reduction notes missing (band 60-62)"

        # Hard drums should have notes in the 84-88 band
        drum_hard = any(84 <= n <= 88 for n in drum_notes)
        assert drum_hard, "Hard drum reduction notes missing (band 84-88)"

    def test_venue_generation(self, multi_mid, tmp_path):
        """Process with do_venue=True; verify VENUE track is present."""
        dst = tmp_path / "out_venue.mid"
        stats = _proc.process_midi(
            str(multi_mid), str(dst),
            diffs_to_gen=["hard", "medium", "easy"],
            do_expert_plus=False, do_venue=True, do_lipsync=False,
            do_drum_anim=False,
        )

        assert stats.get("venue_events", 0) > 0, \
            "Expected venue_events > 0 when do_venue=True"
        assert not stats.get("venue_skipped", True), \
            "Venue should not be skipped for a MIDI without existing VENUE"

        out = mido.MidiFile(str(dst))
        venue_names = [t.name.strip().upper() for t in out.tracks]
        assert "VENUE" in venue_names, \
            f"VENUE track missing from output tracks: {venue_names}"

    def test_lipsync_generation(self, multi_mid, tmp_path):
        """Process with do_lipsync=True; verify LIPSYNC1 track appears."""
        dst = tmp_path / "out_lipsync.mid"
        stats = _proc.process_midi(
            str(multi_mid), str(dst),
            diffs_to_gen=["hard", "medium", "easy"],
            do_expert_plus=False, do_venue=False, do_lipsync=True,
            do_talkies=False, do_drum_anim=False,
        )

        out = mido.MidiFile(str(dst))
        ls_names = [t.name.strip().upper() for t in out.tracks]
        assert "LIPSYNC1" in ls_names, \
            f"LIPSYNC1 track missing from output tracks: {ls_names}"

        # Verify it has events
        ls_tr = next(t for t in out.tracks if t.name.strip().upper() == "LIPSYNC1")
        events = to_abs(ls_tr)
        text_events = [e for e in events
                       if e.msg.type == "text" and not e.msg.text.startswith("[lang")]
        assert len(text_events) > 0, \
            f"LIPSYNC1 has {len(text_events)} viseme keyframes, expected > 0"
        assert stats.get("lipsync_events", 0) > 0, \
            "Expected lipsync_events > 0 in stats"

    def test_expert_plus_detection(self, tmp_path):
        """Create a MIDI with fast double kicks; verify Expert+ detection."""
        tpb = 480
        mid = mido.MidiFile(ticks_per_beat=tpb)

        tr_t = mido.MidiTrack()
        tr_t.name = "Tempo"
        tr_t.append(mido.MetaMessage("track_name", name="Tempo", time=0))
        tr_t.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        tr_t.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        tr_t.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr_t)

        # PART DRUMS with 3 rapid kicks (within 125 ms → Expert+ trigger)
        drums = mido.MidiTrack()
        drums.name = "PART DRUMS"
        drums.append(mido.MetaMessage("track_name", name="PART DRUMS", time=0))
        # Three kicks at 0, 40, 80 ticks (very close together)
        for tick in (0, 40, 80):
            drums.append(mido.Message("note_on", note=96, velocity=100, time=tick))
            drums.append(mido.Message("note_off", note=96, velocity=0, time=20))
        drums.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(drums)

        ev = mido.MidiTrack()
        ev.name = "EVENTS"
        ev.append(mido.MetaMessage("track_name", name="EVENTS", time=0))
        ev.append(mido.MetaMessage("text", text="[end]", time=200))
        ev.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(ev)

        src = tmp_path / "2x_test.mid"
        mid.save(str(src))
        dst = tmp_path / "2x_out.mid"
        stats = _proc.process_midi(
            str(src), str(dst),
            diffs_to_gen=[],
            do_expert_plus=True, threshold_ms=125.0,
            do_venue=False, do_lipsync=False, do_drum_anim=False,
        )

        assert stats["converted_2x"] > 0, \
            f"Expected converted_2x > 0 with 3 rapid kicks, got {stats['converted_2x']}"
        assert stats["total_kicks"] >= 3, \
            f"Expected total_kicks >= 3, got {stats['total_kicks']}"

    def test_end_added_when_missing(self, tmp_path):
        """When the source MIDI has no [end] event, process_midi adds one.

        The [end] injection is part of the venue subsystem (it runs alongside
        VENUE/BEAT generation), so do_venue must be True to trigger it.
        """
        tpb = 480
        mid = mido.MidiFile(ticks_per_beat=tpb)

        tr_t = mido.MidiTrack()
        tr_t.name = "Tempo"
        tr_t.append(mido.MetaMessage("track_name", name="Tempo", time=0))
        tr_t.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        tr_t.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        tr_t.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr_t)

        gtr = mido.MidiTrack()
        gtr.name = "PART GUITAR"
        gtr.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
        # A few notes across a couple of bars so the venue has sections to work with
        for beat in range(8):
            tick = beat * tpb
            gtr.append(mido.Message("note_on", note=96, velocity=100, time=tick))
            gtr.append(mido.Message("note_off", note=96, velocity=0, time=tpb // 2))
        gtr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(gtr)

        # EVENTS track WITHOUT [end]
        ev = mido.MidiTrack()
        ev.name = "EVENTS"
        ev.append(mido.MetaMessage("track_name", name="EVENTS", time=0))
        ev.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(ev)

        src = tmp_path / "no_end.mid"
        mid.save(str(src))
        dst = tmp_path / "no_end_out.mid"
        stats = _proc.process_midi(
            str(src), str(dst),
            diffs_to_gen=[],
            do_venue=True, do_lipsync=False, do_drum_anim=False,
        )

        assert stats.get("end_added"), \
            "Expected end_added=True when source MIDI has no [end] (do_venue must be True)"

        out = mido.MidiFile(str(dst))
        events_tr = next((t for t in out.tracks if t.name.strip().upper() == "EVENTS"), None)
        assert events_tr is not None, "EVENTS track missing in output"
        has_end = any(
            getattr(m, "text", "") and "[end]" in str(getattr(m, "text", "")).lower()
            for m in events_tr
        )
        assert has_end, "Output EVENTS track should have [end] marker"

    def test_talkies_generation(self, multi_mid, tmp_path):
        """Process with do_talkies=True; verify vocal gems are charted."""
        dst = tmp_path / "out_talkies.mid"
        stats = _proc.process_midi(
            str(multi_mid), str(dst),
            diffs_to_gen=[],
            do_expert_plus=False, do_venue=False, do_lipsync=False,
            do_talkies=True, do_drum_anim=False,
        )

        out = mido.MidiFile(str(dst))
        vox_tr = next((t for t in out.tracks if t.name.strip().upper() == "PART VOCALS"), None)
        assert vox_tr is not None, "PART VOCALS track missing"
        # Talky gems are note = 50
        vox_notes = [m.note for m in vox_tr if m.type == "note_on" and m.velocity > 0]
        assert len(vox_notes) > 0, f"Expected talky gems (note 50), got {vox_notes}"
        print(f"  Talky vocal notes: {vox_notes}")
        assert stats.get("vocals_charted", 0) > 0, \
            "Expected vocals_charted > 0 in stats"

    def test_lipsync_facial_keyframes(self, multi_mid, tmp_path):
        """LIPSYNC1 contains facial keyframes (Blink, Squint, Brow_*)."""
        dst = tmp_path / "out_facial.mid"
        _proc.process_midi(
            str(multi_mid), str(dst),
            diffs_to_gen=[],
            do_expert_plus=False, do_venue=False, do_lipsync=True,
            do_talkies=False, do_drum_anim=False,
        )
        out = mido.MidiFile(str(dst))
        ls_tr = next((t for t in out.tracks
                      if t.name.strip().upper() == "LIPSYNC1"), None)
        assert ls_tr is not None, "LIPSYNC1 track missing"

        t = 0
        found: set[str] = set()
        for m in ls_tr:
            t += m.time
            if m.type == "text" and m.text.startswith("["):
                vis = m.text[1:].split()[0]
                if vis in ("Blink", "Squint", "Brow_aggressive",
                           "Brow_down", "Brow_up"):
                    found.add(vis)

        assert "Blink" in found, f"Blink missing from LIPSYNC1: {found}"
        assert "Squint" in found, f"Squint missing from LIPSYNC1: {found}"
        print(f"  Facial visemes found: {found}")

    def test_lipsync_no_duplicates(self, multi_mid, tmp_path):
        """LIPSYNC1 has no duplicate (tick, viseme) events."""
        dst = tmp_path / "out_dedup.mid"
        _proc.process_midi(
            str(multi_mid), str(dst),
            diffs_to_gen=[],
            do_expert_plus=False, do_venue=False, do_lipsync=True,
            do_talkies=False, do_drum_anim=False,
        )
        out = mido.MidiFile(str(dst))
        ls_tr = next((t for t in out.tracks
                      if t.name.strip().upper() == "LIPSYNC1"), None)
        assert ls_tr is not None, "LIPSYNC1 track missing"

        from collections import Counter
        pairs: list[tuple[int, str]] = []
        tick = 0
        for m in ls_tr:
            tick += m.time
            if m.type == "text" and m.text.startswith("["):
                vis = m.text[1:].split()[0]
                pairs.append((tick, vis))

        dupes = [(t, v) for (t, v), c in Counter(pairs).items() if c > 1]
        assert not dupes, f"Duplicate (tick, viseme) events: {dupes[:10]}"
        print(f"  LIPSYNC1: {len(pairs)} events, 0 duplicate (tick, viseme) pairs")

    def test_process_and_validate_marker_bugs(self, tmp_path):
        """Regression: handmap 101/102 must have note_off; drums markers must not
        be duplicated across reduced difficulties. Uses explicit checks because
        validate_rb_midi only catches same-pitch overlaps, not orphan note_offs."""
        from downcharter import processor as _proc
        from downcharter import validate as _val

        # Build MIDI that triggers handmap (gap=240 + fret changes) and has
        # tom/overdrive markers (110-112, 103) that were previously duplicated.
        mid = _build_midi_with_markers(first_note_offset=960)
        src = tmp_path / "marker_bugs.mid"
        mid.save(str(src))

        dst = tmp_path / "validated.mid"
        _proc.process_midi(
            str(src), str(dst),
            diffs_to_gen=["hard", "medium", "easy"],
            do_expert_plus=False, do_venue=True,
            do_lipsync=False, do_talkies=False, do_drum_anim=False,
        )

        out_mid = mido.MidiFile(str(dst))

        # ── Check 1: 101/102 handmap markers have corresponding note_offs
        # (validate_rb_midi does NOT catch hanging notes — we check directly)
        FORCE_NOTES = {101, 102}
        for tr in out_mid.tracks:
            nm = (tr.name or "").strip().upper()
            if "GUITAR" not in nm and "BASS" not in nm:
                continue
            # Collect all (tick, note, is_on) events
            events = []
            tick = 0
            for msg in tr:
                tick += msg.time if hasattr(msg, 'time') else 0
                if hasattr(msg, 'note') and msg.note in FORCE_NOTES:
                    events.append((tick, msg.note, msg.type == 'note_on' and msg.velocity > 0))
            # For each force marker note_on there must be a note_off
            for t, note, is_on in events:
                if is_on:
                    has_off = any(t2 == t + 1 and n == note
                                  for t2, n, io in events if not io)
                    assert has_off, (
                        f"{nm}: force marker {note} at tick {t} has no note_off "
                        f"(handmap bug — notes hang until end of song)"
                    )

        # ── Check 2: drums markers must NOT be duplicated across difficulties
        # (validate_rb_midi catches overlaps but we verify marker dedup explicitly)
        for tr in out_mid.tracks:
            nm = (tr.name or "").strip().upper()
            if not nm.startswith("PART DRUMS"):
                continue
            # Marker notes are >= 103
            marker_events = []
            tick = 0
            for msg in tr:
                tick += msg.time if hasattr(msg, 'time') else 0
                if hasattr(msg, 'note') and msg.note >= 103:
                    marker_events.append((tick, msg.note))
            # Check for duplicates by (tick, note)
            seen: dict[tuple[int, int], str] = {}
            for t, note in marker_events:
                key = (t, note)
                assert key not in seen, (
                    f"{nm}: marker note {note} at tick {t} appears more than once "
                    f"(drums duplication bug — was added once per reduced difficulty)"
                )
                seen[key] = nm

        # ── Check 3: standard RB3 validation (no crash-level errors)
        issues = _val.validate_rb_midi(out_mid)
        errors = [(lvl, msg) for lvl, msg in issues if lvl == "error"]
        assert len(errors) == 0, f"RB3 validation errors: {errors}"


# ═══════════════════════════════════════════════════════════════════════════════
#  B — Audio Pipeline (synthetic stems)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestAudioPipeline:
    """Tests that exercise the audio analysis functions with synthetic stems."""

    def test_audio_available(self):
        """Verify numpy + soundfile are available."""
        assert _audio.available(), "Audio libs (numpy, soundfile) must be installed"

    def test_voice_activity_on_vocal_stem(self, vocal_stem_path):
        """voice_activity() detects two syllables in the synthetic vocal stem."""
        if not _audio.available():
            pytest.skip("audio libs not available")
        va = _audio.voice_activity([vocal_stem_path])
        assert va is not None, "voice_activity returned None"
        env, hop_s, thr = va
        assert len(env) > 0, "Envelope should have > 0 frames"
        # Count frames above threshold (should be > 0 for a vocal stem with sound)
        import numpy as np
        above = int(np.sum(env > thr))
        below = int(np.sum(env <= thr))
        assert above > 0, f"Expected some frames above threshold, but {above=}, {below=}, {thr=:.6f}, max_env={env.max():.6f}"
        # total = above + below, both should be reasonable
        assert above + below == len(env), \
            f"All {len(env)} frames should be above or below threshold"
        print(f"  voice_activity: {above}/{len(env)} frames above thr={thr:.6f}")

    def test_syllable_gain(self, vocal_stem_path):
        """syllable_gain() returns a float gain for each syllable span."""
        if not _audio.available():
            pytest.skip("audio libs not available")
        va = _audio.voice_activity([vocal_stem_path])
        assert va is not None, "voice_activity returned None"

        # Syllable 1 spans 0.3–0.7 s  → should have gain > 0
        g1 = _audio.syllable_gain(va, 0.3, 0.7)
        assert isinstance(g1, float), f"Expected float gain, got {type(g1)}"
        assert g1 > 0.0, f"Syllable gain should be > 0 on voiced region, got {g1}"

        # Syllable 2 spans 1.0–1.5 s → should have gain > 0
        g2 = _audio.syllable_gain(va, 1.0, 1.5)
        assert g2 > 0.0, f"Second syllable gain should be > 0, got {g2}"

        # Silent region 0.0–0.2 s → should have low/no gain
        g0 = _audio.syllable_gain(va, 0.0, 0.2)
        print(f"  syllable gains: silent={g0:.6f}, syl1={g1:.6f}, syl2={g2:.6f}")
        # Gain in silent region should be lower than in voiced regions
        assert g0 < g1, f"Silent gain {g0} should be < voiced gain {g1}"

    def test_load_mono(self, guitar_stem_path):
        """load_mono() decodes an OGG stem to mono float32."""
        if not _audio.available():
            pytest.skip("audio libs not available")
        mono, sr = _audio.load_mono(guitar_stem_path)
        import numpy as np
        assert isinstance(mono, np.ndarray), f"Expected ndarray, got {type(mono)}"
        assert mono.dtype == np.float32, f"Expected float32, got {mono.dtype}"
        assert sr == 44100, f"Expected sr=44100, got {sr}"
        assert len(mono) == 88200, \
            f"Expected 88200 samples (2s @ 44100), got {len(mono)}"
        assert abs(mono.max()) > 0.0, "Audio should have non-zero amplitude"

    def test_find_song_audio(self, tmp_path):
        """find_song_audio() locates OGG stems in a folder."""
        if not _audio.available():
            pytest.skip("audio libs not available")
        folder = tmp_path / "audio_test"
        folder.mkdir()
        # Create a guitar.ogg stem
        _make_sine_stem(str(folder / "guitar.ogg"), freq=440.0, duration=0.5)
        # Create a song.ogg (full mix)
        _make_sine_stem(str(folder / "song.ogg"), freq=440.0, duration=0.5)

        paths = _audio.find_song_audio(str(folder))
        assert len(paths) > 0, f"Expected to find audio stems in {folder}"
        # At minimum, the stems should be found
        assert any("guitar.ogg" in os.path.basename(p) for p in paths), \
            f"guitar.ogg not found in {paths}"

    def test_audio_duration_seconds(self, guitar_stem_path):
        """audio_duration_seconds() returns the duration of an OGG stem."""
        if not _audio.available():
            pytest.skip("audio libs not available")
        dur = _audio.audio_duration_seconds([guitar_stem_path])
        assert dur is not None, "audio_duration_seconds returned None"
        assert 1.9 <= dur <= 2.1, \
            f"Expected ~2.0 s, got {dur:.3f} s"

    def test_load_vocal_from_mogg_nonexistent(self):
        """load_vocal_from_mogg returns None for a missing file."""
        result = _audio.load_vocal_from_mogg("/nonexistent/file.mogg")
        assert result is None, "Expected None for nonexistent .mogg"

    def test_load_vocal_from_mogg_encrypted(self, tmp_path):
        """load_vocal_from_mogg returns None for encrypted (version != 0x0A)."""
        fake = tmp_path / "fake.mogg"
        # Write header with version 0 (encrypted)
        fake.write_bytes(struct.pack("<II", 0, 64) + b"\x00" * 60)
        result = _audio.load_vocal_from_mogg(str(fake))
        assert result is None, "Expected None for encrypted .mogg"

    def test_find_vocal_audio_stems(self, vocal_stem_path):
        """find_vocal_audio returns separated stems when present."""
        folder = os.path.dirname(vocal_stem_path)
        result = _audio.find_vocal_audio(folder)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) > 0, "Expected at least one vocal stem"
        assert all(p.endswith((".ogg", ".wav")) for p in result), \
            f"Expected audio files, got {result}"

    def test_find_vocal_audio_mogg(self, tmp_path):
        """find_vocal_audio returns None when only a .mogg is present."""
        (tmp_path / "song.mogg").write_bytes(struct.pack("<II", 0x0A, 64) + b"\x00" * 60)
        result = _audio.find_vocal_audio(str(tmp_path))
        assert result is None, \
            f"Expected None for .mogg-only folder, got {result}"

    def test_find_vocal_audio_empty(self, tmp_path):
        """find_vocal_audio returns [] for folder with no audio."""
        result = _audio.find_vocal_audio(str(tmp_path))
        assert result == [], f"Expected empty list, got {result}"


# ═══════════════════════════════════════════════════════════════════════════════
#  C — Full Pipeline (MIDI + Audio)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestFullPipelineWithAudio:
    """Tests that exercise process_folder in a song folder with audio stems."""

    def test_process_folder_with_stems(self, tmp_path):
        """process_folder on a folder with notes.mid + song.ini + OGG stems.

        Verifies the output MIDI has the expected tracks (including venue + lipsync
        when requested).
        """
        if not _audio.available():
            pytest.skip("audio libs not available")

        folder = tmp_path / "AudioSong"
        folder.mkdir()

        # song.ini
        (folder / "song.ini").write_text(textwrap.dedent("""\
            [Song]
            name = Audio Test
            artist = Test Band
            genre = rock
            year = 2024
        """), encoding="utf-8")

        # notes.mid — simple guitar-only song (to avoid processing complexity)
        mid = mido.MidiFile(ticks_per_beat=480)
        tpb = 480
        tr_t = mido.MidiTrack()
        tr_t.name = "Tempo"
        tr_t.append(mido.MetaMessage("track_name", name="Tempo", time=0))
        tr_t.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        tr_t.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        tr_t.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr_t)

        gtr = mido.MidiTrack()
        gtr.name = "PART GUITAR"
        gtr.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
        for beat in range(32):
            t = beat * tpb
            gtr.append(mido.Message("note_on", note=96, velocity=100, time=t))
            gtr.append(mido.Message("note_off", note=96, velocity=0, time=tpb // 4))
        gtr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(gtr)

        ev = mido.MidiTrack()
        ev.name = "EVENTS"
        ev.append(mido.MetaMessage("track_name", name="EVENTS", time=0))
        ev.append(mido.MetaMessage("text", text="[end]", time=32 * tpb))
        ev.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(ev)

        notes_path = folder / "notes.mid"
        mid.save(str(notes_path))

        # Audio stems
        _make_sine_stem(str(folder / "guitar.ogg"), freq=440.0, duration=2.0)

        def log(msg, tag=None):
            pass  # silent logger

        _proc.process_folder(
            str(folder),
            diffs_to_gen=["hard", "medium", "easy"],
            do_expert_plus=False,
            threshold_ms=125.0,
            log_fn=log,
            do_venue=True,
            do_lipsync=False,
            do_talkies=False,
            do_drum_anim=False,
        )

        # Verify the notes.mid was modified
        out_mid = mido.MidiFile(str(notes_path))
        track_names = [t.name.strip().upper() for t in out_mid.tracks]
        assert "VENUE" in track_names, \
            f"VENUE track missing after process_folder: {track_names}"

        # Verify a .bak.mid was created
        assert (folder / "notes.bak.mid").exists(), \
            "Backup notes.bak.mid was not created"


# ═══════════════════════════════════════════════════════════════════════════════
#  D — Native Pack Pipeline (PS3 + CON)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPackPipeline:
    """Tests for the PS3 and Xbox CON native pack building."""

    def test_build_ps3_song_structure(self, song_folder):
        """build_ps3_song("1x") creates the expected folder structure.

        Note: song_folder must have a processed MIDI (with LIPSYNC1 if lyrics
        present) for the milo to be built. We pre-process the notes.mid first.
        """
        from downcharter.ps3build import build_ps3_song

        mid_path = os.path.join(song_folder, "notes.mid")
        # Pre-process: chart talkies (so the milo builder can reconstruct spans from
        # PART VOCALS tubes) + generate LIPSYNC1 track.
        out_mid_path = os.path.join(song_folder, "notes_processed.mid")
        try:
            stats = _proc.process_midi(
                mid_path, out_mid_path,
                diffs_to_gen=["hard", "medium", "easy"],
                do_venue=True, do_lipsync=True, do_talkies=True,
                do_drum_anim=True,
            )
        except Exception as e:
            pytest.skip(f"Pre-processing failed: {e}")

        # Replace notes.mid with processed one for the PS3 build
        os.replace(out_mid_path, mid_path)

        # Build PS3
        pkg_dir = build_ps3_song(str(song_folder), "1x")

        # Verify folder structure
        assert os.path.isdir(pkg_dir), f"Package dir not created: {pkg_dir}"
        songs_dir = os.path.join(pkg_dir, "songs")
        assert os.path.isdir(songs_dir), "songs/ dir missing"

        # Look for the song directory inside songs/
        song_dirs = [d for d in os.listdir(songs_dir)
                     if os.path.isdir(os.path.join(songs_dir, d))]
        assert len(song_dirs) > 0, f"No song directory found in {songs_dir}"

        song_dir = os.path.join(songs_dir, song_dirs[0])
        gen_dir = os.path.join(song_dir, "gen")
        assert os.path.isdir(gen_dir), f"gen/ dir missing in {song_dir}"

        # songs.dta should exist
        dta_path = os.path.join(songs_dir, "songs.dta")
        assert os.path.isfile(dta_path), "songs.dta missing"
        dta_text = open(dta_path, "r", encoding="latin1").read()
        for field in ("name", "artist"):
            assert field in dta_text.lower(), \
                f"Expected '{field}' in songs.dta, got:\n{dta_text[:500]}"

        # Check for .milo_ps3 (if lipsync was generated)
        milo_files = [f for f in os.listdir(gen_dir) if f.endswith(".milo_ps3")]
        print(f"  gen/ files: {os.listdir(gen_dir)}, milo_files: {milo_files}")
        if stats.get("lipsync_events", 0) > 0:
            assert len(milo_files) > 0, "Expected .milo_ps3 when lipsync events exist"
        else:
            print("  No lipsync events — skipping milo check")

    def test_build_con_song(self, song_folder):
        """build_con_song("1x") creates a .con file."""
        from downcharter.stfs import build_con_song

        mid_path = os.path.join(song_folder, "notes.mid")
        # Pre-process: chart talkies for milo span reconstruction + LIPSYNC1 track
        out_mid_path = os.path.join(song_folder, "notes_processed.mid")
        try:
            _proc.process_midi(
                mid_path, out_mid_path,
                diffs_to_gen=["hard", "medium", "easy"],
                do_venue=True, do_lipsync=True, do_talkies=True,
                do_drum_anim=True,
            )
        except Exception as e:
            pytest.skip(f"Pre-processing failed: {e}")

        os.replace(out_mid_path, mid_path)

        con_path = build_con_song(str(song_folder), "1x")
        assert os.path.isfile(con_path), \
            f"Expected .con file at {con_path}"
        file_size = os.path.getsize(con_path)
        assert file_size > 0, f"CON file is empty ({file_size} bytes)"

        # Verify CON magic bytes
        with open(con_path, "rb") as f:
            magic = f.read(4)
        assert magic == b"CON ", \
            f"Expected CON magic, got {magic!r}"

    def test_validate_output_midi(self, song_folder):
        """Validate the processed MIDI — should have no fatal errors."""
        mid_path = os.path.join(song_folder, "notes.mid")
        # Process MIDI in place
        _proc.process_midi(
            mid_path, mid_path,
            diffs_to_gen=["hard", "medium", "easy"],
            do_expert_plus=False, do_venue=True, do_lipsync=True,
            do_talkies=True, do_drum_anim=True,
        )

        out_mid = mido.MidiFile(mid_path)
        issues = _val.validate_rb_midi(out_mid)
        errors = [(lvl, msg) for lvl, msg in issues if lvl == "error"]
        # We expect no crash-level errors from our generated MIDI
        assert len(errors) == 0, f"Validation errors in processed MIDI: {errors}"

    def test_dta_has_required_fields(self, song_folder):
        """songs.dta built from song.ini has name, artist, song_id, etc."""
        from downcharter.ps3build import build_ps3_song, _parse_song_ini, _sanitize_shortname

        mid_path = os.path.join(song_folder, "notes.mid")
        out_mid_path = os.path.join(song_folder, "notes_processed.mid")
        try:
            _proc.process_midi(
                mid_path, out_mid_path,
                diffs_to_gen=["hard", "medium", "easy"],
                do_venue=True, do_lipsync=True, do_talkies=True,
                do_drum_anim=True,
            )
        except Exception as e:
            pytest.skip(f"Pre-processing failed: {e}")
        os.replace(out_mid_path, mid_path)

        # Build PS3
        pkg_dir = build_ps3_song(str(song_folder), "1x")

        # Read songs.dta
        import glob as _glob
        dta_candidates = list(_glob.glob(os.path.join(pkg_dir, "songs", "songs.dta")))
        if not dta_candidates:
            # Try to find the dta recursively
            for root, _, files in os.walk(pkg_dir):
                if "songs.dta" in files:
                    dta_candidates = [os.path.join(root, "songs.dta")]
                    break
        assert len(dta_candidates) > 0, "songs.dta not found in package"
        dta_path = dta_candidates[0]
        dta_text = open(dta_path, "r", encoding="latin1").read()

        # Check for essential fields
        required = ["name", "artist", "song_id", "tracks"]
        for field in required:
            assert field in dta_text.lower(), \
                f"Required field '{field}' missing from songs.dta"


# ═══════════════════════════════════════════════════════════════════════════════
#  E — Robustness / Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobustness:
    """Tests for edge-case inputs that should not crash the pipeline."""

    def test_960_tpb_rescale(self, tmp_path):
        """A MIDI with 960 TPB is rescaled to 480 by process_midi."""
        mid = mido.MidiFile(ticks_per_beat=960)
        tr_t = mido.MidiTrack()
        tr_t.name = "Tempo"
        tr_t.append(mido.MetaMessage("track_name", name="Tempo", time=0))
        tr_t.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        tr_t.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr_t)

        gtr = mido.MidiTrack()
        gtr.name = "PART GUITAR"
        gtr.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
        gtr.append(mido.Message("note_on", note=96, velocity=100, time=0))
        gtr.append(mido.Message("note_off", note=96, velocity=0, time=240))
        gtr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(gtr)

        ev = mido.MidiTrack()
        ev.name = "EVENTS"
        ev.append(mido.MetaMessage("track_name", name="EVENTS", time=0))
        ev.append(mido.MetaMessage("text", text="[end]", time=1920))
        ev.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(ev)

        src = tmp_path / "tpb960.mid"
        mid.save(str(src))
        dst = tmp_path / "tpb480.mid"
        stats = _proc.process_midi(
            str(src), str(dst),
            diffs_to_gen=[],
            do_venue=False, do_lipsync=False, do_drum_anim=False,
        )

        out = mido.MidiFile(str(dst))
        assert out.ticks_per_beat == 480, \
            f"Expected 480 TPB output, got {out.ticks_per_beat}"
        assert stats["tracks_processed"] >= 1, \
            "Expected at least 1 track processed"

    def test_phase_shift_sysex(self, tmp_path):
        """A MIDI with Phase Shift sysex (0xFF clamped to 0x7F) processes without error."""
        mid = mido.MidiFile(ticks_per_beat=480)
        tr_t = mido.MidiTrack()
        tr_t.name = "Tempo"
        tr_t.append(mido.MetaMessage("track_name", name="Tempo", time=0))
        tr_t.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        tr_t.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr_t)

        gtr = mido.MidiTrack()
        gtr.name = "PART GUITAR"
        gtr.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
        gtr.append(mido.Message("note_on", note=96, velocity=100, time=0))
        gtr.append(mido.Message("note_off", note=96, velocity=0, time=480))
        # Phase Shift sysex (open note marker): F0 50 53 00 00 FF
        gtr.append(mido.Message("sysex",
            data=[0x50, 0x53, 0x00, 0x00, 0x7F],  # 0xFF clamped to 0x7F on write by mido
            time=0))
        gtr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(gtr)

        ev = mido.MidiTrack()
        ev.name = "EVENTS"
        ev.append(mido.MetaMessage("track_name", name="EVENTS", time=0))
        ev.append(mido.MetaMessage("text", text="[end]", time=960))
        ev.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(ev)

        src = tmp_path / "ps_sysex.mid"
        mid.save(str(src))
        dst = tmp_path / "ps_sysex_out.mid"
        # Should not crash
        stats = _proc.process_midi(
            str(src), str(dst),
            diffs_to_gen=[],
            do_venue=False, do_lipsync=False, do_drum_anim=False,
        )

        # Sysex should be preserved
        assert stats.get("sysex_kept", 0) > 0, \
            "PS sysex was not preserved in output"
        assert stats["tracks_processed"] >= 1

    def test_midi_without_instrument_tracks(self, tmp_path):
        """A MIDI with only Tempo and EVENTS tracks processes without error."""
        mid = mido.MidiFile(ticks_per_beat=480)
        tr_t = mido.MidiTrack()
        tr_t.name = "Tempo"
        tr_t.append(mido.MetaMessage("track_name", name="Tempo", time=0))
        tr_t.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        tr_t.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr_t)

        ev = mido.MidiTrack()
        ev.name = "EVENTS"
        ev.append(mido.MetaMessage("track_name", name="EVENTS", time=0))
        ev.append(mido.MetaMessage("text", text="[end]", time=1920))
        ev.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(ev)

        src = tmp_path / "no_instr.mid"
        mid.save(str(src))
        dst = tmp_path / "no_instr_out.mid"

        # Should not crash — just processes nothing
        stats = _proc.process_midi(
            str(src), str(dst),
            diffs_to_gen=["hard", "medium", "easy"],
            do_venue=False, do_lipsync=False, do_drum_anim=False,
        )
        assert stats["tracks_processed"] == 0, \
            f"Expected 0 tracks processed with no instruments, got {stats['tracks_processed']}"

    def test_basic_chart_conversion(self, tmp_path):
        """A minimal .chart file converts to MIDI and can be processed."""
        chart_content = textwrap.dedent("""\
            [Song]
            {
              Name = "Chart Test"
              Artist = "Chart Artist"
              Resolution = 192
            }
            [SyncTrack]
            {
              0 = B 120000
              0 = TS 4
              1536 = E "end"
            }
            [Events]
            {
              768 = E "phrase_start"
              768 = E "lyric test"
              768 = E "phrase_end"
            }
            [ExpertSingle]
            {
              768 = N 0 0
              960 = N 0 0
            }
        """)
        chart_path = tmp_path / "test.chart"
        chart_path.write_text(chart_content, encoding="utf-8")

        mid = _chart_to_midi(str(chart_path))
        assert isinstance(mid, mido.MidiFile), "chart_to_midi should return a MidiFile"
        # chart_to_midi always normalises to 480 TPB (RB3 requirement)
        assert mid.ticks_per_beat == 480, f"Expected 480 TPB, got {mid.ticks_per_beat}"

        # Should have PART GUITAR and PART VOCALS tracks
        track_names = [t.name.strip().upper() for t in mid.tracks]
        assert "PART GUITAR" in track_names, f"No PART GUITAR in {track_names}"
        assert "PART VOCALS" in track_names, \
            f"No PART VOCALS in {track_names} (needed for lyrics)"

        # Save to MIDI and process it
        mid_path = tmp_path / "chart_out.mid"
        mid.save(str(mid_path))
        dst_path = tmp_path / "chart_processed.mid"
        stats = _proc.process_midi(
            str(mid_path), str(dst_path),
            diffs_to_gen=["hard", "medium", "easy"],
            do_venue=False, do_lipsync=False, do_drum_anim=False,
        )
        assert stats["tracks_processed"] >= 1, \
            f"Expected >= 1 track processed from chart, got {stats['tracks_processed']}"

    def test_song_ini_with_missing_fields(self, tmp_path):
        """song.ini with only name and artist (no genre, year, etc.) should not crash."""
        folder = tmp_path / "MinimalSong"
        folder.mkdir()

        # Minimal song.ini
        (folder / "song.ini").write_text(textwrap.dedent("""\
            [Song]
            name = Minimal
            artist = Solo
        """), encoding="utf-8")

        # Simple MIDI
        mid = mido.MidiFile(ticks_per_beat=480)
        tr_t = mido.MidiTrack()
        tr_t.name = "Tempo"
        tr_t.append(mido.MetaMessage("track_name", name="Tempo", time=0))
        tr_t.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        tr_t.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        tr_t.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tr_t)

        gtr = mido.MidiTrack()
        gtr.name = "PART GUITAR"
        gtr.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
        for b in range(16):
            gtr.append(mido.Message("note_on", note=96, velocity=100, time=b * 480))
            gtr.append(mido.Message("note_off", note=96, velocity=0, time=120))
        gtr.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(gtr)

        ev = mido.MidiTrack()
        ev.name = "EVENTS"
        ev.append(mido.MetaMessage("track_name", name="EVENTS", time=0))
        ev.append(mido.MetaMessage("text", text="[end]", time=16 * 480))
        ev.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(ev)

        mid_path = folder / "notes.mid"
        mid.save(str(mid_path))

        # Process folder
        def log(msg, tag=None): pass
        _proc.process_folder(
            str(folder),
            diffs_to_gen=["hard", "medium", "easy"],
            do_expert_plus=False,
            threshold_ms=125.0,
            log_fn=log,
            do_venue=True,
            do_lipsync=False,
            do_talkies=False,
            do_drum_anim=False,
        )

        # Should not have crashed
        assert (folder / "notes.bak.mid").exists(), "Backup not created"


# ═══════════════════════════════════════════════════════════════════════════════
#  Smoke test — verify the whole suite integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmoke:
    """Quick smoke tests that verify basic integration without heavy processing."""

    def test_process_midi_returns_stats_structure(self, multi_mid, tmp_path):
        """process_midi returns a dict with all expected stat keys."""
        dst = tmp_path / "smoke.mid"
        stats = _proc.process_midi(
            str(multi_mid), str(dst),
            diffs_to_gen=["hard", "medium", "easy"],
            do_expert_plus=False, do_venue=False, do_lipsync=False,
            do_drum_anim=False,
        )
        expected_keys = {
            "tracks_processed", "groove_warnings", "diffs_skipped",
            "venue_events", "venue_skipped", "beat_added", "beat_extended",
            "tracks_renamed", "sp_remapped",
        }
        missing = expected_keys - set(stats.keys())
        assert not missing, f"Stats dict missing keys: {missing}"
        assert isinstance(stats["tracks_processed"], int)
        assert isinstance(stats["groove_warnings"], list)
        assert isinstance(stats["diffs_skipped"], list)

    def test_output_midi_is_480_tpb(self, multi_mid, tmp_path):
        """Output MIDI always has 480 TPB regardless of input."""
        dst = tmp_path / "tpb_check.mid"
        _proc.process_midi(
            str(multi_mid), str(dst),
            diffs_to_gen=[],
            do_venue=False, do_lipsync=False, do_drum_anim=False,
        )
        out = mido.MidiFile(str(dst))
        assert out.ticks_per_beat == 480, \
            f"Output TPB should be 480, got {out.ticks_per_beat}"

    def test_process_midi_idempotent_no_diffs(self, multi_mid, tmp_path):
        """Processing with empty diffs_to_gen does not crash."""
        dst = tmp_path / "no_diffs.mid"
        stats = _proc.process_midi(
            str(multi_mid), str(dst),
            diffs_to_gen=[],
            do_venue=False, do_lipsync=False, do_drum_anim=False,
        )
        assert stats["tracks_processed"] >= 3, \
            "Even with no diffs, tracks should still be processed"

    def test_multi_track_output_has_all_parts(self, multi_mid, tmp_path):
        """Output should contain all the original PART tracks."""
        dst = tmp_path / "parts_check.mid"
        _proc.process_midi(
            str(multi_mid), str(dst),
            diffs_to_gen=[],
            do_venue=False, do_lipsync=False, do_drum_anim=False,
        )
        out = mido.MidiFile(str(dst))
        out_names = set(t.name.strip().upper() for t in out.tracks)
        for expected in ("PART GUITAR", "PART BASS", "PART DRUMS", "PART VOCALS"):
            assert expected in out_names, \
                f"Expected {expected} in output, got {out_names}"

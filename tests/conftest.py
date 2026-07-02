"""Shared fixtures for Downcharter+ tests."""
import sys
import os

# Ensure the project root is importable
_proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

import mido
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: tests that process audio (synthetic OGG/WAV stems)"
    )


@pytest.fixture
def minimal_mid() -> mido.MidiFile:
    """A minimal 480 TPB MIDI with one empty track."""
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    tr.name = "Test Track"
    tr.append(mido.MetaMessage("track_name", name="Test Track", time=0))
    tr.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(tr)
    return mid


@pytest.fixture
def simple_song_mid() -> mido.MidiFile:
    """A 480 TPB MIDI with a PART GUITAR 4/4 song, 8 bars at 120 BPM."""
    mid = mido.MidiFile(ticks_per_beat=480)
    tpb = 480

    # Tempo map: 120 BPM = 500000 µs/beat
    tempo_tr = mido.MidiTrack()
    tempo_tr.name = "Tempo"
    tempo_tr.append(mido.MetaMessage("track_name", name="Tempo", time=0))
    tempo_tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    tempo_tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    tempo_tr.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(tempo_tr)

    # PART GUITAR with Expert gems at regular intervals
    gtr = mido.MidiTrack()
    gtr.name = "PART GUITAR"
    gtr.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
    # One chord per beat for 8 bars = 32 notes
    for beat in range(32):
        tick = beat * tpb
        gtr.append(mido.Message("note_on", note=96, velocity=100, time=tick))
        gtr.append(mido.Message("note_off", note=96, velocity=0, time=tpb // 4))
    gtr.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(gtr)

    # EVENTS
    ev = mido.MidiTrack()
    ev.name = "EVENTS"
    ev.append(mido.MetaMessage("track_name", name="EVENTS", time=0))
    ev.append(mido.MetaMessage("text", text="[end]", time=32 * tpb))
    ev.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(ev)

    return mid


@pytest.fixture
def drum_mid() -> mido.MidiFile:
    """MIDI with a PART DRUMS track carrying Expert kicks and snares."""
    mid = mido.MidiFile(ticks_per_beat=480)
    tpb = 480

    tempo_tr = mido.MidiTrack()
    tempo_tr.name = "Tempo"
    tempo_tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    tempo_tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    tempo_tr.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(tempo_tr)

    drums = mido.MidiTrack()
    drums.name = "PART DRUMS"
    for beat in range(16):
        tick = beat * tpb
        # Kick (96) on beats 1 and 3
        if beat % 2 == 0:
            drums.append(mido.Message("note_on", note=96, velocity=100, time=tick))
            drums.append(mido.Message("note_off", note=96, velocity=0, time=tpb // 8))
        # Snare (97) on beats 2 and 4
        if beat % 2 == 1:
            drums.append(mido.Message("note_on", note=97, velocity=100, time=tick))
            drums.append(mido.Message("note_off", note=97, velocity=0, time=tpb // 8))
    drums.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(drums)

    ev = mido.MidiTrack()
    ev.name = "EVENTS"
    ev.append(mido.MetaMessage("text", text="[end]", time=16 * tpb))
    ev.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(ev)

    return mid

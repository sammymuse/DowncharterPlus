"""Tests for downcharter/mogg.py — first-audio detection (_first_audio_s).

The per-frame Python loop was vectorised in b3b849b; the chunked rewrite must
keep the exact semantics of the original loop (first frame whose peak |sample|
exceeds the threshold) with early exit and bounded memory.
"""
import numpy as np

from downcharter.mogg import _first_audio_s


def _sig(n_frames: int, ch: int = 2) -> np.ndarray:
    return np.zeros((n_frames, ch), dtype=np.float32)


def test_first_audio_at_known_frame():
    sr = 44100
    data = _sig(sr * 3)
    data[sr + 137, 1] = 0.5          # spike on channel 1 only
    t = _first_audio_s(data, sr)
    assert t == (sr + 137) / sr


def test_first_audio_all_silence_returns_none():
    assert _first_audio_s(_sig(44100), 44100) is None


def test_first_audio_below_threshold_ignored():
    sr = 44100
    data = _sig(sr)
    data[10, 0] = 0.009              # under the 0.01 default threshold
    data[500, 0] = 0.02
    assert _first_audio_s(data, sr) == 500 / sr


def test_first_audio_crosses_chunk_boundary():
    """Spike beyond the first chunk: the chunked scan must keep the ABSOLUTE
    frame index, not the chunk-relative one."""
    sr = 1000
    chunk = 256
    data = _sig(chunk * 3 + 5)
    data[chunk * 2 + 3, 0] = 1.0
    t = _first_audio_s(data, sr, chunk_frames=chunk)
    assert t == (chunk * 2 + 3) / sr


def test_first_audio_empty_input():
    assert _first_audio_s(_sig(0), 44100) is None


def test_first_audio_first_frame():
    data = _sig(100)
    data[0, 0] = 0.5
    assert _first_audio_s(data, 44100) == 0.0

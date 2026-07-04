"""Tests for separate.py — vocal separation via MDX-NET ONNX.

Uses a fake session (mask=1.0 → identity) so tests run without the real model.
"""

import threading
import numpy as np
import pytest

from downcharter.separate import (
    _stft,
    _istft,
    _hann_window,
    _make_session,
    separate_vocals,
    _MODEL_CONFIG,
)


# ══════════════════════════════════════════════════════════════════════
#  Fake ONNX session (injectable, mask=1.0 → identity)
# ══════════════════════════════════════════════════════════════════════

class FakeSession:
    """Emulates an InferenceSession: returns the input multiplied by *mask*."""

    def __init__(self, mask: float = 1.0):
        self.mask = mask

    def run(self, _, feed_dict: dict) -> list[np.ndarray]:
        inp = feed_dict["input"]
        return [inp * self.mask]

    def get_providers(self):
        return ["FakeProvider"]


def _fake_session(model_path, force_cpu=False):
    return FakeSession(mask=1.0), "FakeProvider"


# Monkey-patch _make_session and _lazy_import_ort for tests
@pytest.fixture(autouse=True)
def patch_ort(monkeypatch):
    monkeypatch.setattr("downcharter.separate._make_session", _fake_session)
    monkeypatch.setattr("downcharter.separate._lazy_import_ort", lambda: True)
    monkeypatch.setattr("downcharter.separate._ort", True)  # avoid import attempt


# ══════════════════════════════════════════════════════════════════════
#  STFT / iSTFT round-trip
# ══════════════════════════════════════════════════════════════════════

class TestSTFT:

    def test_roundtrip_identity(self):
        """STFT → iSTFT recovers the input (within numerical precision).

        Uses a signal length that is a multiple of hop to avoid edge
        truncation (the STFT padding centres the window; reconstruction
        loses up to hop-1 samples at the trailing edge).
        """
        n_fft, hop = 256, 64
        window = _hann_window(n_fft).astype(np.float32)
        sr = 44100
        # Length must be multiple of hop to avoid edge truncation
        n_samples = 22016  # 344 × 64
        t = np.linspace(0, n_samples / sr, n_samples, endpoint=False)
        sig = np.stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.3 * np.sin(2 * np.pi * 880 * t + 0.5),
        ], axis=0).astype(np.float32)

        S = _stft(sig, n_fft, hop, window)
        rec = _istft(S, n_fft, hop, window, length=sig.shape[1])

        assert rec.shape == sig.shape, (
            f"Shape mismatch: {rec.shape} vs {sig.shape}")
        err = np.max(np.abs(sig - rec))
        assert err < 5e-5, f"STFT round-trip error too large: {err:.2e}"

    def test_hann_window(self):
        w = _hann_window(256)
        assert w.shape == (256,)
        assert np.allclose(w[0], 0.0)
        # Symmetric hann(N) peaks between N/2-1 and N/2 for even N
        assert np.allclose(w[127], 1.0, atol=0.02)
        assert np.allclose(w[-1], 0.0)


# ══════════════════════════════════════════════════════════════════════
#  separate_vocals — basic function tests
# ══════════════════════════════════════════════════════════════════════

class TestSeparateVocals:

    def test_returns_vocals_with_stereo_input(self):
        """Given a stereo mix, should return stereo vocals (shapes match)."""
        sr = 44100
        # Use a length that's a multiple of hop for clean chunking
        dur_samples = 261120  # exactly chunk_size — 1 chunk
        t = np.linspace(0, dur_samples / sr, dur_samples, endpoint=False)
        mix = np.stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.3 * np.sin(2 * np.pi * 880 * t),
        ], axis=0).astype(np.float32)

        out = separate_vocals(mix, sr, model_config=_MODEL_CONFIG)
        assert out is not None, f"separate_vocals returned None"
        assert out.shape == mix.shape, f"Shape mismatch: {out.shape} vs {mix.shape}"
        assert out.dtype == np.float32

    def test_mono_input_fails(self):
        """Single-channel input should return None."""
        sr = 44100
        sig = np.zeros((1, sr), dtype=np.float32)
        out = separate_vocals(sig, sr)
        assert out is None

    def test_bad_sample_rate_fails(self):
        """Non-44100 sample rate returns None (caller must resample)."""
        mix = np.zeros((2, 44100), dtype=np.float32)
        out = separate_vocals(mix, 48000)
        assert out is None

    def test_cancel_stops_early(self):
        """Setting the cancel event before processing returns None immediately."""
        sr = 44100
        dur = 3.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        mix = np.stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.3 * np.sin(2 * np.pi * 880 * t),
        ], axis=0).astype(np.float32)

        cancel = threading.Event()
        cancel.set()  # cancel BEFORE calling separate_vocals
        out = separate_vocals(mix, sr, model_config=_MODEL_CONFIG, cancel=cancel)
        assert out is None, "Should return None when cancelled"

    def test_progress_callback(self):
        """Progress callback receives values from 0 to 1."""
        sr = 44100
        dur = 2.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        mix = np.stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.3 * np.sin(2 * np.pi * 880 * t),
        ], axis=0).astype(np.float32)

        progress_values = []

        def cb(frac):
            progress_values.append(frac)

        out = separate_vocals(mix, sr, model_config=_MODEL_CONFIG, progress=cb)
        assert out is not None
        assert len(progress_values) > 0
        assert 0 < progress_values[-1] <= 1.0

    def test_short_signal(self):
        """Very short signal (< 1 chunk) still works."""
        sr = 44100
        sig = np.zeros((2, 2048), dtype=np.float32)
        out = separate_vocals(sig, sr, model_config=_MODEL_CONFIG)
        assert out is not None
        assert out.shape == (2, 2048)


# ══════════════════════════════════════════════════════════════════════
#  Model config loading
# ══════════════════════════════════════════════════════════════════════

class TestModelConfig:

    def test_default_config(self):
        from downcharter.separate import load_model_config, _MODEL_CONFIG
        # Path to a nonexistent file → returns default
        cfg = load_model_config("/nonexistent/path.json")
        assert cfg["n_fft"] == _MODEL_CONFIG["n_fft"]
        assert cfg["hop"] == _MODEL_CONFIG["hop"]
        assert cfg["dim_f"] == _MODEL_CONFIG["dim_f"]
        assert cfg["dim_t"] == _MODEL_CONFIG["dim_t"]

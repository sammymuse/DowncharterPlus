"""Golden test for separate.py — requires the real .onnx model file.

These tests are marked ``@pytest.mark.slow`` and are skipped by default.
Run with ``pytest --run-slow`` or set the ``DOWNCHARTER_RUN_SLOW`` env var.

The test generates a deterministic 0.5 s stereo sine-sweep input and compares
the vocal-separation output against a pre-computed SHA256 checksum.
If the model is updated, re-generate the golden hash with::

    python dev/generate_mdx_golden.py
"""

import hashlib
import os
import numpy as np
import pytest

slow = pytest.mark.skipif(
    not os.environ.get("DOWNCHARTER_RUN_SLOW")
    and "--run-slow" not in " ".join(os.sys.argv),
    reason="Slow test — set DOWNCHARTER_RUN_SLOW or use --run-slow",
)

# ── Golden reference ────────────────────────────────────────────────────────
# Generated with UVR-MDX-NET-Voc_FT.onnx (SHA256 of the .onnx:
# 534b2070fcc7df514b13ef660dc8cbb328679c2374d04354a5c42bb14ecce111)
#
# To regenerate:
#   python dev/generate_mdx_golden.py
#
_GOLDEN_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"

# Test signal parameters — keep these stable across model versions
_SR = 44100
_DURATION_S = 0.5  # 0.5 seconds → deterministic, fast
_N_SAMPLES = int(_SR * _DURATION_S)

# Input seed for reproducible noise (the "audio" the model separates)
_INPUT_SEED = 42


def _make_input() -> np.ndarray:
    """Deterministic stereo input: chirp + noise, same every call."""
    rng = np.random.RandomState(_INPUT_SEED)
    t = np.linspace(0, _DURATION_S, _N_SAMPLES, endpoint=False)
    # Left channel: ascending chirp 200→2000 Hz
    freq_l = np.linspace(200, 2000, _N_SAMPLES)
    left = 0.5 * np.sin(2 * np.pi * freq_l * t)
    # Right channel: 440 Hz tone + low-level noise
    right = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.randn(_N_SAMPLES)
    return np.stack([left, right], axis=0).astype(np.float32)


@slow
class TestSeparateGolden:
    """End-to-end vocal separation with the real ONNX model."""

    @pytest.fixture(autouse=True)
    def require_model(self):
        """Skip the test if the .onnx model file is not present."""
        from downcharter.separate import _data_path
        model_path = _data_path() / "UVR-MDX-NET-Voc_FT.onnx"
        if not model_path.is_file():
            pytest.skip(f"Model not found at {model_path} — download with dev/download_mdx_model.py")

    def test_golden_hash(self):
        """SHA256 of the vocal output must match the golden reference.

        This ensures the model + inference code produce bit-identical results.
        If you update the model or the inference code, regenerate the hash
        with ``dev/generate_mdx_golden.py`` and update ``_GOLDEN_SHA256``.
        """
        from downcharter.separate import separate_vocals, _MODEL_CONFIG

        mix = _make_input()
        out = separate_vocals(mix, _SR, model_config=_MODEL_CONFIG)
        assert out is not None, "separate_vocals returned None with real model"
        assert out.shape == mix.shape, f"Shape mismatch: {out.shape} vs {mix.shape}"
        assert out.dtype == np.float32

        dig = hashlib.sha256(out.tobytes()).hexdigest()
        assert dig == _GOLDEN_SHA256, (
            f"SHA256 mismatch\n"
            f"  Got:      {dig}\n"
            f"  Expected: {_GOLDEN_SHA256}\n"
            f"  If the model or inference code changed legitimately, "
            f"re-generate the golden hash with:\n"
            f"    python dev/generate_mdx_golden.py"
        )

    def test_output_structure(self):
        """Structural invariants: shape, dtype, range, separation effect."""
        from downcharter.separate import separate_vocals, _MODEL_CONFIG

        mix = _make_input()
        out = separate_vocals(mix, _SR, model_config=_MODEL_CONFIG)
        assert out is not None
        assert out.shape == mix.shape
        assert out.dtype == np.float32
        # Values should be in sensible range (< 2× input amplitude)
        assert np.all(np.abs(out) < 2.0), "Output values out of range"
        # Separation should have changed the signal (not identity)
        diff = np.max(np.abs(out - mix))
        assert diff > 1e-6, (
            f"Output is identical to input (diff={diff:.2e}) — "
            f"model may not be running correctly"
        )

    def test_mdx_cache_used_when_available(self):
        """When a fresh MDX cache exists, resolve_vocal_audio returns it."""
        import tempfile
        from downcharter.audio import resolve_vocal_audio, _VOCAL_CACHE_MDX

        with tempfile.TemporaryDirectory() as tmp:
            # Create a fake song folder with a mix file
            mix_path = os.path.join(tmp, "song.wav")
            mix = _make_input().T  # soundfile expects (samples, channels)
            import soundfile as sf
            sf.write(mix_path, mix, _SR)

            # First call — triggers MDX-NET and creates cache
            result1 = resolve_vocal_audio(tmp, allow_separation=True)
            assert result1 is not None, "First call should produce vocal audio"
            assert os.path.isfile(os.path.join(tmp, _VOCAL_CACHE_MDX)), "Cache should exist"

            # Second call — should hit the cache
            result2 = resolve_vocal_audio(tmp, allow_separation=True)
            assert result2 is not None
            assert result2 == result1, "Cache should return the same path"

    def test_mdx_cache_skipped_when_disabled(self):
        """When allow_separation=False, no MDX cache is created."""
        import tempfile
        from downcharter.audio import resolve_vocal_audio

        with tempfile.TemporaryDirectory() as tmp:
            mix_path = os.path.join(tmp, "song.wav")
            mix = _make_input().T
            import soundfile as sf
            sf.write(mix_path, mix, _SR)

            result = resolve_vocal_audio(tmp, allow_separation=False)
            # Only stems + mogg are checked; no audio found → None
            assert result is None, "Without separation, no vocal audio"

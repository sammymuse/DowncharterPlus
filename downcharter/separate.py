"""
Vocal separation via MDX-NET ONNX model (UVR-MDX-NET-Voc_FT).

Embedded inference in pure numpy + onnxruntime, no torch/librosa dependency.
GPU (DirectML) when available, CPU fallback always works.

Licence:
  Model: UVR-MDX-NET-Voc_FT (TRvlvr/model_repo, redistributed permissively).
  Inference code: original, inspired by seanghay/uvr-mdx-infer (Apache-2.0).
  Full UVR ecosystem attribution in README.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import typing as t
from pathlib import Path

import numpy as np

# ── onnxruntime lazy import (soft-fail se não instalado) ──────────────

_ort = None  # None = not yet attempted
_ORT_SENTINEL = object()  # distinct sentinel for failed import

def _lazy_import_ort() -> bool:
    """Returns True if onnxruntime is importable; never raises."""
    global _ort
    if _ort is not None and _ort is not _ORT_SENTINEL:
        return True
    if _ort is _ORT_SENTINEL:
        return False
    try:
        import onnxruntime as _ort
        return True
    except ImportError:
        _ort = _ORT_SENTINEL
        return False


# ══════════════════════════════════════════════════════════════════════
#  STFT / iSTFT  —  numpy puro
# ══════════════════════════════════════════════════════════════════════

def _hann_window(n: int) -> np.ndarray:
    """Symmetric Hann window of length *n*."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def _stft(
    x: np.ndarray,
    n_fft: int,
    hop: int,
    window: np.ndarray,
) -> np.ndarray:
    """Short-time Fourier transform, numpy edition.

    Args:
        x: Input signal, shape ``(channels, samples)``.
        n_fft: FFT size (window length).
        hop: Hop length between frames.
        window: Hann (or other) window, shape ``(n_fft,)``.

    Returns:
        Complex spectrum, shape ``(channels, n_frames, n_bins)``
        where ``n_bins = n_fft // 2 + 1``.
    """
    ch, samples = x.shape
    pad = n_fft // 2
    # Reflect-pad to emulate torch.stft(center=True)
    x = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    n_frames = (x.shape[1] - n_fft) // hop + 1
    # Build frame matrix via strided sliding window
    strides = (x.strides[0], x.strides[1] * hop, x.strides[1])
    shape = (ch, n_frames, n_fft)
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)
    frames = frames * window  # broadcast over channels & frames
    # Real FFT per frame
    S = np.fft.rfft(frames, n=n_fft, axis=-1)
    return S  # complex64/128


def _istft(
    S: np.ndarray,
    n_fft: int,
    hop: int,
    window: np.ndarray,
    length: int | None = None,
) -> np.ndarray:
    """Inverse short-time Fourier transform with COLA normalization.

    Analysis window *window* is also used as synthesis window
    (matched pair). Overlap-add is normalised by the sum of squared
    window copies so that a round-trip is near-identical for signals
    longer than n_fft.

    Args:
        S: Complex spectrum, shape ``(channels, n_frames, n_bins)``.
        n_fft: FFT size.
        hop: Hop length.
        window: Window, shape ``(n_fft,)``.  Must be the SAME window
            used at analysis (STFT).
        length: If given, trim/pad output to this many samples.

    Returns:
        Time-domain signal, shape ``(channels, samples)``.
    """
    ch, n_frames, n_bins = S.shape
    # Inverse real FFT → windowed frames
    frames = np.fft.irfft(S, n=n_fft, axis=-1)  # (ch, n_frames, n_fft)
    frames = frames * window                     # analysis-synthesis windowing
    # Overlap-add
    out_len = (n_frames - 1) * hop + n_fft
    out = np.zeros((ch, out_len), dtype=np.float64)
    norm = np.zeros(out_len, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        out[:, start : start + n_fft] += frames[:, i, :]
        norm[start : start + n_fft] += window ** 2
    # Divide by the overlap-add sum of window² → COLA reconstruction
    norm = np.clip(norm, 1e-12, None)
    out /= norm
    # Remove center-padding
    pad = n_fft // 2
    out = out[:, pad : out_len - pad]
    out = out.astype(np.float32)
    if length is not None and out.shape[1] != length:
        out = out[:, :length] if out.shape[1] > length else np.pad(
            out, ((0, 0), (0, length - out.shape[1])), mode="constant")
    return out


# ══════════════════════════════════════════════════════════════════════
#  Model config helpers
# ══════════════════════════════════════════════════════════════════════

_MODEL_CONFIG: dict[str, t.Any] = {
    "n_fft": 7680,
    "hop": 1024,
    "dim_f": 3072,
    "dim_t": 256,
    "compensate": 1.02,
}


def load_model_config(config_path: str | Path) -> dict[str, t.Any]:
    """Load model parameters from JSON (beside the .onnx).

    Falls back to default ``_MODEL_CONFIG`` if file missing or invalid.
    """
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        required = {"n_fft", "hop", "dim_f", "dim_t"}
        missing = required - set(cfg.keys())
        if missing:
            raise ValueError(f"Missing keys in config: {missing}")
        return cfg
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return dict(_MODEL_CONFIG)


def _data_path() -> Path:
    """Return the directory containing model files (dev / packaged)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "downcharter" / "data"
    return Path(__file__).resolve().parent / "data"


# ══════════════════════════════════════════════════════════════════════
#  ONNX session factory
# ══════════════════════════════════════════════════════════════════════

def _make_session(
    model_path: str | Path,
    force_cpu: bool = False,
) -> tuple[t.Any, str | None]:
    """Create an ONNX InferenceSession with provider fallback.

    Priority:
        1. ``DmlExecutionProvider`` (any DX12 GPU: NVIDIA, AMD, Intel, iGPU)
        2. ``CPUExecutionProvider`` (always works)

    Args:
        model_path: Path to the ``.onnx`` file.
        force_cpu: Skip DML even if available (e.g. env var).

    Returns:
        ``(session, provider_name)``.  On failure ``(None, error_msg)``.
    """
    if not _lazy_import_ort():
        return None, "onnxruntime not installed"
    if not os.path.isfile(model_path):
        return None, f"model file not found: {model_path}"

    providers = []
    # DmlExecutionProvider is Windows-only; skip on Linux/Mac to avoid
    # the unnecessary creation failure + exception noise.
    if not force_cpu and sys.platform == "win32":
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")

    try:
        sess = _ort.InferenceSession(str(model_path), providers=providers)
        active = sess.get_providers()[0]  # first = highest priority that worked
        return sess, active
    except Exception as exc:
        # DML creation can fail on old GPUs/drivers → try CPU-only
        try:
            sess = _ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            return sess, "CPUExecutionProvider (DML failed: " + str(exc) + ")"
        except Exception as exc2:
            return None, str(exc2)


# ══════════════════════════════════════════════════════════════════════
#  Core inference
# ══════════════════════════════════════════════════════════════════════

def separate_vocals(
    samples_stereo: np.ndarray,
    sr: int,
    *,
    model_path: str | Path | None = None,
    model_config: dict[str, t.Any] | None = None,
    progress: t.Callable[[float], None] | None = None,
    cancel: threading.Event | None = None,
) -> np.ndarray | None:
    """Separate vocals from a stereo mix using MDX-NET.

    Args:
        samples_stereo: Audio samples, shape ``(2, N)``, any dtype (converted to float32).
        sr: Sample rate of the input (must be 44100 — resample beforehand).
        model_path: Path to the ``.onnx`` file.  Falls back to bundled model.
        model_config: Model parameters (see ``_MODEL_CONFIG``).  Falls back to
            ``load_model_config`` from JSON beside the model, or default.
        progress: Optional callback ``progress(frac)`` where 0 ≤ frac ≤ 1.
        cancel: Optional ``threading.Event`` — if set between chunks, return ``None``.

    Returns:
        Stereo vocal array ``(2, N)``, or ``None`` on any failure
        (logged by caller).
    """
    # ── Validate input ────────────────────────────────────────────────
    if not _lazy_import_ort():
        return None

    if samples_stereo.ndim != 2 or samples_stereo.shape[0] != 2:
        return None

    if sr != 44100:
        # Caller must resample; we don't pull in scipy/librosa
        return None

    # Ensure float32
    mix = np.asarray(samples_stereo, dtype=np.float32)
    n_samples = mix.shape[1]

    # ── Resolve model path ────────────────────────────────────────────
    if model_path is None:
        model_path = _data_path() / "UVR-MDX-NET-Voc_FT.onnx"

    # ── Load config ──────────────────────────────────────────────────
    if model_config is None:
        cfg_path = Path(str(model_path).rsplit(".", 1)[0] + ".json")
        cfg = load_model_config(cfg_path)
    else:
        cfg = model_config

    # Validate config: required keys + dim_f cannot exceed n_bins
    _REQUIRED_CFG_KEYS = {"n_fft", "hop", "dim_f", "dim_t"}
    if not _REQUIRED_CFG_KEYS.issubset(cfg):
        return None
    n_fft = cfg["n_fft"]
    hop = cfg["hop"]
    dim_f = cfg["dim_f"]
    dim_t = cfg["dim_t"]
    compensate = cfg.get("compensate", 1.02)

    n_bins = n_fft // 2 + 1  # full frequency resolution
    if dim_f > n_bins:
        return None
    chunk_size = hop * (dim_t - 1)  # samples per ONNX chunk
    trim = n_fft // 2

    # ── Create session ───────────────────────────────────────────────
    force_cpu = os.environ.get("DOWNCHARTER_FORCE_CPU", "0").strip().lower() in (
        "1", "true", "yes", "on")
    session, provider = _make_session(str(model_path), force_cpu=force_cpu)
    if session is None:
        return None

    # ── Window ────────────────────────────────────────────────────────
    window = _hann_window(n_fft).astype(np.float32)

    # ── Chunked processing with overlap-add ──────────────────────────
    gen_size = chunk_size - 2 * trim  # samples generated per chunk after trim
    n_chunks = (n_samples + gen_size - 1) // gen_size

    # Pad input: trim on both sides + pad to multiple of gen_size
    pad_samples = gen_size - (n_samples % gen_size) if n_samples % gen_size else 0
    mix_p = np.pad(mix, ((0, 0), (trim, trim + pad_samples)), mode="constant", constant_values=0)

    # Accumulator and normaliser for overlap-add
    total_len = mix_p.shape[1]
    result = np.zeros((2, total_len), dtype=np.float32)
    divider = np.zeros((2, total_len), dtype=np.float32)

    chunk_window = _hann_window(chunk_size).astype(np.float32)
    # Tile for 2 channels
    chunk_window_2ch = chunk_window[np.newaxis, :]  # (1, chunk_size)

    for ci in range(n_chunks):
        if cancel is not None and cancel.is_set():
            return None

        start = ci * gen_size
        end = start + chunk_size
        chunk = mix_p[:, start:end]

        # Pad last chunk if short
        if chunk.shape[1] < chunk_size:
            chunk = np.pad(chunk, ((0, 0), (0, chunk_size - chunk.shape[1])), mode="constant")

        # ── STFT ──────────────────────────────────────────────────────
        S = _stft(chunk, n_fft, hop, window)  # (2, dim_t, n_bins)

        # Keep only dim_f frequency bins (model input size)
        S_trimmed = S[:, :, :dim_f]  # (2, dim_t, dim_f)

        # Model expects [4, dim_f, dim_t] = [4, freq, time]
        # STFT gives [2, time, freq]; transpose to [2, freq, time]
        S_trimmed = S_trimmed.transpose(0, 2, 1)  # (2, dim_f, dim_t)

        # Interleave real/imag for 2 channels: [4, dim_f, dim_t]
        # Order: ch0_real, ch0_imag, ch1_real, ch1_imag
        S_real = np.real(S_trimmed)  # (2, dim_f, dim_t)
        S_imag = np.imag(S_trimmed)
        model_in = np.stack(
            [S_real[0], S_imag[0], S_real[1], S_imag[1]], axis=0
        )  # (4, dim_f, dim_t)
        model_in = model_in[np.newaxis, :, :, :]  # (1, 4, dim_f, dim_t)  — already float32

        # ── ONNX inference ────────────────────────────────────────────
        try:
            model_out: np.ndarray = session.run(None, {"input": model_in})[0]
        except Exception:
            return None  # soft-fail

        # ── iSTFT ────────────────────────────────────────────────────
        # model_out: (1, 4, dim_f, dim_t)
        out = model_out[0]  # (4, dim_f, dim_t)

        # Split back to 2-channel complex
        out_real = np.stack([out[0], out[2]], axis=0)  # (2, dim_f, dim_t)
        out_imag = np.stack([out[1], out[3]], axis=0)
        out_complex = out_real + 1j * out_imag  # (2, dim_f, dim_t)

        # Pad frequency dimension back to n_bins
        freq_pad = np.zeros((2, n_bins - dim_f, dim_t), dtype=np.complex64)
        out_complex = np.concatenate([out_complex, freq_pad], axis=1)  # (2, n_bins, dim_t)

        # Transpose frames → (2, dim_t, n_bins)
        out_complex = out_complex.transpose(0, 2, 1)  # (2, dim_t, n_bins)

        # iSTFT
        chunk_out = _istft(out_complex, n_fft, hop, window, length=chunk_size)  # (2, chunk_size)

        # Apply window for overlap-add
        chunk_out *= chunk_window_2ch
        result[:, start:end] += chunk_out
        divider[:, start:end] += chunk_window_2ch

        if progress is not None:
            progress((ci + 1) / n_chunks)

    # Normalise overlap-add and trim padding
    divider = np.clip(divider, 1e-12, None)  # avoid div-by-zero
    out = result / divider
    out = out[:, trim : trim + n_samples]  # remove padding

    # Compensate gain
    out *= compensate

    # Clip to float32 range
    np.clip(out, -1.0, 1.0, out=out)

    return out  # already float32

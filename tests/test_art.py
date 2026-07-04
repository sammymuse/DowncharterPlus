"""Tests for downcharter/art.py — DXT1 texture builder (.png_ps3 / .png_xbox).

The encoder was vectorised (b3b849b) and shipped with two crashes nothing
covered: float64 left_shift in _unpack565b and a read-only frombuffer view in
build_png_xbox. These tests decode the DXT1 output with an independent
reference decoder so encoder regressions show up as quality/round-trip
failures, not just "doesn't raise".
"""
import os

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image

from downcharter import art as _art


def _decode_dxt1(data: bytes, w: int, h: int) -> np.ndarray:
    """Minimal reference DXT1 decoder (independent of the encoder's helpers)."""
    out = np.zeros((h, w, 3), np.float64)
    bi = 0
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            c0 = int.from_bytes(data[bi:bi + 2], "little")
            c1 = int.from_bytes(data[bi + 2:bi + 4], "little")
            bits = int.from_bytes(data[bi + 4:bi + 8], "little")
            bi += 8

            def up(c):
                r = (c >> 11) & 0x1F
                g = (c >> 5) & 0x3F
                b = c & 0x1F
                return np.array([(r << 3) | (r >> 2),
                                 (g << 2) | (g >> 4),
                                 (b << 3) | (b >> 2)], np.float64)

            p0, p1 = up(c0), up(c1)
            if c0 > c1:
                pal = [p0, p1, (2 * p0 + p1) / 3.0, (p0 + 2 * p1) / 3.0]
            else:
                pal = [p0, p1, (p0 + p1) / 2.0, np.zeros(3)]
            for i in range(16):
                out[by + i // 4, bx + i % 4] = pal[(bits >> (2 * i)) & 3]
    return out


def test_encode_dxt1_size_and_no_crash():
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    enc = _art._encode_dxt1(img)
    assert len(enc) == (16 // 4) * (16 // 4) * 8


def test_encode_dxt1_gradient_quality():
    """Smooth gradient must decode back close to the input (DXT1 is lossy but
    RGB565 quantisation alone stays under ~4 mean abs error)."""
    xx, yy = np.meshgrid(np.arange(64), np.arange(64))
    img = np.stack([xx * 4, yy * 4, (xx + yy) * 2], axis=-1).astype(np.uint8)
    dec = _decode_dxt1(_art._encode_dxt1(img), 64, 64)
    err = np.abs(dec - img.astype(np.float64)).mean()
    assert err < 4.0, f"mean abs error {err:.2f} — encoder degraded"


def test_encode_dxt1_solid_flat_blocks():
    """Solid colour → flat path: both endpoints equal, zero indices, and the
    decode error is only the RGB565 low-bit replication."""
    solid = np.full((8, 8, 3), (200, 50, 120), np.uint8)
    enc = _art._encode_dxt1(solid)
    for bi in range(0, len(enc), 8):
        assert enc[bi:bi + 2] == enc[bi + 2:bi + 4]     # c0 == c1
        assert enc[bi + 4:bi + 8] == b"\x00" * 4        # solid indices
    dec = _decode_dxt1(enc, 8, 8)
    assert np.abs(dec - solid.astype(np.float64)).max() <= 8.0


def _write_cover(tmp_path, size=64) -> str:
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    img = np.stack([xx * 3 % 256, yy * 3 % 256, (xx ^ yy) % 256],
                   axis=-1).astype(np.uint8)
    p = os.path.join(str(tmp_path), "album.png")
    Image.fromarray(img).save(p)
    return p


def test_build_png_ps3_layout(tmp_path):
    cover = _write_cover(tmp_path)
    blob = _art.build_png_ps3(cover, size=64)
    mips = (64, 32, 16, 8, 4)
    assert len(blob) == 32 + sum(s * s // 2 for s in mips)
    assert blob[0] == 0x01 and blob[1] == 0x04          # version, DXT1 bpp
    assert blob[7] | (blob[8] << 8) == 64               # width u16le


def test_build_png_xbox_swaps_payload_only(tmp_path):
    """xbox = ps3 with every u16 of the PAYLOAD byte-swapped, header intact
    (regression: read-only frombuffer view crashed the build)."""
    cover = _write_cover(tmp_path)
    ps3 = _art.build_png_ps3(cover, size=64)
    xb = _art.build_png_xbox(cover, size=64)
    assert len(xb) == len(ps3)
    assert xb[:32] == ps3[:32]
    sw = bytearray(xb[32:])
    sw[0::2], sw[1::2] = xb[33::2], xb[32::2]
    assert bytes(sw) == ps3[32:]

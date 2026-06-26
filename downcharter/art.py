"""art.py — generate a Rock Band PS3 album-art texture (.png_ps3) from a cover image.

YARG / Clone Hero songs ship the cover as a plain ``album.png`` (or .jpg). RB3 on
PS3 instead wants an **HMX texture** ``<id>_keep.png_ps3``: a 32-byte HMX bitmap
header followed by DXT1-compressed image data with a mipmap chain.

Format (reverse-engineered byte-exactly from the Onyx-converted packs in
``midis/PS3 Converted/`` — a 256×256 cover comes out as exactly 43 720 bytes):

    0x00  32-byte HMX header  (constant for a 256×256 DXT1 texture)
          01 04 08 00 00 00 04 00 01 00 01 80  then 20 zero bytes
            01 = version
            04 = bits-per-pixel (DXT1 = 4 bpp)
            08 = encoding (8 = DXT1)
            00 = mipmap flag
            …  = the 256×256 dims / stride, baked into the constant header
    0x20  DXT1 data, little-endian, for mips 256,128,64,32,16,8,4 concatenated
          (each block 8 bytes: c0 u16le, c1 u16le, then 32 2-bit indices)

Platform note (verified empirically by decoding a real .png_ps3 both ways): the
PS3 variant stores DXT1 blocks in **plain little-endian** — NOT byte-swapped.
The Xbox-360 .png_xbox variant is the same payload with every 16-bit word
byte-swapped; .png_ps3 needs no swap. We only emit PS3 here.

No external DXT library: a compact range-fit DXT1 encoder lives below (good enough
for album art, and keeps the dependency surface at numpy + Pillow, both already
required by audio.py).
"""
from __future__ import annotations

# 32-byte HMX header for a 256×256 DXT1 texture, taken verbatim from real
# Onyx-produced .png_ps3 files (identical across every cover we checked).
_HEADER_256_DXT1 = bytes([
    0x01, 0x04, 0x08, 0x00, 0x00, 0x00, 0x04, 0x00,
    0x01, 0x00, 0x01, 0x80, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
])

# RB album art is square 256×256; mip chain goes down to the 4×4 DXT block floor.
_SIZE = 256
_MIP_SIZES = (256, 128, 64, 32, 16, 8, 4)

# Cover filenames CH / YARG songs ship.
_COVER_NAMES = ("album.png", "album.jpg", "album.jpeg", "cover.png", "cover.jpg",
                "cover.jpeg", "albumart.png", "album.bmp")


def available() -> bool:
    """True if Pillow + numpy are importable (the optional deps art needs)."""
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except Exception:
        return False


def find_cover(folder: str) -> str | None:
    """First cover image in `folder` (case-insensitive, common CH/YARG names)."""
    import os
    try:
        entries = {f.lower(): f for f in os.listdir(folder)}
    except OSError:
        return None
    for name in _COVER_NAMES:
        if name in entries:
            return os.path.join(folder, entries[name])
    # any image as a last resort
    for low, real in entries.items():
        if low.rsplit(".", 1)[-1] in ("png", "jpg", "jpeg", "bmp"):
            return os.path.join(folder, real)
    return None


def _rgb565(block):
    """Pack an (N,3) uint8 array of RGB to (N,) uint16 RGB565."""
    import numpy as np
    r = (block[:, 0].astype(np.uint16) >> 3) & 0x1F
    g = (block[:, 1].astype(np.uint16) >> 2) & 0x3F
    b = (block[:, 2].astype(np.uint16) >> 3) & 0x1F
    return (r << 11) | (g << 5) | b


def _unpack565(c):
    """uint16 RGB565 → (3,) float RGB in 0..255 (with low-bit replication)."""
    r = (c >> 11) & 0x1F
    g = (c >> 5) & 0x3F
    b = c & 0x1F
    return (
        float((r << 3) | (r >> 2)),
        float((g << 2) | (g >> 4)),
        float((b << 3) | (b >> 2)),
    )


# Decoder interpolation weight of c0 for each of the 4 DXT1 indices:
#   idx0 = c0 (w=1) · idx1 = c1 (w=0) · idx2 = (2c0+c1)/3 · idx3 = (c0+2c1)/3
_C0_WEIGHT = (1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0)


def _principal_axis(centered):
    """Dominant colour-variation axis of a block via power iteration on the
    3×3 covariance. `centered` is (N,3) float (block colours minus their mean).
    Returns a unit (3,) vector, or None if the block is (near) flat."""
    import numpy as np
    cov = centered.T @ centered                      # (3,3)
    v = cov.sum(axis=0)                              # seed along total variance
    n = np.linalg.norm(v)
    if n < 1e-6:
        v = np.array([1.0, 1.0, 1.0])                # degenerate seed
    else:
        v = v / n
    for _ in range(8):
        v = cov @ v
        n = np.linalg.norm(v)
        if n < 1e-9:
            return None                              # flat block
        v = v / n
    return v


def _fit_endpoints(blk, axis):
    """Initial float endpoints: extreme projections of the block onto `axis`."""
    import numpy as np
    mean = blk.mean(axis=0)
    proj = (blk - mean) @ axis
    return mean + axis * proj.max(), mean + axis * proj.min()


def _encode_dxt1(rgb) -> bytes:
    """High-quality DXT1 encode of an (H,W,3) uint8 image (H,W multiples of 4).

    Per 4×4 block we fit the colour line along the PRINCIPAL AXIS (PCA via power
    iteration) instead of the per-channel bounding box — the bbox over-expands
    the range and desaturates, the principal axis tracks the real colour spread.
    Endpoints are then refined by LEAST SQUARES against the chosen indices (a
    couple of stb_dxt-style iterations), recovering detail the old range-fit lost.
    Output is byte-identical in layout (c0 u16le, c1 u16le, 32 2-bit indices).
    """
    import numpy as np
    h, w, _ = rgb.shape
    weights = np.array(_C0_WEIGHT)                   # (4,)
    out = bytearray()
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            blk = rgb[by:by + 4, bx:bx + 4, :].reshape(-1, 3).astype(np.float64)
            axis = _principal_axis(blk - blk.mean(axis=0))
            if axis is None:                         # flat block
                c0 = int(_rgb565(blk[:1].astype(np.int32))[0])
                out += bytes([c0 & 0xFF, (c0 >> 8) & 0xFF,
                              c0 & 0xFF, (c0 >> 8) & 0xFF, 0, 0, 0, 0])
                continue

            e0, e1 = _fit_endpoints(blk, axis)
            idx = None
            for _ in range(3):                       # refine endpoints ↔ indices
                c0q = int(_rgb565(np.clip(np.round(e0), 0, 255)
                                  .astype(np.int32)[None, :])[0])
                c1q = int(_rgb565(np.clip(np.round(e1), 0, 255)
                                  .astype(np.int32)[None, :])[0])
                p0 = np.array(_unpack565(c0q))
                p1 = np.array(_unpack565(c1q))
                palette = np.stack([p0, p1,
                                    (2 * p0 + p1) / 3.0,
                                    (p0 + 2 * p1) / 3.0])      # (4,3)
                d = ((blk[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
                idx = d.argmin(axis=1)                          # (16,)
                # least-squares: minimise Σ‖p − (w·E0 + (1−w)·E1)‖² over E0,E1
                wpix = weights[idx]                             # (16,)
                a = float((wpix * wpix).sum())
                b = float((wpix * (1.0 - wpix)).sum())
                c = float(((1.0 - wpix) * (1.0 - wpix)).sum())
                det = a * c - b * b
                if abs(det) < 1e-9:
                    break
                pw0 = (wpix[:, None] * blk).sum(axis=0)         # Σ w·p
                pw1 = ((1.0 - wpix)[:, None] * blk).sum(axis=0) # Σ (1−w)·p
                e0 = (c * pw0 - b * pw1) / det
                e1 = (a * pw1 - b * pw0) / det

            c0 = int(_rgb565(np.clip(np.round(e0), 0, 255)
                             .astype(np.int32)[None, :])[0])
            c1 = int(_rgb565(np.clip(np.round(e1), 0, 255)
                             .astype(np.int32)[None, :])[0])
            if c0 == c1:                             # collapsed → flat block
                out += bytes([c0 & 0xFF, (c0 >> 8) & 0xFF,
                              c1 & 0xFF, (c1 >> 8) & 0xFF, 0, 0, 0, 0])
                continue
            if c0 < c1:                              # keep c0>c1 → 4-colour mode
                c0, c1 = c1, c0
            # final index assignment against the quantised palette
            p0 = np.array(_unpack565(c0))
            p1 = np.array(_unpack565(c1))
            palette = np.stack([p0, p1,
                                (2 * p0 + p1) / 3.0,
                                (p0 + 2 * p1) / 3.0])
            d = ((blk[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
            idx = d.argmin(axis=1).astype(np.uint32)
            bits = 0
            for i in range(16):
                bits |= int(idx[i]) << (2 * i)
            out += bytes([
                c0 & 0xFF, (c0 >> 8) & 0xFF,
                c1 & 0xFF, (c1 >> 8) & 0xFF,
                bits & 0xFF, (bits >> 8) & 0xFF,
                (bits >> 16) & 0xFF, (bits >> 24) & 0xFF,
            ])
    return bytes(out)


def build_png_ps3(cover_path: str) -> bytes:
    """Build a 256×256 DXT1 .png_ps3 texture (header + mip chain) from `cover_path`.

    Raises if Pillow/numpy are unavailable or the image can't be read.
    """
    import numpy as np
    from PIL import Image

    img = Image.open(cover_path).convert("RGB")
    base = img.resize((_SIZE, _SIZE), Image.LANCZOS)

    data = bytearray(_HEADER_256_DXT1)
    for sz in _MIP_SIZES:
        mip = base if sz == _SIZE else base.resize((sz, sz), Image.LANCZOS)
        data += _encode_dxt1(np.asarray(mip, dtype=np.uint8))
    return bytes(data)

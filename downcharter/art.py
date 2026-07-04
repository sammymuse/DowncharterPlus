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
The Xbox-360 .png_xbox variant has the IDENTICAL 32-byte header but every 16-bit
word of the DXT *payload* byte-swapped (the header stays little-endian — confirmed
against a real Onyx .png_xbox whose width/height fields are un-swapped). So
``build_png_xbox`` = ``build_png_ps3`` with the post-header data 2-byte-swapped.

No external DXT library: a compact range-fit DXT1 encoder lives below (good enough
for album art, and keeps the dependency surface at numpy + Pillow, both already
required by audio.py).
"""
from __future__ import annotations

# HMX bitmap header (32 bytes), built per-resolution. The dimension fields were
# reverse-engineered from real Onyx .png_ps3 files (256×256 → byte-identical):
#   0      version          = 0x01
#   1      bpp              = 0x04   (DXT1 = 4 bits/pixel)
#   2..5   encoding (u32le) = 8      (DXT1, PS3 plain little-endian)
#   6      mipmaps (u8)     = levels-3  (Onyx writes 4 for a 7-level 256 chain)
#   7..8   width  (u16le)
#   9..10  height (u16le)
#   11..12 bytes-per-line (u16le) = width·bpp/8 = width/2
#   13..31 zero padding
def _hmx_header(size: int, levels: int) -> bytes:
    h = bytearray(32)
    h[0] = 0x01
    h[1] = 0x04
    h[2] = 0x08                               # encoding u32le = 8
    h[6] = levels - 3                         # 256→4, matching Onyx
    h[7] = size & 0xFF;  h[8] = (size >> 8) & 0xFF        # width
    h[9] = size & 0xFF;  h[10] = (size >> 8) & 0xFF       # height
    bpl = size // 2                           # DXT1: width·4bpp/8
    h[11] = bpl & 0xFF;  h[12] = (bpl >> 8) & 0xFF
    return bytes(h)


def _mip_sizes(size: int) -> tuple:
    """Full mip chain from `size` down to the 4×4 DXT block floor."""
    sizes = []
    s = size
    while s >= 4:
        sizes.append(s)
        s //= 2
    return tuple(sizes)


# RB album art is square. 512×512 is the default: verified in-game on RB3/RPCS3
# (renders sharp, no crash) and YARG reads the dimensions from the header. 256
# stays selectable and is byte-identical to Onyx if ever needed.
_SIZE = 512

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

    All 4×4 blocks are processed in a single vectorised pass: PCA via batched
    power-iteration, least-squares refinement, and final packing into 8-byte
    DXT1 blocks (c0 u16le, c1 u16le, 32 2-bit indices).
    """
    import numpy as np
    h, w, _c = rgb.shape
    bh, bw = h // 4, w // 4
    if bh == 0 or bw == 0:
        return b""

    # ── 1. Extract all 4×4 blocks → (N, 16, 3) ──────────────────────────
    blocks = (rgb.reshape(bh, 4, bw, 4, 3)
              .transpose(0, 2, 1, 3, 4)
              .reshape(-1, 16, 3)
              .astype(np.float64))
    N = blocks.shape[0]
    weights = np.array(_C0_WEIGHT, dtype=np.float64)           # (4,)

    # ── 2. Per-block means & centred data ───────────────────────────────
    means = blocks.mean(axis=1)                                 # (N, 3)
    centred = blocks - means[:, None, :]                        # (N, 16, 3)

    # ── 3. Per-block 3×3 covariance & flat-block detection ──────────────
    cov = np.einsum("nij,nik->njk", centred, centred)          # (N, 3, 3)
    trace = cov[:, 0, 0] + cov[:, 1, 1] + cov[:, 2, 2]
    flat = trace < 1e-4                                         # (N,) bool

    # ── 4. Batched power-iteration for principal axis ───────────────────
    axis = cov.sum(axis=1)                                     # seed  (N, 3)
    axis /= np.sqrt((axis * axis).sum(axis=1, keepdims=True)) + 1e-12

    active = ~flat
    if active.any():
        cov_a = cov[active]
        ax_a = axis[active]
        for _ in range(8):
            ax_new = np.einsum("nij,nj->ni", cov_a, ax_a)
            ax_new /= np.sqrt((ax_new * ax_new).sum(axis=1,
                                                     keepdims=True)) + 1e-12
            ax_a = ax_new
        axis[active] = ax_a

    # ── 5. Initial endpoints via projection onto principal axis ─────────
    proj = np.einsum("nij,nj->ni", centred, axis)              # (N, 16)
    e0 = means + axis * proj.max(axis=1)[:, None]              # (N, 3)
    e1 = means + axis * proj.min(axis=1)[:, None]              # (N, 3)

    # ── Helper: batch RGB565 pack / unpack ──────────────────────────────
    def _pack565b(c):
        """(…,3) float → (…,) uint16."""
        r = (np.clip(np.round(c[..., 0]), 0, 255).astype(np.uint16) >> 3) & 0x1F
        g = (np.clip(np.round(c[..., 1]), 0, 255).astype(np.uint16) >> 2) & 0x3F
        b = (np.clip(np.round(c[..., 2]), 0, 255).astype(np.uint16) >> 3) & 0x1F
        return (r << 11) | (g << 5) | b

    def _unpack565b(c):
        """(…,) uint16 → (…,3) float64 with low-bit replication."""
        r = (c >> 11) & 0x1F
        g = (c >> 5) & 0x3F
        b = c & 0x1F
        r_f = ((r.astype(np.float64) << 3) | (r.astype(np.float64) >> 2))
        g_f = ((g.astype(np.float64) << 2) | (g.astype(np.float64) >> 4))
        b_f = ((b.astype(np.float64) << 3) | (b.astype(np.float64) >> 2))
        return np.stack([r_f, g_f, b_f], axis=-1)

    # ── 6. Least-squares refinement (3 iterations) ──────────────────────
    refine = active.copy()
    for _ in range(3):
        if not refine.any():
            break
        c0 = _pack565b(e0)
        c1 = _pack565b(e1)

        p0 = _unpack565b(c0)                                   # (N, 3)
        p1 = _unpack565b(c1)                                   # (N, 3)
        palette = np.stack(
            [p0, p1, (2 * p0 + p1) / 3.0, (p0 + 2 * p1) / 3.0],
            axis=1)                                            # (N, 4, 3)

        d = ((blocks[:, :, None, :] - palette[:, None, :, :])
             ** 2).sum(axis=3)                                 # (N, 16, 4)
        idx = d.argmin(axis=2)                                 # (N, 16)

        wpix = weights[idx]                                    # (N, 16)
        a = (wpix * wpix).sum(axis=1)                          # (N,)
        b = (wpix * (1.0 - wpix)).sum(axis=1)                  # (N,)
        c = ((1.0 - wpix) * (1.0 - wpix)).sum(axis=1)         # (N,)
        det = a * c - b * b                                    # (N,)

        ok = refine & (np.abs(det) > 1e-9)
        if not ok.any():
            break
        pw0 = (wpix[ok, :, None] * blocks[ok]).sum(axis=1)    # (n_ok, 3)
        pw1 = ((1.0 - wpix[ok])[:, :, None] *
               blocks[ok]).sum(axis=1)                         # (n_ok, 3)
        det_ok = det[ok]
        e0[ok] = (c[ok, None] * pw0 - b[ok, None] * pw1) / det_ok[:, None]
        e1[ok] = (a[ok, None] * pw1 - b[ok, None] * pw0) / det_ok[:, None]
        # Update refinement mask: stop refining blocks whose det fell below threshold
        refine = ok

    # ── 7. Final quantisation & mode selection ──────────────────────────
    c0 = _pack565b(e0).astype(np.uint64)
    c1 = _pack565b(e1).astype(np.uint64)
    collapsed = flat | (c0 == c1)                              # treat as flat

    # Ensure c0 > c1 on non-collapsed blocks (triggers DXT1 4-colour mode)
    swap = (c0 < c1) & (~collapsed)
    c0_n = np.where(swap, c1, c0)
    c1_n = np.where(swap, c0, c1)

    # ── 8. Final index assignment against the quantised palette ─────────
    p0 = _unpack565b(c0_n.astype(np.uint16))
    p1 = _unpack565b(c1_n.astype(np.uint16))
    palette = np.stack(
        [p0, p1, (2 * p0 + p1) / 3.0, (p0 + 2 * p1) / 3.0],
        axis=1)
    d = ((blocks[:, :, None, :] - palette[:, None, :, :]) ** 2).sum(axis=3)
    idx = d.argmin(axis=2).astype(np.uint32)                   # (N, 16)

    # ── 9. Pack indices into 32-bit words ───────────────────────────────
    shifts = np.arange(16, dtype=np.uint32) * 2
    bits = (idx.astype(np.uint32) << shifts[None, :]).sum(axis=1)

    # ── 10. Assemble 8-byte blocks ──────────────────────────────────────
    out = np.empty(N * 8, dtype=np.uint8)
    out[0::8] = c0_n & 0xFF;                    out[1::8] = (c0_n >> 8) & 0xFF
    out[2::8] = c1_n & 0xFF;                    out[3::8] = (c1_n >> 8) & 0xFF
    out[4::8] = bits & 0xFF;                    out[5::8] = (bits >> 8) & 0xFF
    out[6::8] = (bits >> 16) & 0xFF;            out[7::8] = (bits >> 24) & 0xFF

    # ── 11. Override flat/collapsed blocks with solid colour ────────────
    c_flat = _pack565b(blocks[:, 0, :]).astype(np.uint16)
    cflat = collapsed
    out_2d = out.reshape(N, 8)
    out_2d[cflat, 0] = c_flat[cflat] & 0xFF
    out_2d[cflat, 1] = (c_flat[cflat] >> 8) & 0xFF
    out_2d[cflat, 2] = c_flat[cflat] & 0xFF
    out_2d[cflat, 3] = (c_flat[cflat] >> 8) & 0xFF
    out_2d[cflat, 4:] = 0

    return out.tobytes()


def build_png_ps3(cover_path: str, size: int = _SIZE) -> bytes:
    """Build a `size`×`size` DXT1 .png_ps3 texture (header + mip chain).

    `size` must be a power of two (256 = platform standard / byte-identical to
    Onyx; 512 = higher-resolution, experimental). Raises if Pillow/numpy are
    unavailable or the image can't be read.
    """
    import numpy as np
    from PIL import Image

    if size < 4 or (size & (size - 1)) != 0:
        raise ValueError(f"art size must be a power of two ≥4, got {size}")

    img = Image.open(cover_path).convert("RGB")
    base = img.resize((size, size), Image.LANCZOS)

    mip_sizes = _mip_sizes(size)
    data = bytearray(_hmx_header(size, len(mip_sizes)))
    for sz in mip_sizes:
        mip = base if sz == size else base.resize((sz, sz), Image.LANCZOS)
        data += _encode_dxt1(np.asarray(mip, dtype=np.uint8))
    return bytes(data)


def build_png_xbox(cover_path: str, size: int = _SIZE) -> bytes:
    """Build a `size`×`size` DXT1 ``.png_xbox`` texture for Xbox-360 / YARG.

    Identical to :func:`build_png_ps3` except the DXT payload (everything after
    the 32-byte HMX header) has every 16-bit word byte-swapped — the documented
    PS3↔Xbox texture-endianness difference, confirmed against a real Onyx
    ``.png_xbox`` (whose HMX header stays little-endian).
    """
    import numpy as np
    data = np.frombuffer(build_png_ps3(cover_path, size), dtype=np.uint8)
    # Byte-swap every u16 word in the payload (skip the 32-byte HMX header)
    payload = data[32:]
    even_len = (len(payload) // 2) * 2
    pairs = payload[:even_len].reshape(-1, 2)
    data[32:32 + even_len] = pairs[:, ::-1].ravel()
    return data.tobytes()

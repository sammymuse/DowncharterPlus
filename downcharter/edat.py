"""edat.py — write an *unencrypted* (debug) PS3 .edat container.

Rock Band 3 on PS3/RPCS3 loads a song's chart as ``<id>.mid.edat`` — it will not
read a plain ``<id>.mid``. The .edat is an NPDRM container. Normally the data is
AES-encrypted with a per-folder klicensee, but RB3DX *nightly* also accepts an
**unencrypted** library: moggs left as version 0x0A and EDATs built with the
``EDAT_DEBUG_DATA_FLAG`` set.

That debug flag is the whole trick. In RPCS3's ``unedat.cpp``:

  * ``validate_npd_hashes()`` does ``if (flags & EDAT_DEBUG_DATA_FLAG) return true;``
    — so NONE of the NPD header CMAC/hash fields are checked. We can leave the
    digest / title_hash / dev_hash and the trailing signatures all zero.
  * ``decrypt_block()`` copies the block bytes verbatim (no AES) when the debug
    flag is set, and the per-block metadata hashes are not enforced either.

So a debug EDAT needs no NPDRM keys at all — only the correct *structure*:

    0x000  NPD header        (0x80)  magic, version, license, type, content_id …
    0x080  EDAT header       (0x10)  flags, block_size, file_size  (all big-endian)
    0x090  → 0x100  zero padding (metadata + header signatures live here; unchecked)
    0x100  metadata section  (total_blocks * 0x10)  per-block hashes (left zero)
    ....   data              the plaintext payload, written block_size-contiguous

Block geometry (from unedat.cpp, non-compressed / non-0x20 case):
    metadata_section_size = 0x10
    total_blocks          = ceil(file_size / block_size)
    block N data offset   = 0x100 + N*block_size + total_blocks*0x10
Because every block but the last is exactly ``block_size`` long, the data region
is just the raw payload concatenated right after the metadata section.
"""
from __future__ import annotations
import os
import struct

NPD_MAGIC = b"NPD\x00"
EDAT_DEBUG_DATA_FLAG = 0x80000000
_METADATA_OFFSET = 0x100
_METADATA_ENTRY = 0x10          # non-compressed, non-0x20 case
_DEFAULT_BLOCK = 0x4000         # 16 KiB, same as the Onyx/official packs

# Rock Band 3 (US, BLUS30463). The content-id label is the package folder name,
# e.g. UP8802-BLUS30463_00-ARCHITECTSELEGY2X . Not validated in debug mode, but
# we set it correctly so the file is well-formed and manager tools are happy.
CONTENT_ID_PREFIX = "UP8802-BLUS30463_00-"


def _content_id(label: str) -> bytes:
    cid = (CONTENT_ID_PREFIX + label).encode("ascii", "replace")[:0x30]
    return cid.ljust(0x30, b"\x00")


def build_debug_edat(data: bytes, out_path: str, content_label: str,
                     block_size: int = _DEFAULT_BLOCK) -> str:
    """Write `data` into an unencrypted (debug) .edat at `out_path`.

    `content_label` is the package folder name (the content-id label segment).
    Returns `out_path`.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    file_size = len(data)
    total_blocks = (file_size + block_size - 1) // block_size if file_size else 0

    buf = bytearray(_METADATA_OFFSET)            # 0x100 zero-filled header region
    buf[0x00:0x04] = NPD_MAGIC
    struct.pack_into(">I", buf, 0x04, 2)         # NPD version
    struct.pack_into(">I", buf, 0x08, 3)         # license = free
    struct.pack_into(">I", buf, 0x0C, 0)         # type = EDAT
    buf[0x10:0x40] = _content_id(content_label)  # 0x30 content-id
    # 0x40 digest / 0x50 title_hash / 0x60 dev_hash / 0x70 timestamps: left zero.
    struct.pack_into(">I", buf, 0x80, EDAT_DEBUG_DATA_FLAG)  # flags
    struct.pack_into(">I", buf, 0x84, block_size)            # block_size
    struct.pack_into(">Q", buf, 0x88, file_size)             # file_size (u64)
    # 0x90..0x100: metadata/header signatures — unchecked in debug, left zero.

    buf += b"\x00" * (total_blocks * _METADATA_ENTRY)        # per-block hashes (zero)
    buf += data                                              # plaintext payload

    with open(out_path, "wb") as f:
        f.write(buf)
    return out_path


# ── self-check: mirror RPCS3's read geometry to confirm the writer round-trips ─
def _extract_debug_edat(path: str) -> bytes:
    """Re-derive the payload from a debug .edat using RPCS3's block formula.
    For tests only — proves our layout is internally consistent."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[0:4] != NPD_MAGIC:
        raise ValueError("not an NPD/EDAT file")
    flags = struct.unpack_from(">I", raw, 0x80)[0]
    block_size = struct.unpack_from(">I", raw, 0x84)[0]
    file_size = struct.unpack_from(">Q", raw, 0x88)[0]
    if not (flags & EDAT_DEBUG_DATA_FLAG):
        raise ValueError("not a debug (unencrypted) EDAT")
    total_blocks = (file_size + block_size - 1) // block_size if file_size else 0
    out = bytearray()
    for n in range(total_blocks):
        offset = _METADATA_OFFSET + n * block_size + total_blocks * _METADATA_ENTRY
        length = block_size
        if n == total_blocks - 1 and file_size % block_size:
            length = file_size % block_size
        out += raw[offset:offset + length]
    return bytes(out)

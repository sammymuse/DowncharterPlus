"""sng.py — assemble a YARG / Clone Hero **SNG** package (Phase D, sibling of stfs.py).

SNG is the native single-file container for YARG and Clone Hero. Unlike the RB
PS3/CON builds (songs.dta + milo + mogg), an SNG simply packs a Clone-Hero-style
song folder: the song.ini key/values become the **Metadata** section, and the
chart + audio + artwork become the embedded **FileData** (XOR-masked, not encrypted).

Binary layout (spec: github.com/mdsitton/SngFileFormat — all numbers little-endian):
  Header:    "SNGPKG" (6) · version u32 · xorMask[16]
  Metadata:  u64 sectionLen · u64 count · [i32 keyLen·key · i32 valLen·val] …
  FileIndex: u64 sectionLen · u64 count · [u8 nameLen·name · u64 size · u64 dataIdx] …
  FileData:  u64 sectionLen · masked file bytes (concatenated)
Masking (per file, index i from 0): out[i] = in[i] XOR (xorMask[i & 15] XOR (i & 0xFF)).

Unlike the RB PS3/CON builds, the SNG is a pure verbatim repackage: it takes the
song folder as-is (the chart, audio, artwork) and the song.ini → Metadata. No pedal
variants, no MIDI validation, no milo — YARG/CH read what is already there.
"""
from __future__ import annotations
import os
import struct

from . import ps3build as _ps3

_parse_song_ini = _ps3._parse_song_ini
_find_one = _ps3._find_one
_sanitize_shortname = _ps3._sanitize_shortname

MAGIC = b"SNGPKG"
VERSION = 1

# Media copied verbatim into the package (CH/YARG conventions).
_AUDIO_EXT = (".ogg", ".opus", ".mp3", ".wav", ".flac")
_IMAGE_EXT = (".png", ".jpg", ".jpeg")
_VIDEO_EXT = (".mp4", ".avi", ".webm", ".ogv", ".mpeg", ".mpg", ".vp8")
_MEDIA_EXT = _AUDIO_EXT + _IMAGE_EXT + _VIDEO_EXT


def _noop_log(msg, tag=None):
    pass


def _read_ini_pairs(path: str) -> list:
    """(key, value) pairs from a song.ini's [song] section, byte-faithful to the
    reference SNG encoder's IniParser so YARG/Clone Hero round-trip the metadata
    intact: read only the `song` section (case-insensitive), skip `#`/`;` comment
    lines, split on the FIRST `=`, trim key and value, keep original case, and let
    the LAST duplicate key win — no lowercasing, no character stripping."""
    # Mirror the reference's OrdinalIgnoreCase dictionary: keys collide
    # case-insensitively (first-seen casing kept), and the last value wins.
    canon: dict[str, str] = {}         # lower(key) → original key casing
    values: dict[str, str] = {}        # original key → value
    order: list[str] = []              # first-seen order
    in_song = False
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line[0] in "#;":
                    continue
                if line.startswith("[") and line.endswith("]"):
                    in_song = line[1:-1].strip().lower() == "song"
                    continue
                if not in_song or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip()
                if not k:
                    continue
                lk = k.lower()
                if lk not in canon:
                    canon[lk] = k
                    order.append(k)
                values[canon[lk]] = v
    except Exception:
        pass
    return [(k, values[k]) for k in order]


def pack_sng(metadata, files: dict, xor_mask: bytes | None = None) -> bytes:
    """Pack metadata pairs + a {filename: bytes} map into an SNG container.

    `metadata` is an ordered iterable of (key, value) string pairs.
    `files` maps a relative filename (max 255 bytes UTF-8) to its raw bytes.
    """
    if xor_mask is None:
        xor_mask = os.urandom(16)
    if len(xor_mask) != 16:
        raise ValueError("xor_mask must be 16 bytes")

    # ── Metadata section ──────────────────────────────────────────────────────
    meta_body = bytearray()
    count = 0
    for k, v in metadata:
        kb = str(k).encode("utf-8")
        vb = str(v).encode("utf-8")
        meta_body += struct.pack("<i", len(kb)) + kb
        meta_body += struct.pack("<i", len(vb)) + vb
        count += 1
    meta_section = struct.pack("<Q", len(meta_body) + 8) + struct.pack("<Q", count) + meta_body

    # ── FileIndex + masked FileData ───────────────────────────────────────────
    # contentsIndex is the ABSOLUTE offset of the file's data from the start of
    # the whole .sng (verified against the reference encoder), not relative to
    # the FileData body. The index body has a fixed size once filenames are
    # known, so we can pre-compute where FileData begins and offset from there.
    names = [n.encode("utf-8") for n in files]
    for nb in names:
        if len(nb) > 255:
            raise ValueError("filename too long (>255 bytes)")
    index_body_len = sum(1 + len(nb) + 16 for nb in names)  # u8 len + name + u64 size + u64 idx
    index_section_len = 8 + index_body_len                   # + u64 count
    data_start = (len(MAGIC) + 4 + 16              # header
                  + len(meta_section)               # metadata section (with its len field)
                  + (8 + index_section_len)         # file-index section (with its len field)
                  + 8)                              # FileData section length field

    index_body = bytearray()
    data_body = bytearray()
    offset = data_start
    for nb, blob in zip(names, files.values()):
        index_body += struct.pack("<B", len(nb)) + nb
        index_body += struct.pack("<Q", len(blob))
        index_body += struct.pack("<Q", offset)
        masked = bytearray(len(blob))
        for i, b in enumerate(blob):
            masked[i] = b ^ (xor_mask[i & 0x0F] ^ (i & 0xFF))
        data_body += masked
        offset += len(blob)

    index_section = (struct.pack("<Q", index_section_len)
                     + struct.pack("<Q", len(files)) + index_body)
    data_section = struct.pack("<Q", len(data_body)) + data_body

    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", VERSION)
    out += xor_mask
    out += meta_section
    out += index_section
    out += data_section
    return bytes(out)


# Chart + non-media song files packed verbatim alongside the media.
_CHART_NAMES = ("notes.mid", "notes.chart")


def build_sng_song(src_folder: str, log_fn=None, out_base: str | None = None) -> str:
    """Pack `src_folder` into a YARG/Clone-Hero `<PKG>.sng`, verbatim.

    No pedal variants, no MIDI validation, no milo: the chart, audio, artwork and
    video are embedded exactly as they sit in the folder, and the song.ini becomes
    the SNG Metadata section. Returns the .sng path.
    """
    log = log_fn or _noop_log

    ini_path = _find_one(src_folder, lambda p: os.path.basename(p).lower() == "song.ini")

    # Name the .sng exactly like the source folder (matches the original).
    folder_name = os.path.basename(os.path.abspath(src_folder)) or "song"

    base_dir = os.path.abspath(out_base) if out_base \
        else os.path.dirname(os.path.abspath(src_folder))
    out_sng = os.path.join(base_dir, f"{folder_name}.sng")
    os.makedirs(base_dir, exist_ok=True)
    log(f"  → {os.path.basename(out_sng)}\n", "info")

    # 1) embed chart + media verbatim (top level; song.ini becomes metadata).
    files: dict[str, bytes] = {}
    try:
        entries = sorted(os.listdir(src_folder))
    except OSError:
        entries = []
    chart_count = other_count = 0
    for fn in entries:
        full = os.path.join(src_folder, fn)
        if not os.path.isfile(full):
            continue
        low = fn.lower()
        # song.ini becomes the Metadata section; everything else is embedded
        # verbatim — including .bak backups, so the user can rebuild the folder.
        if low == "song.ini":
            continue
        with open(full, "rb") as f:
            # known song files are registered lowercase (per the SNG spec)
            files[low] = f.read()
        if low in _CHART_NAMES:
            chart_count += 1
        else:
            other_count += 1
    if not chart_count:
        raise FileNotFoundError("no notes.mid / notes.chart found in source folder")
    log(f"    ◇ chart: {chart_count} file(s)  ·  other: {other_count} file(s)\n", "info")

    # 2) metadata from song.ini (ordered, original-case keys).
    if ini_path:
        metadata = _read_ini_pairs(ini_path)
        log(f"    ◇ meta: {len(metadata)} song.ini field(s)\n", "info")
    else:
        metadata = [(k, v) for k, v in meta.items()] or [("name", shortname)]
        log(f"    ! meta: no song.ini — minimal metadata\n", "warn")

    # 3) pack the SNG
    sng = pack_sng(metadata, files)
    with open(out_sng, "wb") as f:
        f.write(sng)
    log(f"  ✓ {os.path.basename(out_sng)} ({len(sng):,} bytes)\n", "ok")
    return out_sng

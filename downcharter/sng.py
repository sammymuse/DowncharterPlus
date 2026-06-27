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

The chart we embed is the SAME pedal-adjusted, open-note-remapped notes.mid the other
builders produce; audio/album/background/video are copied verbatim from the source.
"""
from __future__ import annotations
import io
import os
import struct

import mido

from . import convert as _convert
from . import validate as _validate
from . import ps3build as _ps3

_parse_song_ini = _ps3._parse_song_ini
_find_one = _ps3._find_one
_find_source_mid = _ps3._find_source_mid
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
    """Ordered (key, value) pairs from a song.ini, preserving original key case
    and dropping the section header(s). Keys/values are sanitised for SNG
    (no NUL / newline; keys carry no '=')."""
    pairs = []
    seen = set()
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("[") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip().replace("=", "")
                v = v.strip()
                if not k:
                    continue
                k = "".join(c for c in k if c not in "\x00\r\n")
                v = "".join(c for c in v if c not in "\x00\r\n")
                lk = k.lower()
                if lk in seen:
                    continue
                seen.add(lk)
                pairs.append((k, v))
    except Exception:
        pass
    return pairs


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

    # ── FileIndex + masked FileData (offsets are from FileData contents start) ──
    index_body = bytearray()
    data_body = bytearray()
    offset = 0
    for name, blob in files.items():
        nb = name.encode("utf-8")
        if len(nb) > 255:
            raise ValueError(f"filename too long (>255 bytes): {name!r}")
        index_body += struct.pack("<B", len(nb)) + nb
        index_body += struct.pack("<Q", len(blob))
        index_body += struct.pack("<Q", offset)
        masked = bytearray(len(blob))
        for i, b in enumerate(blob):
            masked[i] = b ^ (xor_mask[i & 0x0F] ^ (i & 0xFF))
        data_body += masked
        offset += len(blob)

    index_section = (struct.pack("<Q", len(index_body) + 8)
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


def build_sng_song(src_folder: str, mode: str, log_fn=None, art_size: int = 512,
                   out_base: str | None = None) -> str:
    """Assemble a YARG/Clone-Hero SNG from a Downcharter-processed `src_folder`.

    Produces `<PKG>.sng` (next to the source, or under `out_base`) embedding the
    pedal-adjusted notes.mid plus the folder's audio/album/background/video. The
    song.ini becomes the SNG Metadata section. Returns the .sng path.
    """
    log = log_fn or _noop_log
    if mode not in ("1x", "2x"):
        raise ValueError(f"mode must be '1x' or '2x', got {mode!r}")

    mid_path = _find_source_mid(src_folder)
    if not mid_path:
        raise FileNotFoundError("no plain .mid found in source folder")
    ini_path = _find_one(src_folder, lambda p: os.path.basename(p).lower() == "song.ini")
    meta = _parse_song_ini(ini_path) if ini_path else {}

    src_mid = mido.MidiFile(mid_path)
    has_2x = _convert.count_double_kicks(src_mid) > 0
    name_2x = (mode == "2x" and has_2x)
    suffix = "2x" if name_2x else ""

    fallback = os.path.splitext(os.path.basename(mid_path))[0]
    shortname = _sanitize_shortname(meta, fallback, suffix)

    base_dir = os.path.abspath(out_base) if out_base \
        else os.path.dirname(os.path.abspath(src_folder))
    out_sng = os.path.join(base_dir, f"{shortname}.sng")
    os.makedirs(base_dir, exist_ok=True)
    log(f"  → {os.path.basename(out_sng)}\n", "info")

    # 1) MIDI: open-note remap + pedal variant (drum anims already in notes.mid).
    src_mid, os_stats = _convert.convert_open_notes(src_mid)
    if os_stats["converted"]:
        log(f"    ◇ mid: {os_stats['converted']} open note(s) remapped to green\n", "info")
    out_mid, ks = _convert.apply_pedal_variant(src_mid, mode)
    try:
        for level, msg in _validate.validate_rb_midi(out_mid):
            mark = "X" if level == "error" else "!"
            log(f"    {mark} check: {msg}\n", level)
    except Exception as e:
        log(f"    ! check: MIDI validation skipped ({e})\n", "warn")

    midbuf = io.BytesIO()
    out_mid.save(file=midbuf)
    files: dict[str, bytes] = {"notes.mid": midbuf.getvalue()}
    if mode == "2x":
        log(f"    ◇ mid: {ks['converted']} double-kick(s) forced to single lane\n", "info")
    else:
        log(f"    ◇ mid: {ks['removed']} double-kick(s) removed (1x playable)\n", "info")

    # 2) media: copy audio / album / background / video verbatim (top level).
    media_count = 0
    try:
        entries = sorted(os.listdir(src_folder))
    except OSError:
        entries = []
    for fn in entries:
        full = os.path.join(src_folder, fn)
        if not os.path.isfile(full):
            continue
        low = fn.lower()
        if low.endswith(_MEDIA_EXT) and not low.endswith(".bak"):
            with open(full, "rb") as f:
                files[fn] = f.read()
            media_count += 1
    log(f"    ◇ media: {media_count} file(s) embedded\n", "info")

    # 3) metadata from song.ini (ordered, original-case keys).
    if ini_path:
        metadata = _read_ini_pairs(ini_path)
        log(f"    ◇ meta: {len(metadata)} song.ini field(s)\n", "info")
    else:
        metadata = [(k, v) for k, v in meta.items()]
        if not metadata:
            metadata = [("name", shortname)]
        log(f"    ! meta: no song.ini — minimal metadata\n", "warn")

    # 4) pack the SNG
    sng = pack_sng(metadata, files)
    with open(out_sng, "wb") as f:
        f.write(sng)
    log(f"  ✓ {os.path.basename(out_sng)} ({len(sng):,} bytes)\n", "ok")
    return out_sng

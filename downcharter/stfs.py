"""stfs.py — assemble a native RB3 Xbox-360 CON/STFS package (Phase D).

The song payload is the SAME milo + dta + mogg as the PS3 build (ps3build.py);
only the container differs. PS3 = a plain folder tree; Xbox = a single STFS **CON**
file embedding a virtual filesystem with PLAIN `.mid` / `.milo_xbox` / `.png_xbox`
(STFS stores file contents unencrypted — only SHA1-hashed — so it satisfies the
"desencriptado tal como na PS3" requirement; no EDAT-style wrapping). The CON also
serves YARG on PC.

Container layout (verified byte-exactly against an Onyx-built reference CON — see
dev/stfs_inspect.py):
  * header 0x000..0xB000:  magic `CON ` + community console cert/signature
    (downcharter/data/stfs_cert.bin, 0x000..0x344) + metadata + volume descriptor.
  * data region from 0xB000 (= (header_size 0xAD0E + 0xFFF) & ~0xFFF), 0x1000 blocks.
  * "female" layout (block_separation = 1): ONE level-0 hash table per 0xAA data
    blocks (status 0x80 per data block, next-block chain), and — once total data
    blocks exceed 0xAA — one level-1 hash table (status 0x00, hashing the L0 tables)
    at backing block 0xAA. The master hash (volume descriptor) = SHA1 of the L1 block
    (or of the single L0 table when ≤ 0xAA blocks).
  * header hash @0x32C = SHA1(bytes 0x344..0xB000).

Signing: the reference is signed with the public community console cert we embed.
The package signature @0x1AC signs OUR header hash and so needs the matching RSA
private key. That key is NOT recoverable from a CON (only the public cert is), so
unless a community key is supplied the package signature is left as a placeholder:
**YARG ignores the signature and loads the CON regardless**; installing on retail
Xbox-360 hardware (RGH/JTAG) needs the community key to re-sign (hook below).
"""
from __future__ import annotations
import hashlib
import os
import struct

import mido

from . import milo as _milo
from . import convert as _convert
from . import validate as _validate
from . import mogg as _mogg
from . import art as _art
from . import ps3build as _ps3
from . import processor as _proc
from .midi_utils import build_tempo_map, tick_to_ms, to_abs

# Reuse every platform-agnostic helper from the tested PS3 path verbatim.
_parse_song_ini = _ps3._parse_song_ini
_build_dta = _ps3._build_dta
_patch_dta = _ps3._patch_dta
_dta_shortname = _ps3._dta_shortname
_sanitize_shortname = _ps3._sanitize_shortname
_pkg_folder_name = _ps3._pkg_folder_name
_audio_guided_spans = _ps3._audio_guided_spans
_extract_spans_from_track = _ps3._extract_spans_from_track
_charted_instruments = _ps3._charted_instruments
_find_one = _ps3._find_one
_find_source_mid = _ps3._find_source_mid
_find_source_mogg = _ps3._find_source_mogg
_find_source_dta = _ps3._find_source_dta

BLOCK = 0x1000
HEADER_SIZE = 0xAD0E                 # metadata-v2 header size (→ data base 0xB000)
DATA_BASE = (HEADER_SIZE + 0xFFF) & 0xFFFFF000
HASH_PER_TABLE = 0xAA                # data blocks (or sub-tables) per hash table
HASH_ENTRY = 0x18                    # 0x14 SHA1 + status byte + int24 next-block
TITLE_ID = 0x45410914               # Rock Band 3
CONTENT_TYPE = 0x00000001           # as the reference CON
NULL_BLOCK = 0xFFFFFF


def _noop_log(msg, tag=None):
    pass


def _data_path(name: str) -> str:
    """Path to a bundled data file, working in dev and onedir (sys._MEIPASS)."""
    import sys
    base = getattr(sys, "_MEIPASS", None)
    if base:
        cand = os.path.join(base, "downcharter", "data", name)
        if os.path.isfile(cand):
            return cand
    return os.path.join(os.path.dirname(__file__), "data", name)


# ── STFS block geometry (female layout; ported from Velocity StfsPackage) ──────
def _backing_data_block(block: int) -> int:
    """Logical data block → backing block index (skips inserted hash tables)."""
    ret = ((block + HASH_PER_TABLE) // HASH_PER_TABLE) + block        # sex 0 → ×1
    if block < HASH_PER_TABLE:
        return ret
    if block < 0x70E4:
        return ret + ((block + 0x70E4) // 0x70E4)
    return ret + 1


def _level0_table_block(block: int) -> int:
    """Backing block index of the level-0 hash table covering `block`."""
    if block < HASH_PER_TABLE:
        return 0
    num = (block // HASH_PER_TABLE) * 0xAB
    num += (block // 0x70E4) + 1
    return num


def _level1_table_block() -> int:
    """Backing block index of the (single) level-1 hash table (< 0x70E4 blocks).
    Sits right after the first L0 table + its 0xAA data blocks (backing 0xAB)."""
    return HASH_PER_TABLE + 1                                        # 0xAB = 171


def _block_off(backing: int) -> int:
    return DATA_BASE + backing * BLOCK


# ── file-table model ───────────────────────────────────────────────────────────
class _Entry:
    __slots__ = ("name", "is_dir", "parent", "index", "start", "blocks", "size")

    def __init__(self, name, is_dir, parent):
        self.name = name
        self.is_dir = is_dir
        self.parent = parent          # parent _Entry or None (root)
        self.index = -1               # filled when ordered
        self.start = 0
        self.blocks = 0
        self.size = 0


def _build_entries(files: dict) -> list:
    """Turn a {posix_path: bytes} map into an ordered STFS entry list (all dirs
    first in tree order, then files), mirroring the reference layout."""
    dirs: dict[str, _Entry] = {}                # path → dir _Entry

    def ensure_dir(path: str) -> _Entry | None:
        if path == "":
            return None
        if path in dirs:
            return dirs[path]
        parent = ensure_dir(path.rsplit("/", 1)[0] if "/" in path else "")
        e = _Entry(path.rsplit("/", 1)[-1], True, parent)
        dirs[path] = e
        return e

    file_entries = []
    for path, blob in files.items():
        parent_path = path.rsplit("/", 1)[0] if "/" in path else ""
        parent = ensure_dir(parent_path)
        fe = _Entry(path.rsplit("/", 1)[-1], False, parent)
        fe.size = len(blob)
        fe.blocks = max(1, (len(blob) + BLOCK - 1) // BLOCK)
        file_entries.append(fe)

    ordered = list(dirs.values()) + file_entries     # dirs first, then files
    for i, e in enumerate(ordered):
        e.index = i
    return ordered


def _serialize_file_table(entries: list) -> bytes:
    """64-byte STFS file-table entries for `entries` (start/blocks already set)."""
    out = bytearray()
    for e in entries:
        rec = bytearray(64)
        nm = e.name.encode("latin1", "replace")[:40]
        rec[0:len(nm)] = nm
        flags = len(nm) & 0x3F
        if e.is_dir:
            flags |= 0x80
        else:
            flags |= 0x40                          # contiguous blocks
        rec[0x28] = flags
        # blocks allocated (int24 LE) stored twice (alloc + actual)
        rec[0x29] = e.blocks & 0xFF
        rec[0x2A] = (e.blocks >> 8) & 0xFF
        rec[0x2B] = (e.blocks >> 16) & 0xFF
        rec[0x2C] = e.blocks & 0xFF
        rec[0x2D] = (e.blocks >> 8) & 0xFF
        rec[0x2E] = (e.blocks >> 16) & 0xFF
        # start block (int24 LE)
        rec[0x2F] = e.start & 0xFF
        rec[0x30] = (e.start >> 8) & 0xFF
        rec[0x31] = (e.start >> 16) & 0xFF
        # parent path index (int16 BE), -1 for root
        pidx = e.parent.index if e.parent is not None else -1
        struct.pack_into(">h", rec, 0x32, pidx)
        struct.pack_into(">I", rec, 0x34, e.size if not e.is_dir else 0)
        # FAT timestamps (left zero — readers don't require them)
        out += rec
    return bytes(out)


def _u16be_name(s: str, maxlen: int = 0x80) -> bytes:
    b = s.encode("utf-16-be")[:maxlen]
    return b + b"\x00" * (maxlen - len(b))


def pack_stfs(files: dict, display_name: str, title_name: str = "Rock Band 3",
              sign=None) -> bytes:
    """Pack a {posix_path: bytes} virtual filesystem into a CON/STFS package.

    `files` paths are POSIX (e.g. "songs/songs.dta"); directories are inferred.
    `sign(header_bytes, header_hash) -> bytes|None` may return a 128-byte package
    signature for retail-hardware installs; if None/absent the signature is left
    as a placeholder (YARG loads it regardless).
    """
    entries = _build_entries(files)
    blobs = {e.index: files["/".join(_path_of(e))] for e in entries if not e.is_dir}

    # ── allocate data blocks: file table first (block 0..), then each file ──────
    # The file table size depends only on the entry COUNT, so we can size it before
    # assigning start blocks (the start blocks must be known before we serialise).
    ft_blocks = max(1, (len(entries) * 64 + BLOCK - 1) // BLOCK)
    cur = ft_blocks
    for e in entries:
        if e.is_dir:
            e.start = 0
        else:
            e.start = cur
            cur += e.blocks
    total_data_blocks = cur

    # now the file table can be serialised with correct start blocks
    ft_bytes = _serialize_file_table(entries)

    # block → raw 0x1000 payload, and the next-block chain over ALL data blocks.
    block_payload: dict[int, bytes] = {}
    next_chain: dict[int, int] = {}
    for i in range(ft_blocks):
        seg = ft_bytes[i * BLOCK:(i + 1) * BLOCK]
        block_payload[i] = seg + b"\x00" * (BLOCK - len(seg))
        next_chain[i] = i + 1 if i < ft_blocks - 1 else NULL_BLOCK
    for e in entries:
        if e.is_dir:
            continue
        blob = blobs[e.index]
        for i in range(e.blocks):
            blk = e.start + i
            seg = blob[i * BLOCK:(i + 1) * BLOCK]
            block_payload[blk] = seg + b"\x00" * (BLOCK - len(seg))
            next_chain[blk] = blk + 1 if i < e.blocks - 1 else NULL_BLOCK

    # ── lay out the backing buffer (data + hash tables) ─────────────────────────
    n_l0 = (total_data_blocks + HASH_PER_TABLE - 1) // HASH_PER_TABLE
    has_l1 = total_data_blocks > HASH_PER_TABLE
    # highest backing index used = last data block's backing position
    max_backing = _backing_data_block(total_data_blocks - 1)
    for b in range(n_l0):
        max_backing = max(max_backing, _level0_table_block(b * HASH_PER_TABLE))
    if has_l1:
        max_backing = max(max_backing, _level1_table_block())
    total_backing = max_backing + 1

    buf = bytearray(DATA_BASE + total_backing * BLOCK)

    # write data blocks
    for blk, payload in block_payload.items():
        off = _block_off(_backing_data_block(blk))
        buf[off:off + BLOCK] = payload

    # ── level-0 hash tables (status 0x80, next-block chain) ─────────────────────
    for blk in range(total_data_blocks):
        toff = _block_off(_level0_table_block(blk))
        eoff = toff + (blk % HASH_PER_TABLE) * HASH_ENTRY
        sha = hashlib.sha1(block_payload[blk]).digest()
        buf[eoff:eoff + 0x14] = sha
        buf[eoff + 0x14] = 0x80
        nb = next_chain[blk]
        buf[eoff + 0x15] = (nb >> 16) & 0xFF
        buf[eoff + 0x16] = (nb >> 8) & 0xFF
        buf[eoff + 0x17] = nb & 0xFF

    # ── level-1 hash table (status 0x00, hashes each L0 table) ──────────────────
    if has_l1:
        l1off = _block_off(_level1_table_block())
        for i in range(n_l0):
            l0off = _block_off(_level0_table_block(i * HASH_PER_TABLE))
            sha = hashlib.sha1(buf[l0off:l0off + BLOCK]).digest()
            eoff = l1off + i * HASH_ENTRY
            buf[eoff:eoff + 0x14] = sha
        master_off = l1off
    else:
        master_off = _block_off(_level0_table_block(0))
    master_hash = hashlib.sha1(buf[master_off:master_off + BLOCK]).digest()

    # ── header (0x000..0xB000) ──────────────────────────────────────────────────
    header = bytearray(DATA_BASE)
    cert = open(_data_path("stfs_cert.bin"), "rb").read()       # magic + cert/sig
    header[0:len(cert)] = cert

    content_size = total_backing * BLOCK
    struct.pack_into(">I", header, 0x340, HEADER_SIZE)
    struct.pack_into(">I", header, 0x344, CONTENT_TYPE)
    struct.pack_into(">I", header, 0x348, 2)                    # metadata version
    struct.pack_into(">Q", header, 0x34C, content_size)
    struct.pack_into(">I", header, 0x354, 0)                    # media id
    struct.pack_into(">I", header, 0x360, TITLE_ID)
    header[0x364] = 0                                           # platform
    header[0x365] = 0                                           # exec type

    # volume descriptor @0x379
    vd = 0x379
    header[vd + 0x00] = 0x24
    header[vd + 0x01] = 0x00
    header[vd + 0x02] = 0x01                                    # block separation (female)
    header[vd + 0x03] = ft_blocks & 0xFF                       # file-table block count (u16 LE)
    header[vd + 0x04] = (ft_blocks >> 8) & 0xFF
    header[vd + 0x05] = 0                                       # file-table block num (int24 LE) = 0
    header[vd + 0x08:vd + 0x08 + 0x14] = master_hash
    struct.pack_into(">I", header, vd + 0x1C, total_data_blocks)   # allocated blocks
    struct.pack_into(">I", header, vd + 0x20, 0)                    # unallocated blocks

    # names + thumbnails (thumbnails optional → sizes 0)
    header[0x411:0x411 + 0x80] = _u16be_name(display_name)
    header[0x1691:0x1691 + 0x80] = _u16be_name(title_name)
    struct.pack_into(">I", header, 0x1712, 0)                  # thumbnail size
    struct.pack_into(">I", header, 0x1716, 0)                  # title thumbnail size

    # header hash @0x32C = SHA1(0x344 .. 0xB000)
    header_hash = hashlib.sha1(bytes(header[0x344:DATA_BASE])).digest()
    header[0x32C:0x32C + 0x14] = header_hash

    # package signature @0x1AC (128 bytes): re-sign for hardware if a key is given.
    if sign is not None:
        try:
            sig = sign(bytes(header), header_hash)
            if sig and len(sig) == 0x80:
                header[0x1AC:0x1AC + 0x80] = sig
        except Exception:
            pass

    buf[0:DATA_BASE] = header
    return bytes(buf)


def _path_of(e: "_Entry") -> list:
    parts = []
    cur = e
    while cur is not None:
        parts.append(cur.name)
        cur = cur.parent
    return list(reversed(parts))


# ── full Xbox CON song ──────────────────────────────────────────────────────────
def build_con_song(src_folder: str, mode: str, log_fn=None, art_size: int = 512,
                   out_base: str | None = None) -> str:
    """Assemble a native unencrypted Xbox-360 CON from a Downcharter-processed
    `src_folder`, mirroring ps3build.build_ps3_song (same milo/dta/mogg payload).

    Produces `<PKG>_rb3con` (next to the source, or under `out_base`) embedding:
        songs/songs.dta
        songs/<id>/<id>.mid           (plain, pedal-adjusted)
        songs/<id>/<id>.mogg          (verbatim)
        songs/<id>/gen/<id>.milo_xbox (our lipsync milo)
        songs/<id>/gen/<id>_keep.png_xbox (art, byte-swapped DXT)
    Returns the .con file path. Raises on missing essentials.
    """
    log = log_fn or _noop_log
    if mode not in ("1x", "2x"):
        raise ValueError(f"mode must be '1x' or '2x', got {mode!r}")

    mid_path = _find_source_mid(src_folder)
    if not mid_path:
        raise FileNotFoundError("no plain .mid found in source folder")
    mogg_path = _find_source_mogg(src_folder)
    dta_path = _find_source_dta(src_folder)
    ini_path = _find_one(src_folder, lambda p: os.path.basename(p).lower() == "song.ini")
    meta = _parse_song_ini(ini_path) if ini_path else {}

    try:
        src_mid = mido.MidiFile(mid_path)
    except (ValueError, IOError):
        src_mid = mido.MidiFile(mid_path, clip=True)   # Phase Shift 0xFF sysex
    # RB3 requires 480 TPB; normalise any non-480 source (see ps3build).
    _ps3.rescale_midi_tpb(src_mid, 480)
    has_2x = _convert.count_double_kicks(src_mid) > 0
    # Keep a raw copy for magmaPad: Onyx pads from the source BEFORE processing.
    src_mid_raw = src_mid
    name_2x = (mode == "2x" and has_2x)
    suffix = "2x" if name_2x else ""

    dta_text = ""
    if dta_path:
        with open(dta_path, "r", encoding="latin1") as f:
            dta_text = f.read()
    if dta_text and _dta_shortname(dta_text):
        shortname = _dta_shortname(dta_text)
    else:
        fallback = os.path.splitext(os.path.basename(mid_path))[0]
        shortname = _sanitize_shortname(meta, fallback, suffix)

    pkg = _pkg_folder_name(shortname, dta_text, suffix)
    base_dir = os.path.abspath(out_base) if out_base \
        else os.path.dirname(os.path.abspath(src_folder))
    out_con = os.path.join(base_dir, f"{pkg}_rb3con")
    os.makedirs(base_dir, exist_ok=True)
    log(f"  → {os.path.basename(out_con)}\n", "info")

    # 1) MIDI: open-note remap + pedal variant (drum anims already in notes.mid).
    src_mid, os_stats = _convert.convert_open_notes(src_mid)
    if os_stats["converted"]:
        log(f"    ◇ mid: {os_stats['converted']} open note(s) remapped to green\n", "info")
    out_mid, ks = _convert.apply_pedal_variant(src_mid, mode)
    # RB3 crash-safety: fix overlapping/stuck same-pitch notes + strip Phase Shift
    # sysex (open/tap markers) the YARG/CH path keeps.
    out_mid, san = _convert.sanitize_for_rb(out_mid)
    if san["overlaps_fixed"] or san["sysex_removed"] or san["tap_removed"]:
        log(f"    > mid: RB-safety - fixed {san['overlaps_fixed']} overlapping "
            f"note(s), removed {san['sysex_removed']} Phase Shift sysex, "
            f"{san['tap_removed']} tap marker(s)\n", "info")
    if san.get("ps_tracks_dropped"):
        log(f"    > mid: dropped {san['ps_tracks_dropped']} Phase-Shift-only "
            f"track(s) (e.g. PART REAL_DRUMS_PS)\n", "info")
    # Onyx no-Magma fixups: empty overdrive (fixNotelessOD) + drum [mix] events.
    out_mid, fx = _convert.apply_rb_fixups(out_mid)
    if fx["noteless_od_removed"] or fx["drum_mix_added"]:
        log(f"    > mid: removed {fx['noteless_od_removed']} empty overdrive "
            f"phrase(s), added {fx['drum_mix_added']} drum mix event(s)\n", "info")
    if fx["end_added"] or fx["beat_added"]:
        log(f"    > mid: basicTiming - "
            f"{'added [end]; ' if fx['end_added'] else ''}"
            f"{('generated BEAT track (%d beats)' % fx['beat_added']) if fx['beat_added'] else 'BEAT present'}\n",
            "info")
    if fx["unison_removed"] or fx["close_fills_removed"]:
        log(f"    > mid: fixed {fx['unison_removed']} partial-unison phrase(s), "
            f"removed {fx['close_fills_removed']} close drum fill(s)\n", "info")
    if fx.get("fills_extended") or fx.get("fills_removed"):
        log(f"    > mid: extended {fx.get('fills_extended', 0)} short drum fill(s) "
            f"to a full measure, removed {fx.get('fills_removed', 0)}"
            f" (blocked by OD)\n", "info")
    if fx["music_start_added"] or fx["music_end_added"]:
        log(f"    > mid: added "
            f"{'[music_start] ' if fx['music_start_added'] else ''}"
            f"{'[music_end]' if fx['music_end_added'] else ''}\n", "info")
    # Collapse the track_name copies each to_abs→to_track stage accumulated
    # (official mids carry exactly one per track).
    out_mid = _convert.dedupe_track_names(out_mid)
    # Lead-in pad (magmaPad): RB3 needs >=6 beats (2.6s at 120 BPM) before the
    # first gem.  Pad is computed from the RAW source (like Onyx), not the
    # processed output.  Both MIDI and audio are padded so they stay in sync.
    pad_seconds = 0.0
    pad_ticks = _convert.lead_in_pad_ticks(src_mid_raw, min_beats=6.0)
    if pad_ticks > 0:
        tpb = out_mid.ticks_per_beat
        orig_tmap = _ps3.build_tempo_map(src_mid_raw)
        first_tick = int(6.0 * tpb) - pad_ticks
        orig_first_ms = tick_to_ms(first_tick, orig_tmap, tpb)
        out_mid = _convert.pad_start(out_mid, pad_ticks)
        pad_tmap = _ps3.build_tempo_map(out_mid)
        pad_first_ms = tick_to_ms(first_tick + pad_ticks, pad_tmap, tpb)
        pad_seconds = (pad_first_ms - orig_first_ms) / 1000.0
        log(f"    > mid: padded {pad_ticks} tick(s) ({pad_seconds:.3f}s) "
            f"lead-in before the first gem\n", "info")
    charted = _charted_instruments(out_mid)
    # crash-relevant sanity gate (pack time only) — advisory, never fatal.
    try:
        for level, msg in _validate.validate_rb_midi(out_mid):
            mark = "X" if level == "error" else "!"
            log(f"    {mark} check: {msg}\n", level)
    except Exception as e:
        log(f"    ! check: MIDI validation skipped ({e})\n", "warn")
    # Ensure song_length covers the full MIDI content (see ps3build.py for
    # the detailed rationale — song.ini timestamps the audio duration, but
    # our RB3 output extends the MIDI with markers/pads).
    try:
        midi_len_ms = int(out_mid.length * 1000)
    except Exception:
        midi_len_ms = 0
    if midi_len_ms > 0:
        tmap = _ps3.build_tempo_map(out_mid)
        init_us = tmap[0][1] if tmap else 500000
        pad_ms = int(2 * out_mid.ticks_per_beat * init_us / 1_000_000)
        needed = midi_len_ms + pad_ms
        cur = _ps3._ini_int(meta, "song_length") or 0
        if needed > cur:
            meta["song_length"] = str(needed)
    elif not meta.get("song_length"):
        try:
            meta["song_length"] = str(int(out_mid.length * 1000))
        except Exception:
            pass

    import io
    midbuf = io.BytesIO()
    out_mid.save(file=midbuf)
    mid_bytes = midbuf.getvalue()
    if mode == "2x":
        log(f"    ◇ mid: {ks['converted']} double-kick(s) forced to single lane\n", "info")
    else:
        log(f"    ◇ mid: {ks['removed']} double-kick(s) removed (1x playable)\n", "info")
    log(f"    ◇ charted: {', '.join(sorted(charted)) or 'none'}\n", "info")

    files: dict[str, bytes] = {}
    base = f"songs/{shortname}"
    files[f"{base}/{shortname}.mid"] = mid_bytes

    # 2) MOGG: reuse source verbatim, else build from stems.
    mogg_layout = None
    if mogg_path:
        import tempfile
        # Re-encode to 44.1 kHz if needed (RB3 crashes at LOAD on other rates);
        # 44.1 kHz sources are copied verbatim. Channel count is preserved.
        tmp = os.path.join(tempfile.gettempdir(), f"{shortname}.mogg")
        _mogg.ensure_mogg_44100(mogg_path, tmp, log, pad_seconds=pad_seconds,
                              src_mid_raw=mid_path)
        with open(tmp, "rb") as f:
            files[f"{base}/{shortname}.mogg"] = f.read()
        try:
            os.remove(tmp)
        except OSError:
            pass
    else:
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), f"{shortname}.mogg")
        mogg_layout = _mogg.build_mogg_from_stems(src_folder, tmp, log,
                                                  pad_seconds=pad_seconds,
                                                  src_mid_raw=mid_path)
        with open(tmp, "rb") as f:
            files[f"{base}/{shortname}.mogg"] = f.read()
        try:
            os.remove(tmp)
        except OSError:
            pass

    # 3) MILO: build OUR lipsync milo (.milo_xbox body == .milo_ps3 body).
    #     Multi-entry for PART VOCALS + HARM1/2/3 — same as the PS3 path.
    try:
        vocal_tracks = ["PART VOCALS", "HARM1", "HARM2", "HARM3"]
        all_spans: list[list] = []
        for tr_name in vocal_tracks:
            spans = _extract_spans_from_track(out_mid, tr_name, src_folder)
            all_spans.append(spans)

        phrase_ends_s: list[float] = []
        vocal_notes: list[tuple[float, int]] = []
        pv = next((tr for tr in out_mid.tracks
                   if (tr.name or "").strip().upper() == "PART VOCALS"), None)
        if pv is not None:
            tpb = out_mid.ticks_per_beat
            tempo_map = build_tempo_map(out_mid)
            abs_pv = to_abs(pv)
            pe_ticks = _proc._abs_phrase_ends(list(abs_pv))
            phrase_ends_s = [tick_to_ms(t, tempo_map, tpb) / 1000.0
                             for t in pe_ticks]
            for e in abs_pv:
                m = e.msg
                n = getattr(m, "note", None)
                if (m.type == "note_on" and getattr(m, "velocity", 0) > 0
                        and n is not None and 36 <= n <= 84):
                    sec = tick_to_ms(e.abs_tick, tempo_map, tpb) / 1000.0
                    vocal_notes.append((sec, n))

        lipsync_list = _milo.build_multi_lipsync(
            all_spans, out_mid.length,
            phrase_ends=phrase_ends_s,
            vocal_notes=vocal_notes,
        )
        if lipsync_list:
            files[f"{base}/gen/{shortname}.milo_xbox"] = \
                _milo.build_milo(lipsync_list)
            n_entries = len(lipsync_list)
            entry_hint = f" ({n_entries} entry)" if n_entries > 1 else ""
            total_spans = sum(len(s) for s in all_spans if s)
            log(f"    ◇ milo: built from {total_spans} audio-guided syllable(s)"
                f"{entry_hint}\n", "info")
        else:
            src_milo = _find_one(src_folder, lambda p: p.lower().endswith((".milo_xbox", ".milo_ps3")))
            if src_milo:
                with open(src_milo, "rb") as f:
                    files[f"{base}/gen/{shortname}.milo_xbox"] = f.read()
                log(f"    ◇ milo: no charted vocals — reused source milo\n", "info")
            else:
                log(f"    ! milo: no charted vocals — skipped (no lipsync)\n", "warn")
    except Exception as e:
        log(f"    ! milo: lipsync build failed ({e}) — skipped\n", "warn")

    # 4) Album art → .png_xbox (DXT byte-swapped vs .png_ps3).
    has_art = False
    src_png = _find_one(src_folder, lambda p: p.lower().endswith(".png_xbox"))
    if src_png:
        with open(src_png, "rb") as f:
            files[f"{base}/gen/{shortname}_keep.png_xbox"] = f.read()
        has_art = True
        log(f"    ◇ art: copied (.png_xbox)\n", "info")
    else:
        cover = _art.find_cover(src_folder)
        if cover and _art.available():
            try:
                files[f"{base}/gen/{shortname}_keep.png_xbox"] = \
                    _art.build_png_xbox(cover, art_size)
                has_art = True
                log(f"    ◇ art: generated from {os.path.basename(cover)} "
                    f"({art_size}×{art_size})\n", "info")
            except Exception as e:
                log(f"    ! art: cover convert failed ({e}) — skipped\n", "warn")
        else:
            log(f"    ! art: no cover image found — skipped\n", "warn")

    # 5) songs.dta
    if dta_text:
        out_dta = _patch_dta(dta_text, shortname, name_2x)
        files["songs/songs.dta"] = out_dta.encode("latin1", "replace")
        log(f"    ◇ dta: patched ({'2x' if name_2x else '1x'})\n", "info")
    elif mogg_layout is not None:
        out_dta, dta_codec = _build_dta(meta, shortname, mogg_layout, name_2x,
                                        charted, has_art=has_art, out_mid=out_mid)
        files["songs/songs.dta"] = out_dta.encode(dta_codec, "replace")
        log(f"    ◇ dta: generated from song.ini ({'2x' if name_2x else '1x'})\n", "info")
    else:
        log(f"    ! dta: no songs.dta and no built mogg layout — skipped\n", "warn")

    # 6) pack the CON
    title = _ps3._dta_str(meta.get("name") or shortname)
    artist = _ps3._dta_str(meta.get("artist") or "")
    disp = f"{artist} - {title}" if artist else title
    con = pack_stfs(files, disp)
    with open(out_con, "wb") as f:
        f.write(con)
    log(f"  ✓ {os.path.basename(out_con)} ({len(con):,} bytes)\n", "ok")
    return out_con

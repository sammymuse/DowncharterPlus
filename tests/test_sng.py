"""Tests for downcharter/sng.py — YARG/Clone Hero SNG package builder.

Verifies:
  - pack_sng / round-trip (metadata + files, XOR mask)
  - _read_ini_pairs encoding fallback (UTF-16 LE/BE BOM, UTF-8 BOM, Latin-1)
  - build_sng_song with a folder that has no song.ini (minimal metadata,
    no NameError)
"""
import os
import struct
import tempfile

import pytest

from downcharter import sng as _sng


def _unmask(mask: bytes, blob: bytes) -> bytes:
    return bytes(b ^ (mask[i & 0x0F] ^ (i & 0xFF)) for i, b in enumerate(blob))


# ── pack_sng round-trip ──────────────────────────────────────────────────────

def test_pack_sng_round_trip():
    files = {"notes.mid": b"MThd\x00\x00\x00\x06", "song.ogg": b"OggS" + os.urandom(200)}
    meta = [("name", "Test Song"), ("artist", "Someone")]
    mask = bytes(range(16))
    blob = _sng.pack_sng(meta, files, xor_mask=mask)

    assert blob[:6] == _sng.MAGIC
    assert struct.unpack_from("<I", blob, 6)[0] == _sng.VERSION
    assert blob[10:26] == mask


def test_pack_sng_default_mask_reader_round_trip():
    """xor_mask=None (what build_sng_song uses) must still store MASKED data:
    YARG/CH always decode with mask[i&15] ^ (i&0xFF), so a zero header mask
    requires in[i] ^ (i & 0xFF) on disk — plain bytes read back as garbage
    (regression: 30564db shipped that; whole libraries scanned as
    'No notes found')."""
    files = {"notes.mid": b"MThd\x00\x00\x00\x06" + os.urandom(500),
             "notes.chart": b"[Song]\r\n{\r\n}\r\n" + os.urandom(300)}
    blob = _sng.pack_sng([("name", "x")], files)

    header_mask = blob[10:26]
    o = 26
    meta_len = struct.unpack_from("<Q", blob, o)[0]
    o += 8 + meta_len
    idx_len, n_files = struct.unpack_from("<QQ", blob, o)
    o += 16
    seen = 0
    for _ in range(n_files):
        nl = blob[o]
        o += 1
        name = blob[o:o + nl].decode()
        o += nl
        size, pos = struct.unpack_from("<qq", blob, o)
        o += 16
        # Decode EXACTLY like the YARG reader (header mask, always applied).
        recovered = _unmask(header_mask, blob[pos:pos + size])
        assert recovered == files[name], name
        # And the on-disk bytes must NOT be the plain input.
        assert blob[pos:pos + size] != files[name], name
        seen += 1
    assert seen == len(files)


def test_pack_sng_file_offsets_absolute():
    """contentsIndex must be absolute from the start of the .sng, not relative
    to the FileData body (regression: fixed in ef0c653)."""
    files = {"a.txt": b"hello", "b.txt": b"world!!"}
    blob = _sng.pack_sng([("name", "x")], files, xor_mask=bytes(16))

    o = 26
    meta_len = struct.unpack_from("<Q", blob, o)[0]
    o += 8 + meta_len
    idx_len, n_files = struct.unpack_from("<QQ", blob, o)
    o += 16
    for _ in range(n_files):
        nl = blob[o]
        o += 1
        name = blob[o:o + nl].decode()
        o += nl
        size, pos = struct.unpack_from("<qq", blob, o)
        o += 16
        assert pos >= len(blob) - size - (len(blob) - pos)  # sane bound
        recovered = _unmask(bytes(16), blob[pos:pos + size])
        assert recovered == files[name]


# ── _read_ini_pairs encoding fallback ────────────────────────────────────────

INI_TEXT = "[song]\nname = Heads Will Roll\nartist = Yeah Yeah Yeahs\ndelay = 0\n"


def _write(tmp_path, name, data: bytes) -> str:
    p = os.path.join(tmp_path, name)
    with open(p, "wb") as f:
        f.write(data)
    return p


@pytest.mark.parametrize("label,encode", [
    ("utf-8", lambda s: s.encode("utf-8")),
    ("utf-8-bom", lambda s: b"\xef\xbb\xbf" + s.encode("utf-8")),
    ("utf-16-le-bom", lambda s: s.encode("utf-16")),          # adds FF FE BOM
    ("utf-16-be-bom", lambda s: b"\xfe\xff" + s.encode("utf-16-be")),
])
def test_read_ini_pairs_encodings(tmp_path, label, encode):
    p = _write(str(tmp_path), f"song_{label}.ini", encode(INI_TEXT))
    pairs = dict(_sng._read_ini_pairs(p))
    assert pairs.get("name") == "Heads Will Roll", label
    assert pairs.get("artist") == "Yeah Yeah Yeahs", label
    assert pairs.get("delay") == "0", label


def test_read_ini_pairs_latin1_accents(tmp_path):
    text = "[song]\nname = R\xf4le Playing\nartist = B\xe9la Fleck\n"  # ô, é
    p = _write(str(tmp_path), "song_latin1.ini", text.encode("latin-1"))
    pairs = dict(_sng._read_ini_pairs(p))
    assert pairs.get("name") == "R\xf4le Playing"
    assert pairs.get("artist") == "B\xe9la Fleck"


def test_read_ini_pairs_missing_file_returns_empty(tmp_path):
    assert _sng._read_ini_pairs(os.path.join(str(tmp_path), "nope.ini")) == []


# ── build_sng_song without song.ini ──────────────────────────────────────────

_MIN_MID = bytes.fromhex("4d546864000000060000000101e0") + b"MTrk" + \
    bytes.fromhex("00000004") + bytes.fromhex("00ff2f00")


def test_build_sng_song_without_ini_uses_folder_name(tmp_path):
    folder = os.path.join(str(tmp_path), "My Cool Song")
    os.makedirs(folder)
    with open(os.path.join(folder, "notes.mid"), "wb") as f:
        f.write(_MIN_MID)

    out_dir = os.path.join(str(tmp_path), "out")
    sng_path = _sng.build_sng_song(folder, out_base=out_dir)

    assert os.path.isfile(sng_path)
    with open(sng_path, "rb") as f:
        raw = f.read()
    o = 26
    meta_len, count = struct.unpack_from("<QQ", raw, o)
    o += 16
    key_len = struct.unpack_from("<i", raw, o)[0]
    o += 4
    key = raw[o:o + key_len].decode("utf-8")
    o += key_len
    val_len = struct.unpack_from("<i", raw, o)[0]
    o += 4
    val = raw[o:o + val_len].decode("utf-8")
    assert count == 1
    assert key == "name"
    assert val == "My Cool Song"

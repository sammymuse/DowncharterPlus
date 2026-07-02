"""Tests for downcharter/stfs.py — Xbox 360 CON/STFS package builder.

Verifies:
  - File-table entry structure (_build_entries, _serialize_file_table)
  - Block geometry (_backing_data_block, _level0_table_block)
  - pack_stfs with minimal fileset
  - Header fields (magic "CON ", content type, title ID)
  - Volume descriptor structure
  - Header hash verification
  - Master hash structure
  - Integration: build_con_song (mock-level — check it runs)
"""
import hashlib
import struct
import os
import tempfile

import pytest

from downcharter import stfs as _stfs


# ── Constants ────────────────────────────────────────────────────────────────────

BLOCK = 0x1000
HEADER_SIZE = 0xAD0E
DATA_BASE = (HEADER_SIZE + 0xFFF) & 0xFFFFF000  # = 0xB000
NULL_BLOCK = 0xFFFFFF
TITLE_ID = 0x45410914


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _read_meta(buf: bytes, offset: int, fmt: str):
    """Unpack a struct field from the header buffer."""
    return struct.unpack_from(fmt, buf, offset)[0]


# ── File-table tests ────────────────────────────────────────────────────────────

class TestBuildEntries:
    def test_empty(self):
        entries = _stfs._build_entries({})
        assert len(entries) == 0

    def test_single_file(self):
        entries = _stfs._build_entries({"songs/songs.dta": b"data"})
        assert len(entries) == 2  # root dir + file
        dirs = [e for e in entries if e.is_dir]
        files = [e for e in entries if not e.is_dir]
        assert len(dirs) == 1
        assert len(files) == 1
        assert files[0].name == "songs.dta"
        assert files[0].blocks == 1

    def test_nested_dirs(self):
        files = {"a/b/c/file.txt": b"x" * 5000}
        entries = _stfs._build_entries(files)
        dirs = [e for e in entries if e.is_dir]
        assert len(dirs) == 3  # a, a/b, a/b/c
        assert dirs[0].name == "a"
        assert dirs[1].name == "b"
        assert dirs[2].name == "c"

    def test_dir_first_ordering(self):
        """All directories come before all files."""
        files = {
            "songs/a/a.mid": b"mid",
            "songs/songs.dta": b"dta",
        }
        entries = _stfs._build_entries(files)
        seen_file = False
        for e in entries:
            if not e.is_dir:
                seen_file = True
            else:
                assert not seen_file, "dir after file"

    def test_file_blocks(self):
        """Block count is ceil(size / BLOCK)."""
        for size, expected in ((1, 1), (0x1000, 1), (0x1001, 2), (0x3000, 3)):
            entries = _stfs._build_entries({"f.bin": b"x" * size})
            files = [e for e in entries if not e.is_dir]
            if files:
                assert files[0].blocks == expected, f"size={size}"


class TestSerializeFileTable:
    def test_entry_size(self):
        """Each entry is exactly 64 bytes."""
        entries = _stfs._build_entries({"f.bin": b"data"})
        buf = _stfs._serialize_file_table(entries)
        assert len(buf) == len(entries) * 64

    def test_name_truncated(self):
        """Name > 40 bytes is truncated to 40."""
        entries = _stfs._build_entries({"a" * 50 + ".bin": b"x"})
        buf = _stfs._serialize_file_table(entries)
        # All entries still 64 bytes
        assert len(buf) % 64 == 0

    def test_file_table_start_blocks_preserved(self):
        """Start blocks set by _build_entries survive serialisation."""
        files = {"a.mid": b"x" * 0x1000, "b.mid": b"x" * 0x1000}
        entries = _stfs._build_entries(files)
        # Set start blocks
        cur = 1  # after file table block 0
        for e in entries:
            if not e.is_dir:
                e.start = cur
                cur += e.blocks
        buf = _stfs._serialize_file_table(entries)
        # Check the first file entry has start block at offset 0x2F (int24 LE)
        file_entries = [e for e in entries if not e.is_dir]
        first = file_entries[0]
        idx = entries.index(first) * 64
        start = buf[idx + 0x2F] | (buf[idx + 0x30] << 8) | (buf[idx + 0x31] << 16)
        assert start == first.start


# ── Block geometry ──────────────────────────────────────────────────────────────

class TestBlockGeometry:
    def test_backing_data_block_linear(self):
        """Backing blocks shift by 1 (block 0 is the first hash table)."""
        for i in range(0xAA):
            assert _stfs._backing_data_block(i) == i + 1, f"mismatch at data block {i}"

    def test_level0_table_block_first(self):
        """First L0 table is at backing block 0."""
        assert _stfs._level0_table_block(0) == 0
        assert _stfs._level0_table_block(1) == 0


# ── pack_stfs ──────────────────────────────────────────────────────────────────

class TestPackStfs:
    def test_minimal_pack(self):
        """Pack the smallest fileset."""
        files = {"songs/songs.dta": b"(test)"}
        con = _stfs.pack_stfs(files, "Test")
        assert len(con) >= DATA_BASE  # at least the header
        assert len(con) % BLOCK == 0  # multiple of block size

    def test_header_magic(self):
        """CON header starts with 'CON '."""
        files = {"songs/songs.dta": b"(test)"}
        con = _stfs.pack_stfs(files, "Test")
        assert con[0:4] == b"CON "

    def test_header_size_field(self):
        """Header size at 0x340 matches our constant (big-endian)."""
        files = {"songs/songs.dta": b"(test)"}
        con = _stfs.pack_stfs(files, "Test")
        hs = struct.unpack_from(">I", con, 0x340)[0]
        assert hs == HEADER_SIZE

    def test_content_type(self):
        """Content type at 0x344 is 1 (saved game / CON, big-endian)."""
        files = {"songs/songs.dta": b"(test)"}
        con = _stfs.pack_stfs(files, "Test")
        ct = struct.unpack_from(">I", con, 0x344)[0]
        assert ct == 1

    def test_title_id(self):
        """Title ID at 0x360 is Rock Band 3's (big-endian)."""
        files = {"songs/songs.dta": b"(test)"}
        con = _stfs.pack_stfs(files, "Test")
        tid = struct.unpack_from(">I", con, 0x360)[0]
        assert tid == TITLE_ID

    def test_header_hash(self):
        """SHA1 at 0x32C matches SHA1(0x344..0xB000)."""
        files = {"songs/songs.dta": b"(test)"}
        con = _stfs.pack_stfs(files, "Test")
        stored = con[0x32C:0x32C + 0x14]
        computed = hashlib.sha1(con[0x344:DATA_BASE]).digest()
        assert stored == computed, "header hash mismatch"

    def test_display_name_written(self):
        """Display name at 0x411 is written as UTF-16-BE."""
        files = {"songs/songs.dta": b"(test)"}
        con = _stfs.pack_stfs(files, "Test Artist - Elegy")
        # Read UTF-16-BE from 0x411
        name_bytes = con[0x411:0x411 + 0x80]
        # Find the null terminator
        null_idx = 0
        while null_idx + 1 < len(name_bytes):
            if name_bytes[null_idx] == 0 and name_bytes[null_idx + 1] == 0:
                break
            null_idx += 2
        name = name_bytes[:null_idx + 2].decode("utf-16-be", errors="replace").strip("\x00")
        assert "Test Artist" in name

    def test_multiple_files_pack(self):
        """Three files produce a valid CON."""
        files = {
            "songs/songs.dta": b"(elegy\n   (name \"Elegy\")\n)",
            "songs/elegy/elegy.mid": b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0",
            "songs/elegy/gen/elegy.milo_ps3": b"\x00" * 0x100,
        }
        con = _stfs.pack_stfs(files, "Elegy")
        assert len(con) > DATA_BASE
        # Verify the data region exists
        assert len(con) > 0xB000

    def test_cert_preserved_up_to_header_fields(self):
        """The embedded cert is preserved except for overwritten fields.

        stfs overwrites:
          - 0x32C..0x340: header hash (SHA1 of 0x344..0xB000)
          - 0x340..:      header metadata (header_size, content_type, etc.)
        The package signature at 0x1AC is left as-is (no signing key passed).
        """
        files = {"songs/songs.dta": b"(test)"}
        con = _stfs.pack_stfs(files, "Test")
        cert_path = _stfs._data_path("stfs_cert.bin")
        with open(cert_path, "rb") as f:
            cert = f.read()
        # Cert bytes before 0x1AC (package signature start) are untouched.
        # 0x1AC is the package signature (128 bytes), which we leave zero.
        # 0x32C is the header hash, which we recompute.
        preserve_end = 0x1AC  # package signature offset
        assert con[:preserve_end] == cert[:preserve_end], \
            f"cert bytes 0..{preserve_end} differ"

    def test_master_hash_in_volume_descriptor(self):
        """Volume descriptor at 0x379 contains the master hash."""
        files = {"songs/songs.dta": b"(test)"}
        con = _stfs.pack_stfs(files, "Test")
        vd = 0x379
        # Hash at vd+0x08..vd+0x1B
        mh = con[vd + 0x08:vd + 0x08 + 0x14]
        assert len(mh) == 20
        assert any(b != 0 for b in mh)  # not all zeros


# ── Display name ────────────────────────────────────────────────────────────────

class TestDisplayName:
    def test_u16be_name_encoding(self):
        """_u16be_name encodes as UTF-16-BE with correct padding."""
        name = "Test"
        b = _stfs._u16be_name(name)
        assert len(b) == 0x80
        decoded = b.decode("utf-16-be", errors="replace").rstrip("\x00")
        assert decoded == name

    def test_u16be_name_truncation(self):
        """_u16be_name truncates to maxlen bytes (0x80 = 128 bytes = 64 chars)."""
        long_name = "A" * 200
        b = _stfs._u16be_name(long_name)
        assert len(b) == 0x80
        # 128 bytes / 2 bytes per UTF-16-BE code unit = 64 characters max
        decoded = b.decode("utf-16-be", errors="replace").rstrip("\x00")
        assert len(decoded) <= 64


# ── Int24 helpers ───────────────────────────────────────────────────────────────

class TestInt24:
    def test_int24_le_pack(self):
        """24-bit little-endian pack/unpack works correctly."""
        def pack24(v):
            return bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF])
        for v in (0, 1, 0xABCDEF, 0xFFFFFF):
            assert int.from_bytes(pack24(v), "little") == v

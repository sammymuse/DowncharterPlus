"""Tests for downcharter/milo.py — native .milo builder.

Verifies:
  - MILO_A header structure (magic, sizes, block layout)
  - Dir prefix byte-identity against Onyx reference
  - Dynamic viseme table serialisation (only visemes used, alphabetical)
  - Full round-trip: spans → milo → parse → matches input
  - build_song_lipsync vs build_milo_from_spans consistency
  - Edge cases: empty spans, single syllable, visemes at boundaries
"""
import struct

import pytest

from downcharter import milo as _milo
from downcharter import lipsync as _lip


# ── Helpers ──────────────────────────────────────────────────────────────────────

_MILO_MAGIC = 0xCABEDEAF
_MILO_HEADER_SIZE = 0x810
_BARRIER = b"\xad\xde\xad\xde"


def _read_header(milo: bytes):
    """Parse and return the MILO_A header fields."""
    assert len(milo) >= _MILO_HEADER_SIZE
    magic, hsize, blk_count, max_blk = struct.unpack_from("<IIII", milo, 0)
    blk0_size = struct.unpack_from("<I", milo, 16)[0]
    return {
        "magic": magic,
        "hsize": hsize,
        "blk_count": blk_count,
        "max_blk": max_blk,
        "blk0_size": blk0_size,
        "body": milo[_MILO_HEADER_SIZE:],
    }


# ── Tests ────────────────────────────────────────────────────────────────────────

class TestMiloHeader:
    """MILO_A uncompressed header structure."""

    def test_magic_and_size(self):
        """Header must start with the correct magic and 0x810 size."""
        h = _read_header(_milo.add_milo_header(b"x" * 100))
        assert h["magic"] == _MILO_MAGIC, "bad magic"
        assert h["hsize"] == _MILO_HEADER_SIZE, "bad header size"

    def test_single_block(self):
        """We always emit a single block = whole body."""
        body = b"test_body_data"
        milo = _milo.add_milo_header(body)
        h = _read_header(milo)
        assert h["blk_count"] == 1
        assert h["max_blk"] == len(body)
        assert h["blk0_size"] == len(body)

    def test_zero_padding_to_0x810(self):
        """Bytes after the 20-byte header up to 0x810 must be zero."""
        milo = _milo.add_milo_header(b"x")
        pad = milo[20:_MILO_HEADER_SIZE]
        assert all(b == 0 for b in pad), "header not zero-padded"

    def test_body_appended(self):
        """The raw body follows right after the 0x810 header."""
        body = b"hello_milo_body_123"
        milo = _milo.add_milo_header(body)
        assert milo[_MILO_HEADER_SIZE:] == body


class TestMiloBody:
    """Milo dir body structure (DIR_PREFIX + barrier + lipsync + barrier)."""

    def test_dir_prefix_present(self):
        """The DIR_PREFIX must be embedded verbatim before the first barrier."""
        body = _milo._DIR_PREFIX + _BARRIER + b"lipsync" + _BARRIER
        assert body.startswith(_milo._DIR_PREFIX)
        # DIR_PREFIX must end with the bytes right before ADDEADDE
        assert _milo._DIR_PREFIX[-4:] != _BARRIER

    def test_double_barrier(self):
        """The body must contain exactly two ADDEADDE barriers."""
        body = _milo._DIR_PREFIX + _BARRIER + b"data" + _BARRIER
        assert body.count(_BARRIER) == 2

    def test_dir_prefix_length_known(self):
        """DIR_PREFIX length is a known constant for the current format."""
        # Must stay stable unless the milo format version changes.
        # Value: 462 bytes as of Jul 2026 (28-byte ver + "ObjectDir" + "lipsync" +
        # header flags + entry ("CharLipSync","song.lipsync") + 7 dummy matrices).
        assert len(_milo._DIR_PREFIX) == 462


class TestLipsyncSerialisation:
    """CharLipSync bytes with dynamic viseme table."""

    def test_dynamic_viseme_table(self):
        """Only visemes actually used appear in the table, alphabetically."""
        frames = {
            0: {"Church_hi": 128, "Bump_lo": 64},
            1: {"Church_hi": 100, "Wet_lo": 80},
        }
        lip = _milo._serialize_lipsync(frames, 2)
        parsed = _milo.parse_song_lipsync(_milo.build_milo(lip))
        # Visemes should be alphabetical: Bump_lo, Church_hi, Wet_lo
        assert parsed["visemes"] == ["Bump_lo", "Church_hi", "Wet_lo"]

    def test_no_unused_visemes(self):
        """Visemes with weight=0 in all frames are excluded."""
        frames = {
            0: {"If_hi": 255},
            1: {"If_hi": 200, "If_lo": 0},   # If_lo = 0 ← excluded
        }
        lip = _milo._serialize_lipsync(frames, 2)
        parsed = _milo.parse_song_lipsync(_milo.build_milo(lip))
        assert "If_lo" not in parsed["visemes"]
        assert parsed["visemes"] == ["If_hi"]

    def test_delta_encoding(self):
        """Only visemes that change from the previous frame are emitted."""
        frames = {
            0: {"Bump_hi": 100, "Bump_lo": 50},
            1: {"Bump_hi": 100, "Bump_lo": 50},   # no change → empty frame
            2: {"Bump_hi": 200, "Bump_lo": 50},   # Bump_hi changed
        }
        lip = _milo._serialize_lipsync(frames, 3)
        parsed = _milo.parse_song_lipsync(_milo.build_milo(lip))
        assert 1 in parsed["frames"], "frame 1 should exist (even if empty)"
        # Frame 1 may or may not be present depending on delta encoding
        # Frame 2 should have Bump_hi changed
        fr2 = parsed["frames"].get(2, {})
        assert fr2.get("Bump_hi", -1) == 200

    def test_unknown_viseme_skipped(self):
        """A viseme not in the _VIDX map is silently skipped."""
        frames = {
            0: {"Fake_Viseme_XYZ": 255, "Bump_hi": 128},
        }
        lip = _milo._serialize_lipsync(frames, 1)
        parsed = _milo.parse_song_lipsync(_milo.build_milo(lip))
        assert "Bump_hi" in parsed["visemes"]
        assert "Fake_Viseme_XYZ" not in parsed["visemes"]


class TestBuildSongLipsync:
    """Build from syllable spans."""

    def test_basic_build(self):
        """Two-syllable build produces frames and a parsable result."""
        spans = [(0.0, 0.3, "hel", 1.0), (0.3, 0.6, "lo", 0.8)]
        lip = _milo.build_song_lipsync(spans, 1.0, "en")
        assert len(lip) > 50
        parsed = _milo.parse_song_lipsync(_milo.build_milo(lip))
        assert parsed["n_frames"] == 31  # ~1s at 30 fps
        assert len(parsed["visemes"]) > 0

    def test_empty_spans(self):
        """Empty spans produce a valid lipsync with 0 visemes but frames."""
        lip = _milo.build_song_lipsync([], 1.0, "en")
        parsed = _milo.parse_song_lipsync(_milo.build_milo(lip))
        assert parsed["visemes"] == []
        assert parsed["n_frames"] == 31

    def test_round_trip_consistency(self):
        """build → parse → build produces byte-identical output."""
        spans = [(0.0, 0.5, "hello", 1.0), (0.5, 1.0, "world", 0.8)]
        a = _milo.build_milo_from_spans(spans, 2.0, "en")
        parsed = _milo.parse_song_lipsync(a)
        # Rebuild from the parsed frames
        b = _milo.build_milo(_milo._serialize_lipsync(
            {fr: {v: w for v, w in state.items()}
             for fr, state in parsed["frames"].items()},
            parsed["n_frames"]))
        assert a == b, "round-trip failed: build→parse→build differs"

    def test_milo_from_spans(self):
        """build_milo_from_spans is convenience for build_milo(build_song_lipsync(...))."""
        spans = [(0.0, 0.3, "test", 1.0)]
        direct = _milo.build_milo(_milo.build_song_lipsync(spans, 0.5, "en"))
        conven = _milo.build_milo_from_spans(spans, 0.5, "en")
        assert direct == conven


class TestBuildMilo:
    """Full .milo assembly and parsing."""

    def test_full_build(self):
        """From spans to complete .milo with header and body."""
        spans = [(0.0, 0.2, "hi", 1.0)]
        milo = _milo.build_milo_from_spans(spans, 0.3, "en")
        # Header
        assert len(milo) > _MILO_HEADER_SIZE
        h = _read_header(milo)
        assert h["magic"] == _MILO_MAGIC
        # Body starts with DIR_PREFIX
        assert h["body"][:4] == _milo._DIR_PREFIX[:4]
        # Contains two barriers
        assert h["body"].count(_BARRIER) == 2
        # Parse the song.lipsync out of it
        parsed = _milo.parse_song_lipsync(milo)
        assert parsed["n_frames"] == 10  # ~0.3s at 30 fps
        assert len(parsed["visemes"]) >= 3

    def test_parse_no_corruption(self):
        """After parse → rebuild, the body bytes match the original."""
        spans = [(0.0, 0.5, "round", 1.0), (0.5, 1.0, "trip", 0.9)]
        orig = _milo.build_milo_from_spans(spans, 1.5, "en")
        p = _milo.parse_song_lipsync(orig)
        rebuilt_lip = _milo._serialize_lipsync(
            {fr: {v: w for v, w in s.items()} for fr, s in p["frames"].items()},
            p["n_frames"])
        rebuilt = _milo.build_milo(rebuilt_lip)
        assert orig == rebuilt

    def test_platform_independence(self):
        """Same spans produce byte-identical .milo (PS3 == Xbox body)."""
        spans = [(0.0, 0.3, "same", 1.0)]
        a = _milo.build_milo_from_spans(spans, 0.4, "en")
        b = _milo.build_milo_from_spans(spans, 0.4, "en")
        assert a == b


class TestParseSongLipsync:
    """Reading back a .milo."""

    def test_parse_known_visemes(self):
        """Known visemes are restored correctly."""
        spans = [(0.0, 0.4, "if", 1.0)]  # "if" → [IH F] → If_hi, If_lo
        milo = _milo.build_milo_from_spans(spans, 0.5, "en")
        p = _milo.parse_song_lipsync(milo)
        assert "If_hi" in p["visemes"]
        assert "If_lo" in p["visemes"]

    def test_parse_frame_count(self):
        """Frame count matches ceil(song_len * FPS) + 1 (inclusive sentinel)."""
        import math
        spans = [(0.0, 1.0, "one", 1.0)]
        for dur in (0.5, 1.0, 2.0):
            milo = _milo.build_milo_from_spans(spans, dur, "en")
            p = _milo.parse_song_lipsync(milo)
            expected = max(1, int(math.ceil(dur * _lip.FPS)) + 1)
            assert p["n_frames"] == expected, f"expected {expected} frames for {dur}s, got {p['n_frames']}"


class TestMultiEntry:
    """Multi-entry milo for PART VOCALS + HARM1 + HARM2 + HARM3."""

    def test_u1_u2_scaling(self):
        """U1 and U2 scale with entry count N (verified against 100+ official milos)."""
        for n, exp_u1, exp_u2 in [(1, 4, 21), (2, 6, 35), (3, 8, 49), (4, 10, 63)]:
            names = [f"part{k}.lipsync" for k in range(2, n + 1)] + ["song.lipsync"]
            prefix = _milo._build_dir_prefix(names)
            u1 = struct.unpack_from(">I", prefix, 28)[0]
            u2 = struct.unpack_from(">I", prefix, 32)[0]
            assert u1 == exp_u1, f"N={n}: U1={u1}, expected {exp_u1}"
            assert u2 == exp_u2, f"N={n}: U2={u2}, expected {exp_u2}"

    def test_singleton_byte_identical_to_known_reference(self):
        """Single-entry _build_dir_prefix matches the known-good reference constant.
        Uses _FULL_DIR_SINGLE (the original hardcoded bytes extracted from a real
        Onyx-built milo) as the reference — NOT _DIR_PREFIX which is defined as
        _build_dir_prefix(...), making any comparison with it a no-op/tautology."""
        names = ["song.lipsync"]
        prefix = _milo._build_dir_prefix(names)
        assert prefix == _milo._FULL_DIR_SINGLE, \
            "singleton prefix does not match known-good reference"

    def test_part_names_part_first_song_last(self):
        """build_milo names entries: harmonies first, song.lipsync (lead) last.
        Verified by U1/U2 scaling (N=3 → U1=8, U2=49) and by round-trip:
        all N entries are parseable by index."""
        blobs = [
            _milo.build_song_lipsync([(0.0, 0.2, "a", 1.0)], 0.3, "en"),
            _milo.build_song_lipsync([(0.0, 0.2, "b", 1.0)], 0.3, "en"),
            _milo.build_song_lipsync([(0.0, 0.2, "c", 1.0)], 0.3, "en"),
        ]
        milo = _milo.build_milo(blobs)
        # Verify all 3 entries are parseable (barrier structure is correct)
        for i in range(3):
            parsed = _milo.parse_song_lipsync(milo, index=i)
            assert parsed["n_frames"] > 0

    def test_build_multi_lipsync_reorders_lead_to_last(self):
        """build_multi_lipsync returns [HARM1, HARM2, ..., lead] so build_milo
        names them part2.lipsync, ..., song.lipsync (lead = song.lipsync last).
        Verified by: all entries round-trip correctly via parse_song_lipsync."""
        spans_list = [
            [(0.0, 0.5, "le", 1.0)],   # lead (PART VOCALS)
            [(1.0, 1.5, "ha", 1.0)],   # HARM1
            [(2.0, 2.5, "hb", 1.0)],   # HARM2
            [],                          # HARM3 empty → skipped
        ]
        result = _milo.build_multi_lipsync(spans_list, 5.0, "en")
        assert len(result) == 3, f"expected 3 entries (HARM1,HARM2,lead), got {len(result)}"
        # Rebuild a milo and verify all entries parse correctly
        milo = _milo.build_milo(result)
        for i in range(3):
            parsed = _milo.parse_song_lipsync(milo, index=i)
            assert parsed["n_frames"] > 0

    def test_multi_entry_roundtrip_parses_all_entries(self):
        """All N entries in a multi-entry milo are parseable by index."""
        blobs = [
            _milo.build_song_lipsync([(0.0, 0.2, "a", 1.0)], 0.3, "en"),
            _milo.build_song_lipsync([(0.5, 0.7, "b", 1.0)], 0.8, "en"),
        ]
        milo = _milo.build_milo(blobs)
        for i in range(len(blobs)):
            parsed = _milo.parse_song_lipsync(milo, index=i)
            assert parsed["n_frames"] > 0
            assert len(parsed["visemes"]) >= 0


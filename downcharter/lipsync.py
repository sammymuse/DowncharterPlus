"""lipsync.py — PHONEME-based lipsync for Rock Band 3 (from lyrics).

Replicates what Onyx's "Turn vocal tracks into LIPSYNC* tracks for RB3" button
does (text → phonemes → visemes → keyframes), but with the TIMING coming from the
lyric events instead of the charted vocal notes (tubes). This way it works on
songs that "only have lyrics" (no pitched gems in PART VOCALS).

Deliberate differences from Onyx (so as NOT to copy its code):
  * OWN grapheme→phoneme, rule-based (without the 3.5 MB cmudict). Each lyric
    event is already a syllable (the charter splits words with '-'/'='), so we run
    G2P on the syllable fragment directly — no dictionary lookup and no
    syllable-count matching.
  * The viseme map (vowels/consonants → facial morphs) is the only data table
    reused from Onyx (rb3.yml) — inlined; it's data, not logic.

Output: MIDI tracks `LIPSYNC1/2/3` (text-commands `[<viseme> <weight>]`) written
into the notes.mid by the `processor`. `lipsync_delta_events` gives the per-frame
deltas.

NOTE (in-game test): in the `.ini→RB3/PS3` conversion Onyx does NOT use these
tracks nor a `.milo_xbox` sidecar — it fabricates unpitched vocals from the lyrics
and generates the lipsync via `autoLipsync` from the LENGTH of the vocal tubes.
So the real lipsync now comes from CHARTING talky vocals (see
`processor._chart_vocals_from_lyrics`); this module stays as a reference
(format/keyframes) and for build paths that read LIPSYNC#.

`build_lipsync_from_lyrics` (+`_serialize`) still produces the raw CharLipSync
bytes (format validated byte-by-byte against official HMX .lipsync files),
big-endian:
    u32 version=1, u32 sub=2, empty DTA str, u8 dtb=0, u32 skip=0,
    u32 visemeCount=34, [u32 len+name]*34, u32 keyframeCount (30 fps),
    u32 followingSize, per frame{u8 eventCount; (u8 visemeIdx,u8 weight)*},
    u32 trailing=0.
The keyframes are DELTA (each frame only lists the visemes that CHANGED; the game
holds the previous value), just like the official files.
"""
from __future__ import annotations
import math
import struct

from .midi_utils import tick_to_ms

FPS = 30

# 34 canonical RB3 visemes (exact order from an official .lipsync).
VISEMES = [
    "Blink", "Brow_down", "Squint", "Brow_aggressive", "Size_hi", "Eat_hi",
    "Fave_lo", "Earth_lo", "If_lo", "Ox_hi", "Cage_hi", "Oat_lo", "Told_hi",
    "New_lo", "Fave_hi", "Though_lo", "Earth_hi", "If_hi", "Church_lo",
    "Roar_lo", "Bump_lo", "Oat_hi", "New_hi", "Wet_lo", "Though_hi", "Size_lo",
    "Eat_lo", "Church_hi", "Roar_hi", "Ox_lo", "Cage_lo", "Bump_hi", "Told_lo",
    "Wet_hi",
]
_VIDX = {n: i for i, n in enumerate(VISEMES)}

_W = 140  # "full" viseme weight (same as rb3.yml)


def _pair(base: str) -> dict[str, int]:
    """_hi+_lo viseme (e.g. 'Ox' → {Ox_hi:140, Ox_lo:140})."""
    return {f"{base}_hi": _W, f"{base}_lo": _W}


# ── VOWEL map: name → (main shape, final shape | None for diphthong) ──
# Reused from Onyx's rb3.yml (data table).
_VOWELS: dict[str, tuple[dict, dict | None]] = {
    "AA": (_pair("Ox"), None),
    "AH": (_pair("If"), None),
    "AY": (_pair("Ox"), _pair("If")),      # diphthong (bite)
    "EH": (_pair("Cage"), None),
    "ER": (_pair("Church"), None),
    "EY": (_pair("Cage"), _pair("If")),    # diphthong (bait)
    "IH": (_pair("If"), None),
    "IY": (_pair("Eat"), None),
    "OW": (_pair("Oat"), _pair("Wet")),    # diphthong (boat)
    "UW": (_pair("Wet"), None),
    "AE": (_pair("Cage"), None),
    "AO": (_pair("Earth"), None),
    "AW": (_pair("Ox"), _pair("Wet")),     # diphthong (bout)
    "OY": (_pair("Oat"), _pair("If")),     # diphthong (boy)
    "UH": (_pair("Though"), None),
}

# ── CONSONANT map: name → mouth shape ({} = invisible, e.g. G/H/K/NG) ──
_CONS: dict[str, dict] = {
    "B": _pair("Bump"), "CH": _pair("Told"), "D": _pair("Told"),
    "DH": _pair("Told"), "F": _pair("Fave"), "G": {}, "HH": {},
    "JH": _pair("Told"), "K": {}, "L": _pair("Told"), "M": _pair("Bump"),
    "N": _pair("Told"), "NG": {}, "P": _pair("Bump"), "R": _pair("Roar"),
    "S": _pair("Size"), "SH": _pair("Size"), "T": _pair("Told"),
    "TH": _pair("Told"), "V": _pair("Fave"), "W": _pair("Wet"),
    "Y": _pair("Eat"), "Z": _pair("Size"), "ZH": _pair("Size"),
}

_VOWEL_SET = set(_VOWELS)

# Mouth timing (seconds). Since we only have the syllable ONSET (not the note
# duration), the mouth opens and closes within a limited window.
_MAX_OPEN = 0.50
_MIN_OPEN = 0.09


# ───────────────────────── grapheme → phoneme (own G2P) ──────────────────────
# Digraphs are checked before single letters. Returns ARPABET-ish tokens.
_VOWEL_DIGRAPHS = {
    "ai": "EY", "ay": "EY", "au": "AO", "aw": "AO", "ea": "IY", "ee": "IY",
    "ei": "EY", "ey": "EY", "eu": "UW", "ew": "UW", "ie": "IY", "oa": "OW",
    "oo": "UW", "oi": "OY", "oy": "OY", "ou": "AW", "ow": "OW", "ue": "UW",
}
_CONS_DIGRAPHS = {
    "ch": "CH", "sh": "SH", "th": "TH", "ph": "F", "wh": "W", "ck": "K",
    "ng": "NG", "gh": None,  # gh: silent
}
_VOWEL_SINGLE = {"a": "AE", "e": "EH", "i": "IH", "o": "AA", "u": "AH"}
_CONS_SINGLE = {
    "b": "B", "c": "K", "d": "D", "f": "F", "g": "G", "h": "HH", "j": "JH",
    "k": "K", "l": "L", "m": "M", "n": "N", "p": "P", "q": "K", "r": "R",
    "s": "S", "t": "T", "v": "V", "w": "W", "z": "Z",
}


def grapheme_to_phonemes(frag: str) -> list[str]:
    """Convert a text fragment (syllable) into a list of phonemes.

    Rule-based approximation — good enough for mouth shapes (visemes), not meant
    to be an exact pronunciation. No dictionary."""
    s = "".join(c for c in frag.lower() if c.isalpha())
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        two = s[i:i + 2]
        c = s[i]
        # 'qu' → K W
        if two == "qu":
            out += ["K", "W"]
            i += 2
            continue
        if two in _CONS_DIGRAPHS:
            ph = _CONS_DIGRAPHS[two]
            if ph is not None:
                out.append(ph)
            i += 2
            continue
        if two in _VOWEL_DIGRAPHS:
            out.append(_VOWEL_DIGRAPHS[two])
            i += 2
            continue
        if c in _VOWEL_SINGLE:
            # silent final 'e': skip if a vowel already appeared and it's the last letter.
            if c == "e" and i == n - 1 and any(p in _VOWEL_SET for p in out):
                i += 1
                continue
            out.append(_VOWEL_SINGLE[c])
            i += 1
            continue
        if c == "y":
            # start + followed by a vowel → consonant; otherwise vowel.
            if i == 0 and i + 1 < n and (s[i + 1] in _VOWEL_SINGLE or
                                         s[i + 1] in "aeiou"):
                out.append("Y")
            else:
                out.append("IH")
            i += 1
            continue
        if c == "x":
            out += ["K", "S"]
            i += 1
            continue
        if c in _CONS_SINGLE:
            out.append(_CONS_SINGLE[c])
            i += 1
            continue
        i += 1  # ignore the rest
    return out


def _syllable_shape(text: str) -> tuple[list[dict], tuple[dict, dict | None], list[dict]]:
    """Lyric fragment → (initial consonants, (vowel, end|None), final consonants).

    Visemes already resolved (name→weight dicts). No vowel → AH fallback (neutral mouth)."""
    phones = grapheme_to_phonemes(text)
    # index of the 1st vowel
    vi = next((k for k, p in enumerate(phones) if p in _VOWEL_SET), None)
    if vi is None:
        return [], _VOWELS["AH"], []
    initial = [_CONS[p] for p in phones[:vi] if p in _CONS]
    vowel = _VOWELS[phones[vi]]
    # final consonants up to the next vowel (ignore extra vowels)
    final: list[dict] = []
    for p in phones[vi + 1:]:
        if p in _VOWEL_SET:
            break
        if p in _CONS:
            final.append(_CONS[p])
    return initial, vowel, final


# ───────────────────────── building the keyframes ────────────────────────────
def _syllable_points(t: float, dur: float, shape) -> list[tuple[float, dict]]:
    """Control points (time, visemes) of a syllable in the window [t, t+dur].

    Closed mouth {} before/after; brief consonants; vowel sustained in the middle;
    diphthong drops from the main shape to the final one. Linear interpolation
    between points creates the transitions (consonant↔vowel)."""
    initial, (vmain, vend), final = shape
    n_i, n_f = len(initial), len(final)
    attack = min(0.04, dur * 0.25)
    release = min(0.05, dur * 0.25)
    inner = max(1e-3, dur - attack - release)
    units = n_i + 3 + n_f          # the vowel is worth 3 units
    u = inner / units

    pts: list[tuple[float, dict]] = [(t, {})]
    cur = t + attack
    for c in initial:
        pts.append((cur, c))
        cur += u
    if vend is not None:           # diphthong: main → final
        pts.append((cur, vmain))
        cur += 1.5 * u
        pts.append((cur, vend))
        cur += 1.5 * u
    else:
        pts.append((cur, vmain))
        cur += 3 * u
    for c in final:
        pts.append((cur, c))
        cur += u
    pts.append((t + dur, {}))
    return pts


def _lerp_state(a: dict, b: dict, f: float) -> dict:
    """Interpolate two viseme states (weights) by fraction f∈[0,1]."""
    out = {}
    for name in set(a) | set(b):
        wa = a.get(name, 0)
        wb = b.get(name, 0)
        w = int(round(wa + (wb - wa) * f))
        if w > 0:
            out[name] = w
    return out


def _sample_into(frames: dict[int, dict], pts: list[tuple[float, dict]]) -> None:
    """Sample the points at 30 fps and merge into `frames` (max per viseme)."""
    if len(pts) < 2:
        return
    f0 = int(math.floor(pts[0][0] * FPS))
    f1 = int(math.ceil(pts[-1][0] * FPS))
    seg = 0
    for fr in range(max(0, f0), f1 + 1):
        tf = fr / FPS
        if tf < pts[0][0] or tf > pts[-1][0]:
            continue
        while seg + 1 < len(pts) - 1 and tf > pts[seg + 1][0]:
            seg += 1
        ta, sa = pts[seg]
        tb, sb = pts[seg + 1]
        frac = 0.0 if tb <= ta else (tf - ta) / (tb - ta)
        frac = max(0.0, min(1.0, frac))
        state = _lerp_state(sa, sb, frac)
        if not state:
            continue
        slot = frames.setdefault(fr, {})
        for name, w in state.items():
            if w > slot.get(name, 0):
                slot[name] = w


def _build_frames(lyrics: list[tuple[float, str]],
                  phrase_ends: list[float] | None = None) -> dict[int, dict]:
    """Lyrics → per-frame viseme states (name→weight, merged by max).

    The mouth extends to the NEXT syllable (sustained vowel, "connected"
    articulation); on the last syllable of each phrase it holds until `phrase_end`
    and only then closes. Without phrase markers, it falls back to the old behavior
    (_MAX_OPEN ceiling)."""
    lyr = sorted((t, txt) for t, txt in lyrics if t >= 0)
    pe = sorted(p for p in (phrase_ends or []) if p >= 0)
    n = len(lyr)
    frames: dict[int, dict] = {}
    for i, (t, txt) in enumerate(lyr):
        nxt = lyr[i + 1][0] if i + 1 < n else None
        # first phrase_end strictly after this syllable
        pend = next((p for p in pe if p > t + 1e-3), None)
        if nxt is not None and (pend is None or nxt <= pend + 1e-3):
            target = nxt                    # next syllable in the same phrase
        elif pend is not None:
            target = pend                   # last in the phrase → hold until phrase_end
        else:
            target = t + _MAX_OPEN          # no end info → fallback
        dur = max(_MIN_OPEN, target - t)
        if pend is None:                    # no phrase markers: don't sustain too long
            dur = min(dur, _MAX_OPEN)
        _sample_into(frames, _syllable_points(t, dur, _syllable_shape(txt)))
    return frames


def _delta_frames(frames: dict[int, dict], n_frames: int):
    """Iterate (frame, [(name, weight)]) with only the visemes that CHANGED in that frame."""
    prev: dict[str, int] = {}
    for fr in range(n_frames):
        cur = {n: (w & 0xFF) for n, w in frames.get(fr, {}).items()
               if n in _VIDX and w > 0}
        deltas: list[tuple[str, int]] = []
        for name, w in cur.items():
            if prev.get(name, 0) != w:
                deltas.append((name, w))
        for name in prev:
            if name not in cur:
                deltas.append((name, 0))
        if deltas:
            deltas.sort()
            yield fr, deltas
        prev = cur


def build_lipsync_from_lyrics(lyrics: list[tuple[float, str]], song_len_s: float) -> bytes:
    """Build the CharLipSync bytes from (time_seconds, text) of the lyrics."""
    frames = _build_frames(lyrics)
    n_frames = max(1, int(math.ceil(song_len_s * FPS)) + 1)
    return _serialize(frames, n_frames)


def lipsync_delta_events(lyrics: list[tuple[float, str]], song_len_s: float,
                         phrase_ends: list[float] | None = None):
    """(frame_idx, [(viseme, weight)]) to write an Onyx LIPSYNC# MIDI track.

    Onyx builds the milo at build time from these tracks (text-commands
    `[<viseme> <weight>]`); it's the path the `.ini` import consumes.
    `phrase_ends` (seconds of the 105/106 note_offs) makes the last syllable of
    each phrase hold until the end of the phrase."""
    frames = _build_frames(lyrics, phrase_ends)
    n_frames = max(1, int(math.ceil(song_len_s * FPS)) + 1)
    return list(_delta_frames(frames, n_frames))


def _serialize(frames: dict[int, dict], n_frames: int) -> bytes:
    """Per-frame states (name→weight) → delta-encoded CharLipSync, 34 visemes."""
    body = bytearray()
    prev: dict[int, int] = {}
    for fr in range(n_frames):
        cur = {}
        for name, w in frames.get(fr, {}).items():
            idx = _VIDX.get(name)
            if idx is not None and w > 0:
                cur[idx] = w & 0xFF
        events: list[tuple[int, int]] = []
        for idx, w in cur.items():
            if prev.get(idx, 0) != w:
                events.append((idx, w))
        for idx in prev:
            if idx not in cur:
                events.append((idx, 0))
        events.sort()
        body.append(len(events) & 0xFF)
        for idx, w in events:
            body += struct.pack("BB", idx & 0xFF, w & 0xFF)
        prev = cur

    out = bytearray()
    out += _be32(1)                 # version (RB3/Magma v2)
    out += _be32(2)                 # subversion
    out += _bestr("")               # DTA import (empty)
    out += struct.pack("B", 0)      # dtb flag
    out += _be32(0)                 # skip
    out += _be32(len(VISEMES))      # viseme count
    for name in VISEMES:
        out += _bestr(name)
    out += _be32(n_frames)          # keyframe count
    out += _be32(len(body))         # following size
    out += body
    out += _be32(0)                 # trailing
    return bytes(out)


def _be32(v: int) -> bytes:
    return struct.pack(">I", v)


def _bestr(s: str) -> bytes:
    b = s.encode("ascii")
    return _be32(len(b)) + b


# ───────────────────────── extracting lyrics from the MIDI ───────────────────
def syllable_pairs(track, tempo_map, tpb: int) -> list[tuple[float, str]]:
    """(seconds, text) of each syllable from a vocal/harmony track.

    `lyrics`/`lyric` events (or `text` without a [bracket]); ignores
    sustain/no-sound markers (+ # ^ * %)."""
    out: list[tuple[float, str]] = []
    t = 0
    for m in track:
        t += m.time
        txt = getattr(m, "text", "")
        if m.type in ("lyrics", "lyric") or (
                m.type == "text" and txt and not txt.startswith("[")):
            if txt.strip() in ("+", "#", "^", "*", "%"):
                continue
            sec = tick_to_ms(t, tempo_map, tpb) / 1000.0
            out.append((sec, txt))
    return out


def phrase_ends(track, tempo_map, tpb: int) -> list[float]:
    """Seconds of the vocal phrase ends (note_off of the 105/106 phrase markers).

    In RB3 each vocal phrase is delimited by a note at pitch 105 (or 106); its
    note_off marks the end of the phrase. Used to hold the last syllable."""
    t = 0
    open_at: dict[int, int] = {}
    ends: list[float] = []
    for m in track:
        t += m.time
        note = getattr(m, "note", None)
        if note not in (105, 106):
            continue
        if m.type == "note_on" and m.velocity > 0:
            open_at[note] = t
        elif m.type == "note_off" or (m.type == "note_on" and m.velocity == 0):
            if note in open_at:
                ends.append(tick_to_ms(t, tempo_map, tpb) / 1000.0)
                del open_at[note]
    return sorted(set(ends))


def syllable_seconds(track, tempo_map, tpb: int) -> list[float]:
    """Compat: just the syllable instants (seconds)."""
    return sorted({s for s, _ in syllable_pairs(track, tempo_map, tpb)})

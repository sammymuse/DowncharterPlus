"""lipsync.py — PHONEME-based lipsync for Rock Band 3 (from lyrics).

Replicates what Onyx's "Turn vocal tracks into LIPSYNC* tracks for RB3" button
does (text → phonemes → visemes → keyframes), but with the TIMING coming from the
lyric events instead of the charted vocal notes (tubes). This way it works on
songs that "only have lyrics" (no pitched gems in PART VOCALS).

Deliberate differences from Onyx (so as NOT to copy its code):
  * grapheme→phoneme: CMUdict (English, public domain) for COMPLETE words +
    OWN rule-based G2P as fallback for hyphenated fragments, out-of-vocabulary
    words and non-English (German/Spanish spelling is phonetic, so the rules
    suffice and no dict is used). Each lyric event is already a syllable (the
    charter splits words with '-'/'='); whole-word fragments hit the dict, the
    rest run G2P directly — no syllable-count matching.
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
import gzip
import math
import os
import struct
import sys

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


# ───────────────────────── CMUdict (English only) ───────────────────────────
# Whole-word ARPABET lookup. English spelling is irregular, so for COMPLETE
# words the dictionary beats the rules. Multi-syllable HYPHENATED fragments
# (e.g. "el-e-gy") are not whole words → they fall back to the rules. German /
# Spanish have phonetic spelling, so the rules already suffice there (no dict).
_CMUDICT: dict[str, list[str]] | None = None


def _data_path(name: str) -> str:
    """Bundled data file, working both in dev and in a PyInstaller onedir."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return os.path.join(base, "downcharter", "data", name)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name)


def _cmudict() -> dict[str, list[str]]:
    """Lazy-loaded English pronunciation dictionary (word → ARPABET phones)."""
    global _CMUDICT
    if _CMUDICT is None:
        d: dict[str, list[str]] = {}
        try:
            with gzip.open(_data_path("cmudict.en.txt.gz"), "rt",
                           encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        d[parts[0]] = parts[1:]
        except Exception:
            d = {}  # missing/corrupt dict → rules-only, never crash
        _CMUDICT = d
    return _CMUDICT


def grapheme_to_phonemes(frag: str, lang: str = "en") -> list[str]:
    """Convert a text fragment (syllable) into a list of phonemes.

    For English COMPLETE words, looks them up in CMUdict (accurate). Otherwise
    (hyphenated fragments, non-English, or out-of-vocabulary) falls back to the
    rule-based approximation — good enough for mouth shapes (visemes)."""
    s = "".join(c for c in frag.lower() if c.isalpha())
    if lang == "en" and s:
        hit = _cmudict().get(s)
        if hit:
            return list(hit)
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


def _shape_from_phones(phones: list[str]) -> tuple[list[dict], tuple[dict, dict | None], list[dict]]:
    """Phoneme list → (initial consonants, (vowel, end|None), final consonants).

    Visemes already resolved (name→weight dicts). No vowel → AH fallback (neutral mouth)."""
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


def _syllable_shape(text: str, lang: str = "en") -> tuple[list[dict], tuple[dict, dict | None], list[dict]]:
    """Lyric fragment → mouth shape, via G2P on the fragment alone (no word context)."""
    return _shape_from_phones(grapheme_to_phonemes(text, lang))


def _strip_markers(text: str) -> str:
    """Lyric text without trailing RB markers (#/^/+/*/% and surrounding space)."""
    return text.rstrip().rstrip("#^+*%").rstrip()


def _word_continues(text: str) -> bool:
    """True if this syllable joins the next one (trailing '-'/'=' = same word)."""
    return _strip_markers(text).endswith(("-", "="))


def align_word_phonemes(syllables: list[str], lang: str = "en") -> list[list[str]] | None:
    """Distribute a word's CMUdict phonemes across its WRITTEN syllables.

    `syllables` = the consecutive lyric fragments of ONE word (e.g. ['el-','e-','gy']).
    Looks the whole word up in the dictionary, then splits its phoneme sequence so each
    written syllable gets exactly one vowel nucleus (consonant clusters between two
    nuclei are split in the middle). Returns one phoneme list per syllable, or None when
    no reliable alignment exists (non-English, OOV, or #nuclei != #syllables) → caller
    falls back to per-fragment G2P. This fixes hyphenated multi-syllable words, whose
    fragments aren't whole words and so missed the dictionary before."""
    if lang != "en":
        return None
    clean = ["".join(c for c in s.lower() if c.isalpha()) for s in syllables]
    word = "".join(clean)
    if not word:
        return None
    phones = _cmudict().get(word)
    if not phones:
        return None
    nuclei = [k for k, p in enumerate(phones) if p in _VOWEL_SET]
    if len(nuclei) != len(syllables):
        return None  # our written split disagrees with the dict → don't force it
    out: list[list[str]] = []
    prev = 0
    for k, nuc in enumerate(nuclei):
        if k + 1 < len(nuclei):
            boundary = (nuc + nuclei[k + 1]) // 2 + 1  # split medial cluster ~evenly
            out.append(phones[prev:boundary])
            prev = boundary
        else:
            out.append(phones[prev:])
    return out


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


def lipsync_events_from_spans(spans, song_len_s: float, lang: str = "en"):
    """(frame_idx, [(viseme, weight)]) for a LIPSYNC# MIDI track, driven by the REAL
    syllable spans (audio-guided start/end) instead of a geometric onset window.

    `spans` = list of (start_s, end_s, text[, gain]) where gain ∈ (0, 1] scales the
    viseme weight by the syllable's loudness (1.0 = full). The mouth opens at
    start_s, sustains the vowel, and closes by end_s — the true note length the audio
    confirmed — so the lipsync matches when the singer actually stops. This is what
    sets us apart from Onyx/YARG's built-in generators (both geometric, audio-blind).

    Consecutive spans of the SAME word (trailing '-'/'=') are grouped so the whole word
    is looked up in CMUdict and its phonemes aligned per written syllable
    (`align_word_phonemes`); on a miss each fragment falls back to per-fragment G2P.

    NOTE: this is the dense 30fps-delta path, kept as reference/fallback. The production
    path is `lipsync_keyframes_from_spans` (sparse keyframes with hold/ease graphs)."""
    frames, n_frames = frames_from_spans(spans, song_len_s, lang)
    return list(_delta_frames(frames, n_frames))


def frames_from_spans(spans, song_len_s: float,
                      lang: str = "en") -> tuple[dict[int, dict], int]:
    """Dense per-frame viseme states (name→weight, 30 fps) from audio-guided spans.

    The shared core of the dense path: resolves one mouth shape per span (grouping
    same-word syllables for CMUdict alignment), samples each syllable's control points
    at 30 fps and merges them (max per viseme), scaling by the per-syllable loudness
    `gain`. Returns `(frames, n_frames)`. `lipsync_events_from_spans` delta-encodes it
    for the MIDI track; `milo.build_song_lipsync` serializes it into the CharLipSync
    that goes inside the .milo (guaranteeing the same lipsync reaches the game)."""
    spans = list(spans)
    shapes = _resolve_shapes(spans, lang)
    frames: dict[int, dict] = {}
    for sp, shape in zip(spans, shapes):
        t, end = sp[0], sp[1]
        gain = sp[3] if len(sp) > 3 else 1.0
        dur = end - t
        if dur <= 0:
            continue
        pts = _syllable_points(t, dur, shape)
        if gain != 1.0:
            pts = [(pt_t, {nm: max(0, min(255, int(round(w * gain))))
                           for nm, w in st.items()})
                   for pt_t, st in pts]
        _sample_into(frames, pts)
    n_frames = max(1, int(math.ceil(song_len_s * FPS)) + 1)
    return frames, n_frames


def _resolve_shapes(spans, lang: str = "en") -> list:
    """One mouth shape per span, grouping consecutive same-word syllables (trailing
    '-'/'=') so the whole word is looked up in CMUdict and aligned per syllable."""
    n = len(spans)
    shapes: list = [None] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and _word_continues(spans[j][2]):
            j += 1
        group = spans[i:j + 1]
        aligned = align_word_phonemes([sp[2] for sp in group], lang)
        for k in range(len(group)):
            shapes[i + k] = (_shape_from_phones(aligned[k]) if aligned is not None
                             else _syllable_shape(group[k][2], lang))
        i = j + 1
    return shapes


# ─────────────────── sparse keyframes with hold/ease graphs ───────────────────
# Onyx graph semantics (Lipsync.hs): the token is the curve OUT of this keyframe to
# the viseme's NEXT event — `hold` keeps the weight flat, `linear` ramps, `ease`
# ramps exponentially (diphthong glide). Default (no token) = linear.

def _syllable_points_g(t: float, dur: float, shape) -> list[tuple[float, dict, str]]:
    """Like `_syllable_points` but each point carries the graph of the segment that
    STARTS at it: a held vowel plateau (`hold`) and the diphthong glide (`ease`)."""
    initial, (vmain, vend), final = shape
    n_i, n_f = len(initial), len(final)
    attack = min(0.04, dur * 0.25)
    release = min(0.05, dur * 0.25)
    inner = max(1e-3, dur - attack - release)
    u = inner / (n_i + 3 + n_f)        # the vowel is worth 3 units
    pts: list[tuple[float, dict, str]] = [(t, {}, "linear")]   # ramp up from closed
    cur = t + attack
    for c in initial:
        pts.append((cur, c, "linear"))
        cur += u
    if vend is not None:               # diphthong: glide main → final (ease)
        pts.append((cur, vmain, "ease"))
        cur += 1.5 * u
        pts.append((cur, vend, "linear"))
        cur += 1.5 * u
    else:                              # monophthong: reach, HOLD the plateau, release
        pts.append((cur, vmain, "hold"))
        cur += 3 * u
        pts.append((cur, vmain, "linear"))
    for c in final:
        pts.append((cur, c, "linear"))
        cur += u
    pts.append((t + dur, {}, "linear"))   # close
    return pts


def _keyframes_from_points(pts, gain: float):
    """Per-viseme keyframes (time, viseme, weight, graph) from graphed control points.

    A viseme's keyframe takes the point's graph only when it's active there (weight>0);
    a viseme sitting at 0 always uses linear (so it ramps in/out, never 'holds' 0).
    Redundant collinear keyframes are dropped (linear interpolation reconstructs them)
    and leading/trailing zero runs trimmed to one bracketing zero (the ramp endpoints)."""
    names: set[str] = set()
    for _t, st, _g in pts:
        names |= set(st)
    out: list[tuple[float, str, int, str]] = []
    for nm in names:
        ser: list[list] = []
        for time, st, pg in pts:
            w = st.get(nm, 0)
            if w and gain != 1.0:
                w = max(0, min(255, int(round(w * gain))))
            ser.append([time, int(w), (pg if w > 0 else "linear")])
        ser = _simplify_series(ser)
        for time, w, g in ser:
            out.append((time, nm, w, g))
    return out


def _simplify_series(ser: list[list]) -> list[list]:
    """Drop collinear linear keyframes and trim zero runs to one bracketing zero."""
    if len(ser) > 2:
        keep = [ser[0]]
        for i in range(1, len(ser) - 1):
            t0, w0, g0 = keep[-1]
            t1, w1, g1 = ser[i]
            t2, w2, _g2 = ser[i + 1]
            if g0 == "linear" and g1 == "linear" and t2 > t0:
                pred = w0 + (w2 - w0) * ((t1 - t0) / (t2 - t0))
                if abs(pred - w1) <= 0.5:      # collinear → linear interp rebuilds it
                    continue
            keep.append(ser[i])
        keep.append(ser[-1])
        ser = keep
    nz = [k for k, (_t, w, _g) in enumerate(ser) if w > 0]
    if not nz:
        return []
    return ser[max(0, nz[0] - 1):nz[-1] + 2]   # keep one zero on each side


# ─────────────────── facial animation keyframes ───────────────────
# Generated alongside mouth visemes in the same LIPSYNC1 track.
# Official RB3 milos carry Blink, Brow_*, Squint mixed with mouth shapes.

_FACIAL_VISEMES = ("Blink", "Brow_aggressive", "Brow_down",
                   "Brow_pouty", "Brow_up", "Squint")

# Timing constants (seconds)
_BLINK_INTERVAL = (2.0, 10.0)     # random uniform range
_BLINK_CLOSE_S = 0.06             # close duration
_BLINK_HOLD_S = 0.05              # hold closed
_BLINK_OPEN_S = 0.08              # open duration
_BLINK_WEIGHT = 140

_SQUINT_INTERVAL = (8.0, 15.0)
_SQUINT_DURATION = 0.25
_SQUINT_WEIGHT = 100

# Eyebrow pitch thresholds (MIDI note numbers).
# C5 = 72, G3 = 55, perfect fifth = 7 semitones.
_BROW_AGGRESSIVE_PITCH = 72       # notes >= C5 → Brow_aggressive
_BROW_DOWN_PITCH = 55             # notes <= G3 sustained → Brow_down
_BROW_UP_JUMP = 7                 # consecutive note jump >= 7 → Brow_up
_BROW_AGGRESSIVE_WEIGHT = 120
_BROW_DOWN_WEIGHT = 110
_BROW_UP_WEIGHT = 100
_BROW_HOLD_S = 0.15               # hold the expression briefly
_BROW_FADE_S = 0.30               # fade out duration


def _generate_blinks(
    song_len_s: float,
    phrase_ends: list[float] | None = None,
    rng=None,
) -> list[tuple[float, str, int, str]]:
    """Periodic eye-blink keyframes with extra blinks at phrase boundaries.

    Returns ``(time_s, "Blink", weight, graph)`` sparse keyframes that
    interleave with mouth visemes.  Blinks are quick: close → hold → open
    in ~0.19 s."""
    if rng is None:
        import random as rng
    out: list[tuple[float, str, int, str]] = []
    t = rng.uniform(*_BLINK_INTERVAL)
    while t < song_len_s:
        out.append((t, "Blink", _BLINK_WEIGHT, "ease"))       # close
        out.append((t + _BLINK_CLOSE_S, "Blink", _BLINK_WEIGHT, "hold"))  # hold
        out.append((t + _BLINK_CLOSE_S + _BLINK_HOLD_S, "Blink", 0, "linear"))  # open
        t += rng.uniform(*_BLINK_INTERVAL)

    # Phrase-boundary bonus blinks: 40 % chance of an extra blink just
    # after each phrase end (but not within 0.5 s of an existing blink).
    if phrase_ends:
        blink_times = {round(e[0], 2) for e in out}
        for pe in phrase_ends:
            if pe < 0 or pe >= song_len_s:
                continue
            if any(abs(pe - bt) < 0.5 for bt in blink_times):
                continue
            if rng.random() < 0.4:
                out.append((pe, "Blink", _BLINK_WEIGHT, "ease"))
                out.append((pe + _BLINK_CLOSE_S, "Blink", _BLINK_WEIGHT, "hold"))
                out.append((pe + _BLINK_CLOSE_S + _BLINK_HOLD_S, "Blink", 0, "linear"))
                blink_times.update({pe, pe + _BLINK_CLOSE_S,
                                    pe + _BLINK_CLOSE_S + _BLINK_HOLD_S})

    out.sort(key=lambda e: (e[0], e[1]))
    return out


def _generate_squints(
    song_len_s: float,
    rng=None,
) -> list[tuple[float, str, int, str]]:
    """Periodic squint keyframes."""
    if rng is None:
        import random as rng
    out: list[tuple[float, str, int, str]] = []
    t = rng.uniform(*_SQUINT_INTERVAL)
    while t < song_len_s:
        out.append((t, "Squint", _SQUINT_WEIGHT, "ease"))
        out.append((t + _SQUINT_DURATION, "Squint", 0, "linear"))
        t += rng.uniform(*_SQUINT_INTERVAL)
    return out


def _generate_eyebrows(
    vocal_notes: list[tuple[float, int]],
    rng=None,
) -> list[tuple[float, str, int, str]]:
    """Eyebrow animation keyframes driven by vocal pitch.

    ``vocal_notes`` is a sorted list of ``(start_s, midi_pitch)`` from the
    PART VOCALS track.  Rules:

    * **Brow_aggressive** — onsets whose pitch >= ``_BROW_AGGRESSIVE_PITCH``
      (C5) raise the inner brows for the note's intensity.  Held for
      ``_BROW_HOLD_S`` then fade out over ``_BROW_FADE_S``.
    * **Brow_down** — sustained notes (pitch <= ``_BROW_DOWN_PITCH`` = G3)
      furrow the brows.  Only triggers when the note extends beyond
      ``_BROW_HOLD_S + _BROW_FADE_S`` so a short low note doesn't scowl.
    * **Brow_up** — a pitch jump >= ``_BROW_UP_JUMP`` semitones between
      consecutive vocal onsets raises the brows in surprise at the higher
      note.  A brief flash (hold + fade = 0.45 s).
    """
    if rng is None:
        import random as rng
    out: list[tuple[float, str, int, str]] = []

    # Group consecutive notes into phrases (gap > 0.5 s = phrase boundary).
    if not vocal_notes:
        return out
    phrases: list[list[tuple[float, int]]] = [[vocal_notes[0]]]
    for i in range(1, len(vocal_notes)):
        if vocal_notes[i][0] - vocal_notes[i - 1][0] > 0.5:
            phrases.append([])
        phrases[-1].append(vocal_notes[i])

    for phrase in phrases:
        if len(phrase) < 1:
            continue
        pitches = [p for _, p in phrase]
        t0 = phrase[0][0]
        t1 = phrase[-1][0] + _BROW_HOLD_S + _BROW_FADE_S
        max_p = max(pitches)
        min_p = min(pitches)
        dur = phrase[-1][0] - phrase[0][0]

        # Brow_aggressive: any note >= C5 in this phrase.
        if max_p >= _BROW_AGGRESSIVE_PITCH:
            out.append((t0, "Brow_aggressive", _BROW_AGGRESSIVE_WEIGHT, "ease"))
            out.append((t0 + _BROW_HOLD_S, "Brow_aggressive",
                        _BROW_AGGRESSIVE_WEIGHT, "hold"))
            out.append((t1, "Brow_aggressive", 0, "linear"))

        # Brow_down: average pitch low AND sustained long enough.
        if dur > _BROW_HOLD_S + _BROW_FADE_S and sum(pitches) / len(pitches) < _BROW_DOWN_PITCH:
            out.append((t0, "Brow_down", _BROW_DOWN_WEIGHT, "ease"))
            out.append((t0 + _BROW_HOLD_S, "Brow_down", _BROW_DOWN_WEIGHT, "hold"))
            out.append((t1, "Brow_down", 0, "linear"))

    # Brow_up: pitch jump between consecutive notes anywhere in the track.
    for i in range(1, len(vocal_notes)):
        prev_t, prev_p = vocal_notes[i - 1]
        cur_t, cur_p = vocal_notes[i]
        if cur_p - prev_p >= _BROW_UP_JUMP:
            jt = cur_t  # flash at the higher note's onset
            out.append((jt, "Brow_up", _BROW_UP_WEIGHT, "ease"))
            out.append((jt + _BROW_HOLD_S, "Brow_up", _BROW_UP_WEIGHT, "hold"))
            out.append((jt + _BROW_HOLD_S + _BROW_FADE_S, "Brow_up", 0, "linear"))

    out.sort(key=lambda e: (e[0], e[1]))
    return out


def generate_facial_keyframes(
    song_len_s: float,
    phrase_ends: list[float] | None = None,
    rng_seed: int | None = None,
    vocal_notes: list[tuple[float, int]] | None = None,
) -> list[tuple[float, str, int, str]]:
    """All facial animation keyframes (Blink, Squint, Brow_*) for the whole song.

    Returns sparse keyframes in the same ``(time_s, viseme, weight, graph)``
    format as :func:`lipsync_keyframes_from_spans`, ready to be merged.

    When ``vocal_notes`` (list of ``(start_s, midi_pitch)``) is provided,
    eyebrow expressions are generated based on pitch: high notes raise inner
    brows (Brow_aggressive), low sustained notes furrow (Brow_down), and
    large pitch jumps cause surprise (Brow_up)."""
    import random as _rng
    rng = _rng.Random(rng_seed) if rng_seed is not None else _rng
    out: list[tuple[float, str, int, str]] = []
    out.extend(_generate_blinks(song_len_s, phrase_ends, rng))
    out.extend(_generate_squints(song_len_s, rng))
    if vocal_notes:
        out.extend(_generate_eyebrows(vocal_notes, rng))
    out.sort(key=lambda e: (e[0], e[1]))
    return out


def merge_facial_into_keyframes(
    mouth_kf: list[tuple[float, str, int, str]],
    facial_kf: list[tuple[float, str, int, str]],
) -> list[tuple[float, str, int, str]]:
    """Merge mouth and facial keyframes, sorted by (time_s, viseme).

    Mouth and facial visemes are disjoint sets — no duplicate-viseme risk.
    The sort order matches ``lipsync_keyframes_from_spans`` so the caller
    gets a single flat list ready for ``_build_lipsync_track``."""
    out = mouth_kf + facial_kf
    out.sort(key=lambda e: (e[0], e[1]))
    return out


def lipsync_keyframes_from_spans(
    spans,
    lang: str = "en",
    phrase_ends: list[float] | None = None,
    song_len_s: float | None = None,
    facial_seed: int | None = None,
    vocal_notes: list[tuple[float, int]] | None = None,
) -> list[tuple[float, str, int, str]]:
    """(time_s, viseme, weight, graph) sparse keyframes for a LIPSYNC# MIDI track.

    Production path. Same audio-guided spans as `lipsync_events_from_spans`, but emits
    sparse keyframes with Onyx graph tokens (held vowels = `hold`, diphthong glides =
    `ease`, transitions = linear) instead of baking the curve into dense 30fps deltas.
    Far fewer events and closer to how Onyx itself authors the milo. Syllable windows
    are disjoint (audio gap between gems) so each closes the mouth before the next.

    When ``song_len_s`` is provided, facial animation keyframes are generated alongside
    the mouth shapes and merged into the output.  ``facial_seed`` makes random blink/
    squint timing reproducible.  ``vocal_notes`` (list of ``(start_s, midi_pitch)``)
    drives eyebrow expressions via :func:`generate_facial_keyframes`."""
    spans = list(spans)
    shapes = _resolve_shapes(spans, lang)
    out: list[tuple[float, str, int, str]] = []
    for sp, shape in zip(spans, shapes):
        t, end = sp[0], sp[1]
        gain = sp[3] if len(sp) > 3 else 1.0
        if end - t <= 0:
            continue
        out.extend(_keyframes_from_points(_syllable_points_g(t, end - t, shape), gain))
    if song_len_s is not None and song_len_s > 0:
        facial = generate_facial_keyframes(
            song_len_s, phrase_ends, facial_seed, vocal_notes)
        out = merge_facial_into_keyframes(out, facial)
    else:
        out.sort(key=lambda e: (e[0], e[1]))
    return out


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

"""lipsync.py — grapheme → phoneme → viseme shape resolution (Rock Band 3).

The good half. Turns lyric text into a per-syllable mouth "shape":

    (initial_consonants, (main_vowel, diphthong_end | None), final_consonants)

where every viseme is already resolved to a name→weight dict using the
official RB3 blend weights below.

Pipeline:
  1. ``grapheme_to_phonemes(text)`` — rule-based G2P + CMUdict (English).
  2. ``align_word_phonemes(syllables)`` — split a whole word's dictionary
     phonemes across its written syllables (fixes hyphenated multi-syllable
     words like ``el-e-gy``).
  3. ``_shape_from_phones(phones)`` — phoneme list → (initial, vowel, final)
     using the ``_VOWELS`` / ``_CONS`` viseme maps.
  4. ``_resolve_shapes(spans)`` — spans ``[(start, end, text, ...)]`` → one
     shape per span, grouping consecutive same-word syllables (trailing
     ``-``/``=``) for CMUdict alignment.

This module deliberately contains NO timing, keyframe or serialization logic.
That layer is being rebuilt — the old (buggy) version is kept in
``lipsync.py.bak`` for reference.
"""
from __future__ import annotations

import gzip
import math
import os
import sys


# ───────────────────────── canonical RB3 visemes ──────────────────────────
# Reference list of the 34 static .lipsync visemes + extras found in official
# milos. Not used for indexing here (shapes are resolved by name), kept as
# the authoritative name set.
VISEMES = [
    "Blink", "Brow_down", "Squint", "Brow_aggressive", "Size_hi", "Eat_hi",
    "Fave_lo", "Earth_lo", "If_lo", "Ox_hi", "Cage_hi", "Oat_lo", "Told_hi",
    "New_lo", "Fave_hi", "Though_lo", "Earth_hi", "If_hi", "Church_lo",
    "Roar_lo", "Bump_lo", "Oat_hi", "New_hi", "Wet_lo", "Though_hi", "Size_lo",
    "Eat_lo", "Church_hi", "Roar_hi", "Ox_lo", "Cage_lo", "Bump_hi", "Told_lo",
    "Wet_hi",
    # Extras found in official milos (not in the 34-static list, but present
    # in some songs' dynamic viseme tables):
    "Brow_openmouthed",
    "exp_banger_roar_01",
]


# Mouth opening ceiling. The official milos peak at ~153 (_lo) with a 2:3 _hi:_lo
# ratio (measured across 83 songs); Onyx ships a flat 140. This is the single
# "how wide should the mouth open" knob — every mouth viseme and blend is scaled
# from it (blends SPREAD this budget across their bases, they don't add to it).
_MOUTH_LO = 170
_MOUTH_HI = round(_MOUTH_LO * 2 / 3)      # 113 — official _hi:_lo ratio
_OAT_LO = round(_MOUTH_LO * 118 / 140)    # 143 — Oat is ~84% of the standard
_OAT_HI = round(_MOUTH_HI * 79 / 93)      # 96

_VISEME_WEIGHTS = {
    # Mouth visemes (pairs) — official MAX values, scaled from _MOUTH_LO
    "Bump_hi": _MOUTH_HI, "Bump_lo": _MOUTH_LO,
    "Cage_hi": _MOUTH_HI, "Cage_lo": _MOUTH_LO,
    "Church_hi": _MOUTH_HI, "Church_lo": _MOUTH_LO,
    "Earth_hi": _MOUTH_HI, "Earth_lo": _MOUTH_LO,
    "Eat_hi": _MOUTH_HI, "Eat_lo": _MOUTH_LO,
    "Fave_hi": _MOUTH_HI, "Fave_lo": _MOUTH_LO,
    "If_hi": _MOUTH_HI, "If_lo": _MOUTH_LO,
    "New_hi": _MOUTH_HI, "New_lo": _MOUTH_LO,
    "Oat_hi": _OAT_HI, "Oat_lo": _OAT_LO,
    "Ox_hi": _MOUTH_HI, "Ox_lo": _MOUTH_LO,
    "Roar_hi": _MOUTH_HI, "Roar_lo": _MOUTH_LO,
    "Size_hi": _MOUTH_HI, "Size_lo": _MOUTH_LO,
    "Though_hi": _MOUTH_HI, "Though_lo": _MOUTH_LO,
    "Told_hi": _MOUTH_HI, "Told_lo": _MOUTH_LO,
    "Wet_hi": _MOUTH_HI, "Wet_lo": _MOUTH_LO,
}

# Legacy constant for backward compatibility
_W = 140


def _pair(base: str) -> dict[str, int]:
    """_hi+_lo viseme with official weights (e.g. 'Ox' → {Ox_hi:102, Ox_lo:153})."""
    return {
        f"{base}_hi": _VISEME_WEIGHTS.get(f"{base}_hi", _W),
        f"{base}_lo": _VISEME_WEIGHTS.get(f"{base}_lo", _W),
    }


def _blend(*bases: str, weights: list[float] | None = None) -> dict[str, int]:
    """Blend multiple viseme pairs into one shape, spreading a fixed opening budget.

    Officials use several visemes simultaneously (blending). The blend must keep
    the TOTAL mouth opening equal to a single viseme (the ``_MOUTH_LO`` ceiling),
    so the relative weights are normalised to sum to 1.0 — the extra bases change
    the SHAPE (mix of morphs), not how WIDE the mouth opens.

    Args:
        *bases: viseme base names (e.g. 'Earth', 'Eat', 'If')
        weights: relative weights for each base (default: 1.0, 0.7, 0.5, ...).
                 Normalised to sum to 1.0 before applying.

    Returns:
        dict of viseme_name → weight (the _lo weights sum to ``_MOUTH_LO``,
        _hi to ``_MOUTH_HI``).
    """
    if weights is None:
        # Default relative weights: first base dominates, rest fall off.
        weights = [1.0, 0.7, 0.5, 0.35, 0.25][:len(bases)]

    total = sum(weights)
    norm = [w / total for w in weights]

    result = {}
    for base, w in zip(bases, norm):
        pair = _pair(base)
        for viseme, weight in pair.items():
            result[viseme] = int(round(weight * w))
    return result


# ── VOWEL map: name → (main shape, final shape | None for diphthong) ──
# Officials use BLENDS of multiple visemes (4-6 simultaneously).
# Based on analysis of official .milo_xbox files.
_VOWELS: dict[str, tuple[dict, dict | None]] = {
    # Vowels with blends based on official milo analysis
    "AA": (_blend("Ox", "Eat"), None),                    # "father" - Ox dominant + Eat blend
    "AH": (_blend("If", "Eat"), None),                    # "but" - If dominant + Eat blend
    "AY": (_blend("Ox", "Eat"), _blend("If", "Eat")),    # diphthong (bite)
    "EH": (_blend("Cage", "Eat", "If"), None),            # "bed" - Cage + Eat + If (308x in officials)
    "ER": (_blend("Church", "If"), None),                 # "bird" - Church + If blend
    "EY": (_blend("Cage", "Eat"), _blend("If", "Eat")),  # diphthong (bait)
    "IH": (_blend("If", "Eat"), None),                    # "bit" - If + Eat blend
    "IY": (_blend("Eat", "If"), None),                    # "beat" - Eat dominant + If blend
    "OW": (_blend("Oat", "Wet"), _blend("Ox", "Wet")),   # diphthong (boat)
    "UW": (_blend("Wet", "Ox"), None),                    # "boot" - Wet + Ox blend
    "AE": (_blend("Earth", "Eat", "If"), None),           # "cat" - Earth+Eat+If (official: 308x)
    "AO": (_blend("Earth", "Ox"), None),                  # "thought" - Earth + Ox blend
    "AW": (_blend("Ox", "Earth"), _blend("Wet", "Ox")),  # diphthong (bout)
    "OY": (_blend("Oat", "If"), _blend("Eat", "If")),    # diphthong (boy)
    "UH": (_blend("Though", "If"), None),                 # "book" - Though + If blend
}

# ── CONSONANT map: name → mouth shape ({} = invisible, e.g. G/H/K/NG) ──
_CONS: dict[str, dict] = {
    "B": _pair("Bump"), "CH": _pair("Told"), "D": _pair("Told"),
    "DH": _pair("Told"), "F": _pair("Fave"), "G": {}, "HH": {},
    "JH": _pair("Told"), "K": {}, "L": _pair("Told"), "M": _pair("Bump"),
    "N": _pair("New"), "NG": {}, "P": _pair("Bump"), "R": _pair("Roar"),
    "S": _pair("Size"), "SH": _pair("Size"), "T": _pair("Told"),
    "TH": _pair("Told"), "V": _pair("Fave"), "W": _pair("Wet"),
    "Y": _pair("Eat"), "Z": _pair("Size"), "ZH": _pair("Size"),
}

_VOWEL_SET = set(_VOWELS)


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


def _resolve_shapes(spans, lang: str = "en") -> list:
    """One mouth shape per span, grouping consecutive same-word syllables (trailing
    '-'/'=') so the whole word is looked up in CMUdict and aligned per syllable.

    `spans` = list of (start_s, end_s, text[, ...]) — the text is the lyric
    fragment. Returns one shape per span, in the same order."""
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


# ══════════════════════════════════════════════════════════════════════
#  Onyx-style keyframe generation
# ══════════════════════════════════════════════════════════════════════
# Mirrors Onyx's Lipsync.hs timing model (syllablesToAnimations +
# animationsToEvents/animationsToStates): each syllable is a vocal tube whose
# LENGTH (start→end) drives the mouth. The mouth opens ~half-transition before
# the note, holds the vowel for the note body, and closes ~half-transition
# after. Transitions are linear (easeInExpo for diphthong glides). Blink comes
# from charted [eyes close]/[eyes open] events (weight 255).

FPS = 30

_VIDX = {name: i for i, name in enumerate(VISEMES)}

_TRANSITION_S = 0.12
_HALF_TRANSITION_S = _TRANSITION_S / 2.0
_BLINK_WEIGHT = 255


def _ease_in_expo(t: float) -> float:
    """Onyx's easeInExpo curve for diphthong glides."""
    return 0.0 if t == 0.0 else 2.0 ** (10.0 * t - 10.0)


def _scale_shapes(shapes, k: float):
    """Scale every mouth viseme weight by ``k`` (mouth_openness, 0.0–1.0)."""
    if k == 1.0:
        return shapes

    def s(d):
        return {n: max(0, min(255, round(w * k))) for n, w in d.items()}

    out = []
    for initial, (vmain, vend), final in shapes:
        out.append(([s(c) for c in initial],
                    (s(vmain), s(vend) if vend else None),
                    [s(c) for c in final]))
    return out


def _syllable_segments(spans, shapes):
    """Onyx ``syllablesToAnimations`` equivalent.

    Each span is a vocal tube ``(start, end)``. Builds the ordered animation
    segments for the whole song as ``(start_s, kind, va, vb)`` where ``kind`` is
    ``"hold"`` (hold ``va``), ``"line"`` (linear ``va``→``vb``) or ``"fall"``
    (easeInExpo ``va``→``vb``, a diphthong glide). Segment durations are
    implicit: a segment lasts until the next segment's start (the frame sampler
    is handed a song length so the last segment extends to the end).

    Timing per syllable (t1=start, t2=end), with ``half`` = 0.06 s:
      initial_front = min(half, gap_before − prev_final_back)   (Onyx trackDrop)
      initial_back  = min(half, note_len / 2)
      final_front   = min(half, note_len / 2)
      final_back    = min(half, gap_after / 2)
    The mouth transitions default→(initial consonants)→vowel over
    ``initial_front + initial_back``, holds the vowel for
    ``note_len − initial_back − final_front``, then transitions
    vowel→(final consonants)→default over ``final_front + final_back``, and
    finally holds default (closed) through the gap."""
    n = len(spans)

    final_back = [0.0] * n
    for i in range(n):
        if i + 1 < n:
            gap = spans[i + 1][0] - spans[i][1]
            final_back[i] = min(_HALF_TRANSITION_S, max(0.0, gap) / 2.0)
        else:
            final_back[i] = _HALF_TRANSITION_S

    initial_front = [0.0] * n
    prev_fb = 0.0
    for i in range(n):
        gap_before = spans[i][0] - (spans[i - 1][1] if i > 0 else 0.0)
        initial_front[i] = min(_HALF_TRANSITION_S,
                               max(0.0, gap_before - prev_fb))
        prev_fb = final_back[i]

    segs: list[tuple] = []
    default: dict = {}
    for i in range(n):
        t1, t2 = spans[i][0], spans[i][1]
        initial, (vmain, vend), final = shapes[i]
        initial = [c for c in initial if c]   # drop invisible consonants {}
        final = [c for c in final if c]
        note_len = max(0.0, t2 - t1)

        ifb = initial_front[i]
        ibk = min(_HALF_TRANSITION_S, note_len / 2.0)
        ffr = min(_HALF_TRANSITION_S, note_len / 2.0)
        fbk = final_back[i]
        vend_state = vend if vend is not None else vmain

        # 1. transition IN: default → initial consonants → vowel
        chain = [default] + initial + [vmain]
        k = len(chain) - 1
        in_start = t1 - ifb
        in_dur = ifb + ibk
        if in_dur > 0.0:
            step = in_dur / k
            for j in range(k):
                segs.append((in_start + j * step, "line",
                             chain[j], chain[j + 1]))

        # 2. vowel body (hold, or easeInExpo glide for a diphthong)
        vowel_start = t1 + ibk
        vowel_dur = note_len - ibk - ffr
        if vowel_dur > 0.0:
            if vend is not None:
                segs.append((vowel_start, "fall", vmain, vend))
            else:
                segs.append((vowel_start, "hold", vmain, None))

        # 3. transition OUT: vowel → final consonants → default
        chain = [vend_state] + final + [default]
        k = len(chain) - 1
        out_start = t2 - ffr
        out_dur = ffr + fbk
        if out_dur > 0.0:
            step = out_dur / k
            for j in range(k):
                segs.append((out_start + j * step, "line",
                             chain[j], chain[j + 1]))

        # 4. hold closed (also drives the weight-0 clears in the keyframes)
        segs.append((t2 + fbk, "hold", default, None))

    segs.sort(key=lambda s: s[0])
    return segs


def _eyes_segments(eyes_closed):
    """Blink animation segments from [eyes close]/[eyes open] spans (Onyx step 7).

    ``eyes_closed`` = list of ``(start_s, end_s)``. Blink weight is 255 while the
    eyes are closed, with a half-transition linear ramp at each edge."""
    segs = []
    for start, end in eyes_closed:
        end = max(end, start + 1e-3)
        segs.append((start - _HALF_TRANSITION_S, "line",
                     {"Blink": 0}, {"Blink": _BLINK_WEIGHT}))
        segs.append((start, "hold", {"Blink": _BLINK_WEIGHT}, None))
        segs.append((end, "line", {"Blink": _BLINK_WEIGHT}, {"Blink": 0}))
        segs.append((end + _HALF_TRANSITION_S, "hold", {"Blink": 0}, None))
    return segs


def _interp_dict(va, vb, f: float) -> dict:
    out = {}
    for k in set(va) | set(vb):
        w = round(va.get(k, 0) + (vb.get(k, 0) - va.get(k, 0)) * f)
        if w > 0:
            out[k] = w
    return out


def _segments_to_keyframes(segs):
    """Onyx ``animationsToEvents`` equivalent → ``[(time_s, viseme, weight, graph)]``.

    Each segment contributes its START state with the graph token that is the
    curve OUT of that keyframe (``hold``/``linear``/``ease``), plus weight-0
    clears for any viseme that was active but is no longer."""
    out: list[tuple] = []
    on: set[str] = set()
    for start, kind, va, vb in segs:
        if kind == "hold":
            start_state = dict(va)
            graph = "hold"
        else:
            start_state = dict(va)
            for v in vb:
                start_state.setdefault(v, 0)
            graph = "linear" if kind == "line" else "ease"
        for name, w in start_state.items():
            out.append((start, name, w, graph))
        for name in sorted(on - set(start_state)):
            out.append((start, name, 0, "hold"))
        on = set(start_state)
    out.sort(key=lambda e: (e[0], e[1]))
    return out


def _segments_to_frames(segs, n_frames: int):
    """Onyx ``animationsToStates`` equivalent → ``{frame: {viseme: weight}}`` (30 fps)."""
    frames: dict[int, dict] = {}
    n = len(segs)
    for i, (start, kind, va, vb) in enumerate(segs):
        end = segs[i + 1][0] if i + 1 < n else None
        if end is None or end <= start:
            continue
        f0 = int(math.floor(start * FPS))
        f1 = int(math.ceil(end * FPS))
        for fr in range(max(0, f0), min(n_frames, f1 + 1)):
            t = fr / FPS
            if t < start:
                continue
            if kind == "hold":
                state = va
            else:
                frac = min(1.0, (t - start) / (end - start))
                f = frac if kind == "line" else _ease_in_expo(frac)
                state = _interp_dict(va, vb, f)
            if state:
                slot = frames.setdefault(fr, {})
                for name, w in state.items():
                    if w > slot.get(name, 0):
                        slot[name] = w
    return frames


def lipsync_keyframes_from_spans(spans, lang: str = "en",
                                 eyes_closed=None,
                                 mouth_openness: float = 1.0):
    """Sparse ``(time_s, viseme, weight, graph)`` keyframes (Onyx LIPSYNC# format).

    Timing follows Onyx's ``autoLipsync`` (tube-driven, 0.12 s transitions); the
    mouth shapes come from our own G2P + viseme map. ``eyes_closed`` (list of
    ``(start_s, end_s)``) adds Blink keyframes from charted [eyes close] events.
    ``mouth_openness`` (0.0–1.0) scales every mouth viseme weight."""
    shapes = _scale_shapes(_resolve_shapes(spans, lang), mouth_openness)
    segs = _syllable_segments(spans, shapes) + _eyes_segments(eyes_closed or [])
    segs.sort(key=lambda s: s[0])
    return _segments_to_keyframes(segs)


def frames_from_spans(spans, song_len_s: float, lang: str = "en",
                      eyes_closed=None, mouth_openness: float = 1.0):
    """Dense 30 fps per-frame states (name→weight) for the milo, plus frame count.

    Same tube-driven model as :func:`lipsync_keyframes_from_spans`, sampled to
    dense frames (Onyx's ``animationsToStates``) so the milo carries the exact
    lipsync."""
    shapes = _scale_shapes(_resolve_shapes(spans, lang), mouth_openness)
    segs = _syllable_segments(spans, shapes) + _eyes_segments(eyes_closed or [])
    segs.sort(key=lambda s: s[0])
    n_frames = max(1, int(math.ceil(song_len_s * FPS)) + 1)
    return _segments_to_frames(segs, n_frames), n_frames


def eyes_closed_seconds(track, tempo_map, tpb: int,
                        song_len_s: float | None = None):
    """Seconds of ``[eyes close]``→``[eyes open]`` spans from a vocal track.

    Reads text events matching Onyx's ``vocalEyesClosed`` commands
    (``[eyes close]`` / ``[eyes open]``). Returns ``[(start_s, end_s)]``."""
    from .midi_utils import tick_to_ms
    spans: list[tuple[float, float]] = []
    open_at = None
    t = 0
    for m in track:
        t += m.time
        if m.type not in ("text", "lyrics", "lyric"):
            continue
        txt = (getattr(m, "text", "") or "").strip().lower()
        if not txt or "eyes" not in txt:
            continue
        if "close" in txt:
            if open_at is None:
                open_at = t
        elif "open" in txt:
            if open_at is not None:
                spans.append((tick_to_ms(open_at, tempo_map, tpb) / 1000.0,
                              tick_to_ms(t, tempo_map, tpb) / 1000.0))
                open_at = None
    if open_at is not None and song_len_s:
        spans.append((tick_to_ms(open_at, tempo_map, tpb) / 1000.0,
                      float(song_len_s)))
    return spans

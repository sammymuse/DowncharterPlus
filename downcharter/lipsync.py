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

# Canonical RB3 visemes (34 base + extras found in official milos).
# The first 34 match the static .lipsync format; extras (Brow_openmouthed,
# exp_banger_roar_01) appear in some official milos and are tracked here so
# the dynamic milo table can include them.
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
_VIDX = {n: i for i, n in enumerate(VISEMES)}

# Viseme weights from official milos analysis (83 songs).
# Officials use DIFFERENT weights for _hi and _lo, using MAX values (not averages).
# These are the MAXIMUM weights observed in official .milo_xbox files.
_VISEME_WEIGHTS = {
    # Mouth visemes (pairs) — using MAX values from officials
    "Bump_hi": 102, "Bump_lo": 153,
    "Cage_hi": 102, "Cage_lo": 153,
    "Church_hi": 102, "Church_lo": 153,
    "Earth_hi": 102, "Earth_lo": 153,
    "Eat_hi": 102, "Eat_lo": 153,
    "Fave_hi": 102, "Fave_lo": 153,
    "If_hi": 102, "If_lo": 153,
    "New_hi": 102, "New_lo": 153,
    "Oat_hi": 86, "Oat_lo": 129,
    "Ox_hi": 102, "Ox_lo": 153,
    "Roar_hi": 102, "Roar_lo": 153,
    "Size_hi": 102, "Size_lo": 153,
    "Though_hi": 102, "Though_lo": 153,
    "Told_hi": 102, "Told_lo": 153,
    "Wet_hi": 102, "Wet_lo": 153,
    # Facial expressions (always-on baselines from official milos)
    "Squint": 51,           # 98.5% of frames, weight ~51 (FIXED, not max)
    "Brow_down": 126,       # 70.3% of frames, avg ~126
    "Blink": 105,           # 59.4% of frames, avg ~105
    "Brow_aggressive": 174, # 32.3% of frames, avg ~174
    "Brow_pouty": 170,      # 34.5% of frames, avg ~170
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
    """Blend multiple viseme pairs with different weights.
    
    Officials use 4-6 visemes simultaneously (blending). For example, 'AE' (cat)
    is not just Cage, but Earth+Eat+If blended together. This creates richer,
    more natural mouth shapes.
    
    Args:
        *bases: viseme base names (e.g. 'Earth', 'Eat', 'If')
        weights: relative weights for each base (default: 1.0, 0.7, 0.5, ...)
                 First base gets full weight, others get progressively lower.
    
    Returns:
        dict of viseme_name → weight
    """
    if weights is None:
        # Default: first base 100%, second 70%, third 50%, etc.
        weights = [1.0, 0.7, 0.5, 0.35, 0.25][:len(bases)]
    
    result = {}
    for base, w in zip(bases, weights):
        pair = _pair(base)
        for viseme, weight in pair.items():
            result[viseme] = int(weight * w)
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
_MAX_SUSTAIN_S = 999.0      # effectively no cap — the vowel holds for the full tube duration
_TRANSITION_S = 1.00        # mouth attack/release ramp — ~30 frames @30fps (officials: 10-22 frames)
                            # Officials use smooth sigmoid curves with k=5 (very gentle S-shape).
                            # Longer transitions = much smoother mouth movement.
_WORD_CLOSE_S = 0.05        # minimum mouth-closure duration at word boundaries (~1.5 frames)


def _pad_spans(spans, pad: float = _TRANSITION_S):
    """Magma/YARG autoLipsync model: the mouth opens ~0.12 s BEFORE the tube and
    closes ~0.12 s AFTER it — the transition lives OUTSIDE the note, so the vowel
    holds for the note's whole length. Tubes closer than 2*pad meet in the middle
    of the gap (gap → 0 → the legato path keeps the mouth from closing between
    them), which is why official milos sustain 0.5-4 s episodes across a phrase
    while unpadded spans punched shut between every tube (measured: official
    open median 55.6% vs our 29.5% before this).

    At word boundaries (where ``_word_continues`` is False for the previous span),
    the padding is asymmetrical: the current span's end gets NO forward padding and
    the next span's start is SHIFTED forward by ``_WORD_CLOSE_S``.  This creates a
    real gap between words where the mouth stays visibly closed — the singer's
    micro-breath between words.  Without this gap, back-to-back spans at the same
    tick produce at most 1 frame of closure (0.033 s), which the eye never sees."""
    if not spans:
        return spans
    out = []
    n = len(spans)
    for i, sp in enumerate(spans):
        s, e = sp[0], sp[1]
        rest = tuple(sp[2:])

        # ── Start padding ──────────────────────────────────────────────
        if i == 0:
            ns = s - pad
        else:
            prev_txt = spans[i - 1][2] if len(spans[i - 1]) > 2 else ""
            if _word_continues(prev_txt):
                gap = s - spans[i - 1][1]
                ns = s - min(pad, max(0.0, gap / 2))
            else:
                # Word boundary: ensure at least _WORD_CLOSE_S gap between
                # the padded spans so the mouth stays visibly closed.
                # The natural gap (from note durations) already provides
                # closure when large enough; only top up when it's tight.
                # The forward shift is capped at 0.02 s (~0.6 frame) to
                # avoid shrinking very short syllables (0.08 s rap notes)
                # into near-zero windows.
                gap = s - spans[i - 1][1]
                if gap < _WORD_CLOSE_S:
                    ns = s + min(_WORD_CLOSE_S - gap, 0.02)
                else:
                    ns = s  # natural gap already >= _WORD_CLOSE_S

        # ── End padding ────────────────────────────────────────────────
        cur_txt = sp[2] if len(sp) > 2 else ""
        if i + 1 == n:
            ne = e + pad
        else:
            if _word_continues(cur_txt):
                gap = spans[i + 1][0] - e
                ne = e + min(pad, max(0.0, gap / 2))
            else:
                ne = e  # word boundary: no forward pad into the gap

        ns = max(0.0, ns)
        out.append((ns, max(ne, ns + 1e-3)) + rest)
    return out


def _syllable_points(t: float, dur: float, shape) -> list[tuple[float, dict]]:
    """Control points (time, visemes) of a syllable in the window [t, t+dur].

    Officials use FEWER control points with LONGER transitions. Instead of one
    point per consonant, we GROUP all initial consonants into one point and all
    final consonants into another. This creates fewer, longer transitions.

    Structure: closed → [initial consonants blended] → vowel (hold) → [final consonants blended] → closed
    """
    initial, (vmain, vend), final = shape
    n_i, n_f = len(initial), len(final)
    attack = min(_TRANSITION_S, dur * 0.4)
    release = min(_TRANSITION_S, dur * 0.4)
    inner = max(1e-3, dur - attack - release)
    
    # Group consonants: blend all initial into one point, all final into another
    # This reduces the number of transitions and makes them longer
    initial_blended = {}
    for c in initial:
        for name, w in c.items():
            initial_blended[name] = max(initial_blended.get(name, 0), w)
    
    final_blended = {}
    for c in final:
        for name, w in c.items():
            final_blended[name] = max(final_blended.get(name, 0), w)
    
    # Transition durations: longer for smoother movement
    cons_dur = 0.40  # ~12 frames for consonant group transition
    vowel_dur = inner - 2 * cons_dur  # 2 transitions: initial→vowel, vowel→final
    if vowel_dur < 0.033:
        vowel_dur = 0.033
        cons_dur = max(0.01, (inner - vowel_dur) / 2)
    
    def _clamp(cur):
        return min(cur, t + dur - 1e-6)
    
    pts: list[tuple[float, dict]] = [(t, {})]
    cur = _clamp(t + attack)
    
    # Initial consonants (blended into one point)
    if initial_blended:
        pts.append((cur, initial_blended))
        cur = _clamp(cur + cons_dur)
    
    # Vowel (with hold)
    if vend is not None:  # diphthong
        pts.append((cur, vmain))
        cur = _clamp(cur + vowel_dur * 0.3)
        pts.append((cur, vmain))  # HOLD
        cur = _clamp(cur + vowel_dur * 0.4)
        pts.append((cur, vend))
        cur = _clamp(cur + vowel_dur * 0.3)
    else:  # monophthong
        pts.append((cur, vmain))
        hold_dur = max(0.033, vowel_dur * 0.6)
        pts.append((_clamp(cur + hold_dur), vmain))  # HOLD
        cur = _clamp(cur + vowel_dur)
    
    # Final consonants (blended into one point)
    if final_blended:
        pts.append((cur, final_blended))
        cur = _clamp(cur + cons_dur)
    
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


def _ease_curve(f: float, opening: bool) -> float:
    """Apply sigmoid curve to interpolation fraction.
    
    Officials use smooth S-shaped curves (sigmoid), not linear or simple ease-in/out.
    This creates natural, fluid mouth movements that accelerate in the middle
    and decelerate at the endpoints.
    
    Sigmoid formula: 1 / (1 + exp(-k*(x-0.5)))
    where k controls the steepness (k=5 gives a very gentle, smooth S-curve)."""
    # Sigmoid with k=5 for very gentle, smooth S-curve (officials use k=5-7)
    k = 5.0
    if opening:
        # Ease-in: slow start, accelerate to peak
        return 1.0 / (1.0 + math.exp(-k * (f - 0.5)))
    else:
        # Ease-out: fast start, decelerate to zero
        return 1.0 - (1.0 / (1.0 + math.exp(-k * (f - 0.5))))


def _ease_lerp_state(a: dict, b: dict, f: float) -> dict:
    """Interpolate with ease curve. Determines direction (opening/closing) based
    on whether total weight is increasing or decreasing."""
    # Determine if this is an opening or closing transition
    total_a = sum(a.values())
    total_b = sum(b.values())
    opening = total_b > total_a
    
    # Apply ease curve
    eased_f = _ease_curve(f, opening)
    
    out = {}
    for name in set(a) | set(b):
        wa = a.get(name, 0)
        wb = b.get(name, 0)
        w = int(round(wa + (wb - wa) * eased_f))
        if w > 0:
            out[name] = w
    return out


def _facial_frames_from_keyframes(
    facial_kf: list[tuple[float, str, int, str]],
    n_frames: int,
) -> dict[int, dict]:
    """Sparse (time_s, viseme, weight, graph) → dense 30fps {frame: {viseme: weight}}.

    Graph tokens (from Lipsync.hs confirmed semantics):
      hold   → mantenha o peso até ao próximo keyframe
      linear → interpole linearmente até ao próximo peso
      ease   → exponencial (1 - (1-t)^2) até ao próximo peso"""
    out: dict[int, dict] = {}
    # Group by viseme and sort by time
    by_viseme: dict[str, list[tuple[float, int, str]]] = {}
    for time_s, viseme, weight, graph in facial_kf:
        by_viseme.setdefault(viseme, []).append((time_s, weight, graph))

    for viseme, kfs in by_viseme.items():
        kfs.sort(key=lambda x: x[0])  # stable: same-time twins keep emission order
        # Same-time twins (e.g. 0 then 157 at the same instant — a step) need NO
        # dedup: the previous segment interpolates INTO the first twin (the
        # bracketing zero) over an interval that truncates to the same frame,
        # and the second twin's own segment carries the new value forward.
        # Collapsing them to "last wins" eats the zero and makes the previous
        # segment ramp toward the nonzero twin across the whole gap (a
        # 26-second mouth-opening ramp in testing).
        n = len(kfs)
        for i, (t_i, w_i, g_i) in enumerate(kfs):
            start_frame = max(0, int(t_i * FPS))
            if i + 1 < n:
                t_next, w_next, _ = kfs[i + 1]
                end_frame = min(n_frames, int(t_next * FPS))
                graph = g_i
            else:
                end_frame = n_frames
                graph = "hold"

            for fr in range(start_frame, end_frame):
                if graph == "hold":
                    w = w_i
                elif graph == "linear":
                    t_range = t_next - t_i
                    if t_range > 0:
                        t_local = (fr / FPS) - t_i
                        alpha = min(1.0, t_local / t_range)
                        w = int(w_i + alpha * (w_next - w_i))
                    else:
                        w = w_i
                elif graph == "ease":
                    t_range = t_next - t_i
                    if t_range > 0:
                        t_local = (fr / FPS) - t_i
                        alpha = min(1.0, t_local / t_range)
                        alpha = 1.0 - (1.0 - alpha) ** 2
                        w = int(w_i + alpha * (w_next - w_i))
                    else:
                        w = w_i
                else:
                    w = w_i
                w_clamped = max(0, min(255, w))
                if fr not in out:
                    out[fr] = {}
                out[fr][viseme] = w_clamped
    return out


def _sample_into(frames: dict[int, dict], pts: list[tuple[float, dict]]) -> None:
    """Sample the points at 30 fps and merge into `frames` using CROSSFADE.
    
    Officials use crossfade (blend) between visemes instead of substitution.
    When transitioning from one viseme to another, BOTH are kept active with
    complementary weights that sum to ~100%. This creates much smoother transitions.
    
    Example: transitioning If→Eat over 6 frames:
      Frame 1: If=100%, Eat=0%
      Frame 2: If=80%, Eat=20%
      Frame 3: If=60%, Eat=40%
      ...
      Frame 6: If=0%, Eat=100%
    """
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
        
        # CROSSFADE: blend between sa and sb using eased fraction
        # Both states are kept active with complementary weights
        state = _crossfade_state(sa, sb, frac)
        if not state:
            continue
        slot = frames.setdefault(fr, {})
        for name, w in state.items():
            # Use max for same viseme (in case of overlap within same state)
            if w > slot.get(name, 0):
                slot[name] = w


def _crossfade_state(a: dict, b: dict, f: float) -> dict:
    """Crossfade between two viseme states using eased fraction.
    
    Unlike _lerp_state which replaces visemes, this KEEPS BOTH states active
    with complementary weights. The eased fraction controls the blend ratio.
    
    For example, if a={If_lo: 153} and b={Eat_lo: 153}, at f=0.5:
      result = {If_lo: 77, Eat_lo: 77}  (both active, weights sum to ~153)
    """
    # Apply ease curve to fraction for smooth transition
    total_a = sum(a.values())
    total_b = sum(b.values())
    opening = total_b > total_a
    eased_f = _ease_curve(f, opening)
    
    out = {}
    # Keep visemes from state A with decreasing weight
    for name, w in a.items():
        new_w = int(round(w * (1.0 - eased_f)))
        if new_w > 0:
            out[name] = new_w
    # Add visemes from state B with increasing weight
    for name, w in b.items():
        new_w = int(round(w * eased_f))
        if new_w > 0:
            # If viseme already exists (from state A), use max
            if name in out:
                out[name] = max(out[name], new_w)
            else:
                out[name] = new_w
    return out


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


def lipsync_events_from_spans(spans, song_len_s: float, lang: str = "en",
                             phrase_ends: list[float] | None = None,
                             vocal_notes: list[tuple[float, int]] | None = None,
                             facial_seed: int | None = None):
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
    frames, n_frames = frames_from_spans(
        spans, song_len_s, lang,
        phrase_ends=phrase_ends,
        vocal_notes=vocal_notes,
        facial_seed=facial_seed,
    )
    return list(_delta_frames(frames, n_frames))


def frames_from_spans(spans, song_len_s: float,
                      lang: str = "en",
                      phrase_ends: list[float] | None = None,
                      vocal_notes: list[tuple[float, int]] | None = None,
                      facial_seed: int | None = None,
                      mouth_openness: float = 1.0,
                      ) -> tuple[dict[int, dict], int]:
    """Dense per-frame viseme states (name→weight, 30 fps) from audio-guided spans.

    The shared core of the dense path: resolves one mouth shape per span (grouping
    same-word syllables for CMUdict alignment), samples each syllable's control points
    at 30 fps and merges them (max per viseme), scaling by the per-syllable loudness
    `gain`. Returns `(frames, n_frames)`. `lipsync_events_from_spans` delta-encodes it
    for the MIDI track; `milo.build_song_lipsync` serializes it into the CharLipSync
    that goes inside the .milo (guaranteeing the same lipsync reaches the game).

    When ``phrase_ends`` or ``vocal_notes`` are provided, facial animation keyframes
    (Blink, Squint, Eyebrows) are also embedded so they reach the game via the milo."""
    spans = _pad_spans(list(spans))
    shapes = _resolve_shapes(spans, lang)
    frames: dict[int, dict] = {}
    for sp, shape in zip(spans, shapes):
        t, end = sp[0], sp[1]
        gain = sp[3] if len(sp) > 3 else 1.0
        dur = end - t
        if dur <= 0:
            continue
        # Official milos use FIXED weights per viseme (not scaled by gain).
        # The gain affects timing/duration, not weight magnitude.
        # Only mouth_openness (user parameter) scales the weights.
        pts = _syllable_points(t, dur, shape)
        if mouth_openness != 1.0:
            pts = [(pt_t, {nm: max(0, min(255, int(round(w * mouth_openness))))
                           for nm, w in st.items()})
                   for pt_t, st in pts]
        _sample_into(frames, pts)
    n_frames = max(1, int(math.ceil(song_len_s * FPS)) + 1)

    # Inject facial animation keyframes (Blink, Squint, Eyebrows) into the milo.
    if song_len_s is not None and song_len_s > 0 and (phrase_ends or vocal_notes):
        gains: list[tuple[float, float]] = [(sp[0], sp[3] if len(sp) > 3 else 1.0)
                                            for sp in spans]
        facial_kf = generate_facial_keyframes(
            song_len_s, phrase_ends, facial_seed, vocal_notes, gains)
        if facial_kf:
            facial_frames = _facial_frames_from_keyframes(facial_kf, n_frames)
            # Merge — max per viseme (mouth and facial visemes are disjoint sets,
            # but max is still the safe choice).
            for fr, visemes in facial_frames.items():
                if fr not in frames:
                    frames[fr] = {}
                for viseme, w in visemes.items():
                    if viseme not in frames[fr]:
                        frames[fr][viseme] = w
                    else:
                        frames[fr][viseme] = max(frames[fr][viseme], w)

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
    STARTS at it: a held vowel plateau (`hold`) and the diphthong glide (`ease`).

    Officials use FEWER control points with LONGER transitions. We GROUP all
    initial consonants into one point and all final consonants into another."""
    initial, (vmain, vend), final = shape
    n_i, n_f = len(initial), len(final)
    attack = min(_TRANSITION_S, dur * 0.4)
    release = min(_TRANSITION_S, dur * 0.4)
    inner = max(1e-3, dur - attack - release)
    
    # Group consonants: blend all initial into one point, all final into another
    initial_blended = {}
    for c in initial:
        for name, w in c.items():
            initial_blended[name] = max(initial_blended.get(name, 0), w)
    
    final_blended = {}
    for c in final:
        for name, w in c.items():
            final_blended[name] = max(final_blended.get(name, 0), w)
    
    # Transition durations: longer for smoother movement
    cons_dur = 0.40  # ~12 frames for consonant group transition
    vowel_dur = inner - 2 * cons_dur  # 2 transitions: initial→vowel, vowel→final
    if vowel_dur < 0.033:
        vowel_dur = 0.033
        cons_dur = max(0.01, (inner - vowel_dur) / 2)
    
    def _clamp(cur):
        return min(cur, t + dur - 1e-6)
    
    pts: list[tuple[float, dict, str]] = [(t, {}, "ease")]
    cur = _clamp(t + attack)
    
    # Initial consonants (blended into one point)
    if initial_blended:
        pts.append((cur, initial_blended, "ease"))
        cur = _clamp(cur + cons_dur)
    
    # Vowel (with hold)
    if vend is not None:  # diphthong
        pts.append((cur, vmain, "ease"))
        cur = _clamp(cur + vowel_dur * 0.3)
        pts.append((cur, vmain, "hold"))  # HOLD
        cur = _clamp(cur + vowel_dur * 0.4)
        pts.append((cur, vend, "ease"))
        cur = _clamp(cur + vowel_dur * 0.3)
    else:  # monophthong
        pts.append((cur, vmain, "ease"))
        hold_dur = max(0.033, vowel_dur * 0.6)
        pts.append((_clamp(cur + hold_dur), vmain, "hold"))  # HOLD
        cur = _clamp(cur + vowel_dur)
        pts.append((cur, vmain, "ease"))  # release
    
    # Final consonants (blended into one point)
    if final_blended:
        pts.append((cur, final_blended, "ease"))
        cur = _clamp(cur + cons_dur)
    
    pts.append((t + dur, {}, "ease"))
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
                   "Brow_openmouthed", "Brow_pouty", "Brow_up", "Squint")

# Timing constants (seconds)
_BLINK_INTERVAL = (2.0, 10.0)     # random uniform range
_BLINK_CLOSE_S = 0.06             # close duration
_BLINK_HOLD_S = 0.05              # hold closed
_BLINK_OPEN_S = 0.08              # open duration
_BLINK_WEIGHT = 98                # official avg (was 140)

# Squint: ALWAYS ACTIVE in officials (98.5% of frames, weight ~51).
# We emit it as a constant baseline, not periodic.
_SQUINT_WEIGHT = 51               # official avg (was 100)

# Eyebrow pitch thresholds (MIDI note numbers).
# C5 = 72, G3 = 55, perfect fifth = 7 semitones.
_BROW_AGGRESSIVE_PITCH = 72       # notes >= C5 → Brow_aggressive
_BROW_DOWN_PITCH = 55             # notes <= G3 sustained → Brow_down
_BROW_UP_JUMP = 7                 # consecutive note jump >= 7 → Brow_up
_BROW_AGGRESSIVE_WEIGHT = 165     # official avg (was 120)
_BROW_DOWN_WEIGHT = 110
_BROW_UP_WEIGHT = 100
_BROW_POUTY_WEIGHT = 164          # official avg (was 90)
_BROW_DEFAULT_W = 112             # official avg for always-on Brow_down (was 70)
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
        out.append((t + _BLINK_CLOSE_S + _BLINK_HOLD_S + _BLINK_OPEN_S,
                    "Blink", 0, "hold"))                       # hold open
        t += rng.uniform(*_BLINK_INTERVAL)

    # Phrase-boundary bonus blinks: 40 % chance of an extra blink just
    # after each phrase end (but not within 0.5 s of an existing blink).
    if phrase_ends:
        blink_times = {e[0] for e in out}
        for pe in phrase_ends:
            if pe < 0 or pe >= song_len_s:
                continue
            if any(abs(pe - bt) < 0.5 for bt in blink_times):
                continue
            if rng.random() < 0.4:
                out.append((pe, "Blink", _BLINK_WEIGHT, "ease"))
                out.append((pe + _BLINK_CLOSE_S, "Blink", _BLINK_WEIGHT, "hold"))
                out.append((pe + _BLINK_CLOSE_S + _BLINK_HOLD_S, "Blink", 0, "linear"))
                out.append((pe + _BLINK_CLOSE_S + _BLINK_HOLD_S + _BLINK_OPEN_S,
                            "Blink", 0, "hold"))
                blink_times.update({pe, pe + _BLINK_CLOSE_S,
                                    pe + _BLINK_CLOSE_S + _BLINK_HOLD_S,
                                    pe + _BLINK_CLOSE_S + _BLINK_HOLD_S + _BLINK_OPEN_S})

    out.sort(key=lambda e: (e[0], e[1]))
    return out


def _generate_squints(
    song_len_s: float,
    rng=None,
    gains: list[tuple[float, float]] | None = None,
) -> list[tuple[float, str, int, str]]:
    """Squint is ALWAYS ACTIVE in official milos (98.5% of frames, weight ~51).
    
    Instead of periodic squints, we emit a constant baseline Squint at weight 51
    for the entire song. Extra squint emphasis on loud syllables is handled by
    temporarily increasing the weight (not implemented yet — future enhancement)."""
    out: list[tuple[float, str, int, str]] = []
    # Constant Squint baseline for the entire song
    out.append((0.0, "Squint", _SQUINT_WEIGHT, "hold"))
    out.append((song_len_s, "Squint", 0, "linear"))  # close at song end
    return out


_BROW_LOUD_GAIN = 0.78     # phrase avg gain >= this reads as a belted/intense phrase
                            # (~50th percentile of [0.55, 1.0], was 0.85)
_BROW_LOUD_JUMP = 0.20     # gain rise between consecutive phrases that reads as a surge
                            # (~44% of max possible jump, was 0.35)


def _nearest_gain(gains: list[tuple[float, float]], t: float) -> float:
    """Loudness gain (see `audio.syllable_gain`) for the syllable closest to `t`,
    or 1.0 (neutral) if `gains` is empty."""
    if not gains:
        return 1.0
    best = min(gains, key=lambda g: abs(g[0] - t))
    return best[1]


def _generate_eyebrows(
    vocal_notes: list[tuple[float, int]],
    song_len_s: float,
    rng=None,
    gains: list[tuple[float, float]] | None = None,
) -> list[tuple[float, str, int, str]]:
    """Eyebrow animation keyframes driven by vocal pitch, loudness and intensity.

    Official milo analysis shows **Brow_down is the DEFAULT** (present in
    ~98 % of frames across all songs), not an event.  Other brow visemes
    (aggressive / up / pouty) override it when active.

    ``vocal_notes`` is a sorted list of ``(start_s, midi_pitch)`` from the
    PART VOCALS track. Most charts here are TALKY (unpitched, fixed pitch) —
    pitch alone then reads as flat and the brows barely react. ``gains``
    (sorted ``(start_s, gain)`` from the audio-guided syllable loudness, see
    `audio.syllable_gain`) is used alongside pitch so a belted/screamed
    passage still gets an expressive brow even with no real melody. Rules, in
    priority order (higher wins):

    * **Brow_aggressive** — high pitch (>= C5/72), dense phrasing (>= 3
      notes/s within a phrase), OR a loud/belted phrase (avg gain >=
      ``_BROW_LOUD_GAIN``).  Overrides Brow_down.
    * **Brow_up** — a pitch jump >= ``_BROW_UP_JUMP`` semitones between
      consecutive notes, OR a loudness surge (``_BROW_LOUD_JUMP``) into the
      next phrase.  Brief flash, overrides Brow_down.
    * **Brow_pouty** — rare; triggered randomly in ~5 % of phrases that
      don't already have another override.  Overrides Brow_down.
    * **Brow_down** — default state at moderate weight (``_BROW_DEFAULT_W``).
      Active whenever no other brow viseme is.
    """
    if rng is None:
        import random as rng
    out: list[tuple[float, str, int, str]] = []
    gains = gains or []
    
    _DW = _BROW_DEFAULT_W
    _AW = _BROW_AGGRESSIVE_WEIGHT
    _DWb = _BROW_DOWN_WEIGHT
    _UW = _BROW_UP_WEIGHT
    _PW = _BROW_POUTY_WEIGHT
    
    # If no vocal_notes, generate Brow_down as default using gains only
    if not vocal_notes:
        if not gains:
            return out
        # Use gains to create brow segments (loud = aggressive, quiet = down)
        # Group gains into phrases (gap > 0.5s = boundary)
        phrases: list[list[tuple[float, float]]] = [[gains[0]]]
        for i in range(1, len(gains)):
            if gains[i][0] - gains[i - 1][0] > 0.5:
                phrases.append([])
            phrases[-1].append(gains[i])
        
        for phrase in phrases:
            if not phrase:
                continue
            t0 = phrase[0][0]
            t1 = min(phrase[-1][0] + _BROW_HOLD_S + _BROW_FADE_S, song_len_s)
            avg_gain = sum(g for _, g in phrase) / len(phrase)
            
            # Loud phrase → Brow_aggressive, quiet → Brow_down
            if avg_gain >= _BROW_LOUD_GAIN:
                expr, w = "aggressive", _AW
            elif rng.random() < 0.05 and len(phrase) >= 3:
                expr, w = "pouty", _PW
            else:
                expr, w = "down", _DWb
            
            # Close-off the default Brow_down before any override
            if expr != "down":
                out.append((t0, "Brow_down", 0, "linear"))
                vis = f"Brow_{expr}"
                out.append((t0, vis, w, "ease"))
                out.append((t0 + _BROW_HOLD_S, vis, w, "hold"))
                out.append((t1, vis, 0, "linear"))
                out.append((t1, "Brow_down", _DW, "linear"))
        
        # Ensure Brow_down is active at start and end
        if out:
            out.insert(0, (0.0, "Brow_down", _DW, "linear"))
            out.append((song_len_s, "Brow_down", 0, "linear"))
        else:
            # No phrases detected, just add default Brow_down
            out.append((0.0, "Brow_down", _DW, "linear"))
            out.append((song_len_s, "Brow_down", 0, "linear"))
        
        out.sort(key=lambda e: (e[0], e[1]))
        return out

    # ── Group consecutive notes into phrases (gap > 0.5 s = boundary) ──
    phrases: list[list[tuple[float, int]]] = [[vocal_notes[0]]]
    for i in range(1, len(vocal_notes)):
        if vocal_notes[i][0] - vocal_notes[i - 1][0] > 0.5:
            phrases.append([])
        phrases[-1].append(vocal_notes[i])

    # ── Determine the effective brow per phrase ──
    # Each entry: (start_s, end_s, expression, weight)
    # expression is one of: "aggressive", "down", "pouty", "up", None (default)
    brow_segments: list[tuple[float, float, str | None, int]] = []
    phrase_gains: list[float] = []

    for phrase in phrases:
        if len(phrase) < 1:
            phrase_gains.append(1.0)
            continue
        pitches = [p for _, p in phrase]
        phrase_g = [_nearest_gain(gains, t) for t, _ in phrase]
        avg_gain = sum(phrase_g) / len(phrase_g)
        phrase_gains.append(avg_gain)
        t0 = phrase[0][0]
        t1 = min(phrase[-1][0] + _BROW_HOLD_S + _BROW_FADE_S,
                 song_len_s)
        max_p = max(pitches)
        dur = phrase[-1][0] - phrase[0][0]
        density = len(phrase) / max(dur, 0.01)

        # Priority order: aggressive > up > pouty > down
        expr: str | None = None
        w = 0

        if (max_p >= _BROW_AGGRESSIVE_PITCH or (density >= 3.0 and len(phrase) >= 4)
                or avg_gain >= _BROW_LOUD_GAIN):
            expr, w = "aggressive", _AW
        # Sustained low-pitch phrase: dur is the phrase span (gap ≤ 0.5 s
        # between notes).  A phrase with several short notes can still be
        # "sustained" by this measure — that is a known trade-off; the span
        # correlates well with the singer staying in a low register.
        elif dur > _BROW_HOLD_S + _BROW_FADE_S and sum(pitches) / len(pitches) < _BROW_DOWN_PITCH:
            expr, w = "down", _DWb
        elif rng.random() < 0.05 and len(phrase) >= 3:
            expr, w = "pouty", _PW
        # else: keep default Brow_down

        brow_segments.append((t0, t1, expr, w))

    # ── Brow_up flashes are independent of phrases ──
    brow_up_times: set[float] = set()
    for i in range(1, len(vocal_notes)):
        prev_t, prev_p = vocal_notes[i - 1]
        cur_t, cur_p = vocal_notes[i]
        if cur_p - prev_p >= _BROW_UP_JUMP:
            brow_up_times.add(cur_t)
    # A loudness surge between consecutive PHRASES also reads as a "surprise"
    # flash (e.g. a quiet verse suddenly bursting into a belted line) — flash
    # at the start of the louder phrase.
    for i in range(1, len(phrases)):
        if phrases[i] and phrase_gains[i] - phrase_gains[i - 1] >= _BROW_LOUD_JUMP:
            brow_up_times.add(phrases[i][0][0])

    # ── Build keyframes ──
    # Start with DEFAULT Brow_down at tick 0.
    out.append((0.0, "Brow_down", _DW, "linear"))

    for seg_start, seg_end, expr, w in brow_segments:
        # Close-off the default Brow_down before any override at this segment.
        if expr is not None:
            out.append((seg_start, "Brow_down", 0, "linear"))
            vis = f"Brow_{expr}"
            out.append((seg_start, vis, w, "ease"))
            out.append((seg_start + _BROW_HOLD_S, vis, w, "hold"))
            out.append((seg_end, vis, 0, "linear"))
            out.append((seg_end, "Brow_down", _DW, "linear"))

        # Insert Brow_up flash if it falls within this segment's time window.
        flash_in = [ut for ut in brow_up_times if seg_start <= ut < seg_end]
        for ft in flash_in:
            if expr == "aggressive":
                continue  # aggressive has higher priority — skip flash
            out.append((ft, "Brow_down", 0, "linear"))
            out.append((ft, "Brow_up", _UW, "ease"))
            out.append((min(ft + _BROW_HOLD_S, seg_end),
                        "Brow_up", _UW, "hold"))
            out.append((min(ft + _BROW_HOLD_S + _BROW_FADE_S, seg_end),
                        "Brow_up", 0, "linear"))
            # Restore the segment's OWN brow expression, not unconditionally
            # Brow_down — this avoids overriding e.g. Brow_pouty.
            if expr is not None:
                rest_vis = f"Brow_{expr}"
                rest_w = w
            else:
                rest_vis = "Brow_down"
                rest_w = _DW
            out.append((min(ft + _BROW_HOLD_S + _BROW_FADE_S, seg_end),
                        rest_vis, rest_w, "linear"))
            brow_up_times.discard(ft)

    # Close Brow_down at song end.
    out.append((song_len_s, "Brow_down", 0, "linear"))

    out.sort(key=lambda e: (e[0], e[1]))
    return out


def generate_facial_keyframes(
    song_len_s: float,
    phrase_ends: list[float] | None = None,
    rng_seed: int | None = None,
    vocal_notes: list[tuple[float, int]] | None = None,
    gains: list[tuple[float, float]] | None = None,
) -> list[tuple[float, str, int, str]]:
    """All facial animation keyframes (Blink, Squint, Brow_*) for the whole song.

    Returns sparse keyframes in the same ``(time_s, viseme, weight, graph)``
    format as :func:`lipsync_keyframes_from_spans`, ready to be merged.

    When ``vocal_notes`` (list of ``(start_s, midi_pitch)``) is provided,
    eyebrow expressions are generated based on pitch: high notes raise inner
    brows (Brow_aggressive), low sustained notes furrow (Brow_down), and
    large pitch jumps cause surprise (Brow_up). ``gains`` (sorted
    ``(start_s, gain)`` per-syllable loudness, see `audio.syllable_gain`) adds
    the same reactions driven by LOUDNESS instead of pitch — most charted
    vocals here are talky/unpitched, so pitch alone reads flat; loudness keeps
    the face expressive on belted or screamed passages regardless."""
    import random as _rng
    rng = _rng.Random(rng_seed) if rng_seed is not None else _rng
    out: list[tuple[float, str, int, str]] = []
    out.extend(_generate_blinks(song_len_s, phrase_ends, rng))
    out.extend(_generate_squints(song_len_s, rng, gains))
    # Always generate eyebrows (Brow_down is default, even without vocal_notes)
    out.extend(_generate_eyebrows(vocal_notes or [], song_len_s, rng, gains))
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
    mouth_openness: float = 1.0,
) -> list[tuple[float, str, int, str]]:
    """(time_s, viseme, weight, graph) sparse keyframes for a LIPSYNC# MIDI track.

    ``mouth_openness`` (0.0–1.0) scales the weight of every mouth viseme,
    letting the user reduce the overall mouth activity for songs where the
    singer barely moves their lips (spoken-word, rap, etc.).  At 1.0 the
    mouth opens fully for each syllable (standard RB3 autoLipsync); at 0.5
    the shapes are half as pronounced; at 0.0 the mouth stays visually
    closed.  Facial animations (blink, squint, brows) are NOT affected.

    Production path. Same audio-guided spans as `lipsync_events_from_spans`, but emits
    sparse keyframes with Onyx graph tokens (held vowels = `hold`, diphthong glides =
    `ease`, transitions = linear) instead of baking the curve into dense 30fps deltas.
    Far fewer events and closer to how Onyx itself authors the milo.

    When ``song_len_s`` is provided, facial animation keyframes are generated alongside
    the mouth shapes and merged into the output.  ``facial_seed`` makes random blink/
    squint timing reproducible.  ``vocal_notes`` (list of ``(start_s, midi_pitch)``)
    drives eyebrow expressions via :func:`generate_facial_keyframes`; the same
    per-syllable loudness gain carried in ``spans`` also feeds it, so belted/
    screamed passages stay expressive even when ``vocal_notes`` is flat (talky)."""
    spans = _pad_spans(list(spans))
    shapes = _resolve_shapes(spans, lang)
    out: list[tuple[float, str, int, str]] = []
    gains: list[tuple[float, float]] = []

    for sp, shape in zip(spans, shapes):
        t, end = sp[0], sp[1]
        gain = sp[3] if len(sp) > 3 else 1.0
        gains.append((t, gain))
        dur = end - t
        if dur <= 0:
            continue
        pts = _syllable_points_g(t, dur, shape)
        out.extend(_keyframes_from_points(pts, mouth_openness))
    if song_len_s is not None and song_len_s > 0:
        facial = generate_facial_keyframes(
            song_len_s, phrase_ends, facial_seed, vocal_notes, gains)
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

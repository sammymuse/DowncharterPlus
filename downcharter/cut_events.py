"""Cut-event detection layer (Phase 0 of the cut-algorithm rework).

PURE detectors that scan a song's structure/onsets and produce a timeline of
*musical events*, each owning the directed cut(s) that make sense for it (see
docs/CUTS_ALGORITHM_STUDY.md §2.2). This module has NO side effects on the current
`build_camera` output — it is a separate, testable layer. Phase 1+ will route the
camera through it.

Each `CutEvent` carries:
  tick     — exact hit tick (where the directed cut should land)
  etype    — event kind (string, see EVENT_TYPES)
  cuts     — candidate cut names, MOST→LEAST specific (first that passes guards wins)
  priority — higher = more specific / wins when two events collide
  dramatic — counts against the full-band/dramatic budget & throttle
  note     — short human description (for the dev diagnostic)

Pipeline: original 10-detector system replaced with 4-level Onyx system
(CLOSEUP / DUO / GENERAL / FULLBAND) for directed cuts.
"""
from __future__ import annotations

from dataclasses import dataclass

from .venue import Section, measure_ticks_at


@dataclass
class CutEvent:
    tick: int
    etype: str
    cuts: list[str]
    priority: int
    dramatic: bool = False
    note: str = ""


# Priority ladder (higher wins on collision). BRE/stagedive are unique moments;
# section_entry/close anchor boundaries.
PRIO = {
    "bre": 100, "stagedive": 90, "section_entry": 85, "section_close": 75,
}

_MELODIC = ("guitar", "bass", "keys")
_RANK = {"calm": 0, "mid": 1, "high": 2}

# The leader rank is SONG-RELATIVE (cnt/own-total). That over-rewards a bass that
# is sparse over the whole song: in a riff the guitar and bass play near-identical
# absolute counts, but the bass's smaller denominator makes it "step up" more, so
# riffs were handed to the bassist. The guitar is the visual focal of a riff, so we
# give it a modest multiplier — enough to win riffs it co-plays with the bass while
# leaving genuine vocal/drum leads (which score far higher) untouched.
_GUITAR_LEAD_BIAS = 1.35


def _onsets_in(onsets: list[int] | None, a: int, b: int) -> list[int]:
    if not onsets:
        return []
    import bisect
    lo = bisect.bisect_left(onsets, a)
    hi = bisect.bisect_left(onsets, b)
    return onsets[lo:hi]


def _nearest_accent(t: int, accents: list[int], tpb: int) -> int:
    """Snap a hit to the nearest structural accent within ±1 beat, else the tick itself."""
    if not accents:
        return t
    import bisect
    i = bisect.bisect_left(accents, t)
    best, bestd = t, tpb + 1
    for ci in (i - 1, i):
        if 0 <= ci < len(accents):
            d = abs(accents[ci] - t)
            if d <= tpb and d < bestd:
                best, bestd = accents[ci], d
    return best


def _phrases(vocal: list[int], time_sig_map: list, tpb: int) -> list[tuple[int, int, int]]:
    """Group vocal onsets into phrases: a gap >= 1 measure splits phrases.
    Returns (start, last_onset, n_notes) per phrase."""
    if not vocal:
        return []
    vocal = sorted(vocal)
    phrases, ps, prev, n = [], vocal[0], vocal[0], 1
    for o in vocal[1:]:
        if o - prev >= measure_ticks_at(prev, time_sig_map, tpb):
            phrases.append((ps, prev, n))
            ps, n = o, 0
        prev = o
        n += 1
    phrases.append((ps, prev, n))
    return phrases


# Single-character close-up per instrument (the feature shot for whoever leads).
_FEATURE_CLOSEUP = {
    "guitar": ["D_Gtr_CLS", "D_Gtr"], "bass": ["D_Bass_CLS", "D_Bass"],
    "keys": ["D_Keys_Cam", "D_Keys"], "drums": ["D_Drums_LT", "D_Drums"],
    "vocal": ["D_Vox_CLS", "D_Vox_Cam_PT"],
}

# Interaction shot for two members that co-lead a section.
_FEATURE_DUO: dict[frozenset[str], str] = {
    frozenset({"guitar", "bass"}): "D_Duo_GB",
    frozenset({"keys", "bass"}): "D_Duo_KB",
    frozenset({"keys", "guitar"}): "D_Duo_KG",
    frozenset({"keys", "vocal"}): "D_Duo_KV",
    frozenset({"guitar", "vocal"}): "D_Duo_Gtr",
    frozenset({"bass", "vocal"}): "D_Duo_Bass",
}


def _section_leaders(inst_onsets: dict[str, list[int]], start: int, end: int,
                     totals: dict[str, int]) -> list[tuple[str, float]]:
    """Rank the instruments by SONG-RELATIVE density in [start,end) — onsets-in-window
    over the instrument's song total, so a continuously-busy drummer doesn't always win;
    we surface whoever STEPS UP here. Returns (inst, score) sorted desc, present only."""
    import bisect
    out: list[tuple[str, float]] = []
    for inst in ("guitar", "bass", "keys", "drums", "vocal"):
        ons = inst_onsets.get(inst)
        tot = totals.get(inst, 0)
        if not ons or tot < 4:
            continue
        cnt = bisect.bisect_left(ons, end) - bisect.bisect_left(ons, start)
        if cnt < 2:
            continue
        score = cnt / tot
        if inst == "guitar":
            score *= _GUITAR_LEAD_BIAS
        out.append((inst, score))
    out.sort(key=lambda x: -x[1])
    return out


def _onset_near(ons: list[int], t: int) -> int:
    """The instrument's own onset closest to t (so the feature lands on a real hit of it)."""
    import bisect
    i = bisect.bisect_left(ons, t)
    best = t
    for ci in (i - 1, i):
        if 0 <= ci < len(ons) and (best == t or abs(ons[ci] - t) < abs(best - t)):
            best = ons[ci]
    return best


def _busiest_onset(ons: list[int], start: int, end: int, tpb: int, prefer: int) -> int:
    """The onset in [start,end) where the instrument is MOST ACTIVE — the one with the
    most of its own onsets within ±1 beat. Anchors a feature on the moment the player is
    actually going off, not a geometric section midpoint that can fall in a lull (so a
    cut no longer 'appears too early' when the busy part is later in the section). Ties
    break toward `prefer` (the midpoint) to stay centered when activity is even."""
    import bisect
    seg = ons[bisect.bisect_left(ons, start):bisect.bisect_left(ons, end)]
    if not seg:
        return prefer
    best, best_d, best_p = seg[0], -1, 1 << 62
    for o in seg:
        d = (bisect.bisect_right(seg, o + tpb) - bisect.bisect_left(seg, o - tpb))
        p = abs(o - prefer)
        if d > best_d or (d == best_d and p < best_p):
            best, best_d, best_p = o, d, p
    return best


def _busiest_combined(ons_a: list[int], ons_b: list[int],
                      start: int, end: int, tpb: int, prefer: int) -> int:
    """The tick in [start,end) where COMBINED activity of both onset lists peaks.

    For each onset (from either instrument), counts how many total onsets from BOTH
    are within ±tpb. Picks the one with the highest combined score — guaranteeing
    that _guard_directed's duo check (both playing ±2 beats) passes.
    Ties break toward `prefer` (the section midpoint).
    """
    import bisect
    seg_a = ons_a[bisect.bisect_left(ons_a, start):bisect.bisect_left(ons_a, end)]
    seg_b = ons_b[bisect.bisect_left(ons_b, start):bisect.bisect_left(ons_b, end)]
    all_ticks = sorted(set(seg_a) | set(seg_b))
    if not all_ticks:
        return prefer
    best, best_score, best_dist = all_ticks[0], -1, 1 << 62
    for o in all_ticks:
        ca = bisect.bisect_right(seg_a, o + tpb) - bisect.bisect_left(seg_a, o - tpb)
        cb = bisect.bisect_right(seg_b, o + tpb) - bisect.bisect_left(seg_b, o - tpb)
        score = ca + cb
        dist = abs(o - prefer)
        if score > best_score or (score == best_score and dist < best_dist):
            best, best_score, best_dist = o, score, dist
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  ONYX LEVELS para directed cuts (simplified, replacing 10-detector pipeline)
# ══════════════════════════════════════════════════════════════════════════════

from bisect import bisect_left, bisect_right

class OnyxLevel:
    FULLBAND = 0   # D_All, D_All_Cam, D_All_LT, D_All_Yeah
    GENERAL  = 1   # D_Drums_LT, D_Keys_Cam, D_Crowd_Gtr
    DUO      = 2   # D_Duo_GB, D_Duo_KG, D_Duo_GK, etc.
    CLOSEUP  = 3   # D_Gtr_CLS, D_Bass_CLS, D_Vox_CLS, D_Drums_CLS

# Section kind → nível Onyx base
_SECTION_ONYX = {
    # Data-driven from 100 official songs (2997 cuts):
    # VOX is #1 entry type for verse/prechorus/postchorus/intro
    # CLOSEUP puts Vox first, then Gtr, Bass, Drums
    "solo":       OnyxLevel.CLOSEUP,   # instrument solo → closeup of soloist
    "riff":       OnyxLevel.CLOSEUP,   # instrumental riff → Gtr_CLS
    "outro":      OnyxLevel.CLOSEUP,   # winding down → vocal or instrument
    "verse":      OnyxLevel.CLOSEUP,   # was GENERAL → VOX missed (13× drums_lt confusion)
    "prechorus":  OnyxLevel.CLOSEUP,   # was GENERAL → VOX building energy
    "postchorus": OnyxLevel.CLOSEUP,   # was GENERAL → VOX after climax
    "intro":      OnyxLevel.CLOSEUP,   # was GENERAL → featured instrument
    "build":      OnyxLevel.CLOSEUP,   # was GENERAL → instrumental Gtr_CLS
    "bridge":     OnyxLevel.DUO,       # breakdown of instruments → duo pairs
    "breakdown":  OnyxLevel.FULLBAND,  # climax → full band shots
    "chorus":     OnyxLevel.FULLBAND,  # full band energy → ALL shots
    "drop":       OnyxLevel.FULLBAND,  # EDM drop → full impact
    "default":    OnyxLevel.GENERAL,   # fallback → drums or keys
}

# Candidatos por nível (ordem: mais→menos preferido)
_ONYX_CUTS = {
    OnyxLevel.CLOSEUP: [
        # D_Vox_Cam_PT first: it's the single most common directed cut (10.3%)
        "D_Vox_Cam_PT", "D_Vocals", "D_Vox_Cam_PR",
        "D_Gtr_CLS", "D_Bass_CLS", "D_Drums_CLS",
        "D_Vox_CLS", "D_Gtr_Cam_PT", "D_Bass_Cam",
    ],
    OnyxLevel.DUO: [
        "D_Duo_GB", "D_Duo_KG", "D_Duo_KB", "D_Duo_KV", "D_Duo_Gtr", "D_Duo_Bass",
        "D_Gtr_CLS", "D_Bass_CLS",
    ],
    OnyxLevel.GENERAL: [
        "D_Drums_LT", "D_Keys_Cam", "D_Drums_Point", "D_Drums_KD",
        "D_All", "D_All_Cam",
    ],
    OnyxLevel.FULLBAND: [
        "D_All", "D_All_Cam", "D_All_LT", "D_All_Yeah",
    ],
}

# Mapa: cut → instrumento (para o guard simplificado)
_DIRECTED_INSTR_CUT = {
    "D_Gtr_CLS": "guitar", "D_Gtr_Cam_PT": "guitar",
    "D_Bass_CLS": "bass", "D_Bass_Cam": "bass",
    "D_Drums_CLS": "drums", "D_Drums_LT": "drums",
    "D_Drums_Point": "drums", "D_Drums_KD": "drums",
    "D_Keys_Cam": "keys",
    "D_Vox_CLS": "vocal", "D_Vox_Cam_PT": "vocal",
    "D_Vocals": "vocal", "D_Vox_Cam_PR": "vocal",
}


def _instrument_active(inst: str, inst_onsets: dict, tick: int, tpb: int) -> bool:
    """True if instrument has an onset within ±2 beats of tick (vocal: ±4)."""
    # Vocal onsets are sparser (sustained notes) — use wider window
    win = tpb * 4 if inst == "vocal" else tpb * 2
    key = inst if inst != "vocal" else "_vocal_real"
    ons = inst_onsets.get(key) if inst_onsets else None
    if not ons:
        return False
    i = bisect_left(ons, tick - win)
    return i < len(ons) and ons[i] <= tick + win


def _choose_onyx_cut(level: int, kind: str,
                     inst_onsets: dict | None,
                     tick: int, tpb: int,
                     _depth: int = 0) -> str:
    """Escolhe o melhor cut para o nível Onyx, com fallback progressivo.

    - Tenta cuts do nível atual
    - Se o instrumento não toca, tenta o próximo cut na lista
    - Se nenhum cut do nível é válido, desce um nível (recursivo, max 4)
    - Se chegar a nível < 0, retorna D_All (fallback seguro)
    """
    if _depth > 4 or level < 0:
        return "D_All"
    candidates = _ONYX_CUTS.get(level, _ONYX_CUTS[OnyxLevel.FULLBAND])
    for cut in candidates:
        inst = _DIRECTED_INSTR_CUT.get(cut)
        if inst is None or inst_onsets is None:
            return cut  # full-band ou crowd — sem verificação de instrumento
        if _instrument_active(inst, inst_onsets, tick, tpb):
            return cut
    # Nenhum cut válido neste nível → desce um nível
    return _choose_onyx_cut(level - 1, kind, inst_onsets, tick, tpb, _depth + 1)


def detect_events(sections: list,
                  inst_onsets: dict | None,
                  accents: list[int] | None,
                  bre_spans: list[tuple[int, int]] | None,
                  time_sig_map: list,
                  tpb: int,
                  audio_onsets: list[int] | None = None,
                  energy_env: list | None = None,
                  band_activity: dict | None = None) -> list:
    """Pipeline simplificado: 2 slots por secção + BRE + stagedive.

    Removeu: detect_impacts, detect_energy_transitions, detect_solos,
    detect_downtime, detect_vocal_peaks, detect_duo_cluster, companions,
    Markov, anti-recency, _with_vox_priority, _cluster_limit.
    """
    acc = sorted(accents) if accents else []
    ev: list = []

    # BRE (mantido, é raro e importante)
    for i, (_, end) in enumerate(bre_spans or []):
        cuts = ["D_BRE_Jump", "D_BRE"] if i % 2 == 0 else ["D_BRE", "D_BRE_Jump"]
        ev.append(CutEvent(end, "bre", cuts, PRIO["bre"], dramatic=True, note="BRE"))

    # Stagedive (mantido)
    if inst_onsets:
        vocal = sorted(inst_onsets.get("_vocal_real") or [])
        if len(vocal) >= 2:
            for a, b in zip(vocal, vocal[1:]):
                gap_beats = (b - a) // tpb
                if gap_beats >= 16:
                    tick = _nearest_accent(a + tpb, acc, tpb) if acc else a + tpb
                    ev.append(CutEvent(tick, "stagedive",
                                       ["D_Stagedive", "D_Crowdsurf"],
                                       PRIO["stagedive"], dramatic=True,
                                       note="16+ bar vocal gap"))

    # Slots Onyx: entry + close por secção
    for s in sections:
        kind = s.kind
        level = _SECTION_ONYX.get(kind, OnyxLevel.GENERAL)

        # Slot 1: Entry (no início da secção, snapped ao accent)
        entry_tick = _nearest_accent(s.start, acc, tpb) if acc else s.start
        # Try sliding forward to find a position where a closeup-level instrument is active
        # (vocals often start a few beats after the section marker)
        for offset in [0, tpb, int(tpb * 1.5), tpb * 2]:
            test = entry_tick + offset if offset else entry_tick
            if test > s.start + tpb * 4:  # don't slide past 4 beats
                break
            entry_cut = _choose_onyx_cut(level, kind, inst_onsets, test, tpb)
            # If we found a specific cut (not fullband), keep this position
            if entry_cut not in ("D_All", "D_All_Cam", "D_All_LT", "D_All_Yeah"):
                entry_tick = test
                break
        else:
            # Fallback: no specific cut found at any offset, use original position
            entry_cut = _choose_onyx_cut(level, kind, inst_onsets, entry_tick, tpb)
        ev.append(CutEvent(entry_tick, "section_entry", [entry_cut],
                           PRIO["section_entry"],
                           note=f"onyx_{kind}_entry_{level}"))

        # Slot 2: Close (último quartil, só para secções ≥ 16 beats)
        length = s.end - s.start
        if length >= tpb * 16:
            close_tick = s.start + length * 3 // 4
            close_tick = _nearest_accent(close_tick, acc, tpb) if acc else close_tick
            close_cut = _choose_onyx_cut(level, kind, inst_onsets, close_tick, tpb)
            ev.append(CutEvent(close_tick, "section_close", [close_cut],
                               PRIO["section_close"], dramatic=True,
                               note=f"onyx_{kind}_close_{level}"))

    ev.sort(key=lambda e: (e.tick, -e.priority))
    return ev

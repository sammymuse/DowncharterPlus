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
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .venue import (Section, section_energy, _camera_energy, find_pause_spans,
                    measure_ticks_at)


@dataclass
class CutEvent:
    tick: int
    etype: str
    cuts: list[str]
    priority: int
    dramatic: bool = False
    note: str = ""


# Priority ladder (higher wins on collision). BRE/stagedive are unique moments;
# rises/impact entries anchor structure; the rest are texture.
PRIO = {
    "bre": 100, "stagedive": 90, "rise": 80, "solo": 70, "impact": 60,
    "vocal_peak": 50, "duo": 45, "downtime": 40, "kick": 35, "technical": 30,
}

_MELODIC = ("guitar", "bass", "keys")
_IMPACT_KINDS = {"intro", "chorus", "drop", "breakdown", "outro"}
_RANK = {"calm": 0, "mid": 1, "high": 2}


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


# ── Individual detectors ─────────────────────────────────────────────────────

def detect_rises(sections: list[Section], accents: list[int], tpb: int) -> list[CutEvent]:
    """Upward energy transition into a 'high' section → dircut_at_start (full-band kick)."""
    out, prev = [], "calm"
    for s in sections:
        e = _camera_energy(s)
        if e == "high" and _RANK[e] > _RANK[prev]:
            out.append(CutEvent(_nearest_accent(s.start, accents, tpb), "rise",
                                ["D_All_Yeah", "D_All_LT", "D_All_Cam"], PRIO["rise"],
                                dramatic=True, note=f"rise→high @{s.kind}"))
        prev = e
    return out


def detect_impacts(sections: list[Section], accents: list[int], tpb: int) -> list[CutEvent]:
    """Entry of an impact section (intro/chorus/drop/breakdown/outro) → full-band pan."""
    out = []
    for s in sections:
        if s.kind in _IMPACT_KINDS:
            out.append(CutEvent(_nearest_accent(s.start, accents, tpb), "impact",
                                ["D_All_Cam", "D_All_LT"], PRIO["impact"],
                                dramatic=True, note=f"impact entry @{s.kind}"))
    return out


def detect_bre(bre_spans: list[tuple[int, int]] | None) -> list[CutEvent]:
    """Final note of each BRE → band jump / guitar smash (alternated for variety)."""
    out = []
    for i, (_, end) in enumerate(bre_spans or []):
        cuts = ["D_BRE_Jump", "D_BRE"] if i % 2 == 0 else ["D_BRE", "D_BRE_Jump"]
        out.append(CutEvent(end, "bre", cuts, PRIO["bre"], dramatic=True, note="BRE final note"))
    return out


def detect_solos(sections: list[Section], accents: list[int], tpb: int) -> list[CutEvent]:
    """Solo section → featured soloist: long pre-roll in, close-up, mid-solo flourish."""
    out = []
    table = {
        "guitar": ["D_Gtr_Cam_PR", "D_Gtr_CLS", "D_Gtr_Cam_PT"],
        "bass":   ["D_Bass_Cam", "D_Bass_CLS", "D_Bass"],
        "drums":  ["D_Drums_LT", "D_Drums_KD", "D_Drums"],
        "keys":   ["D_Keys_Cam", "D_Keys"],
        "vocal":  ["D_Vox_Cam_PR", "D_Vox_CLS", "D_Vox_Cam_PT"],
    }
    for s in sections:
        if s.kind != "solo":
            continue
        n = s.name.lower()
        inst = next((k for k in ("bass", "drums", "keys", "vocal", "guitar")
                     if k[:4] in n or k in n), "guitar")
        out.append(CutEvent(_nearest_accent(s.start, accents, tpb), "solo",
                            table[inst], PRIO["solo"], note=f"solo {inst}"))
    return out


def detect_downtime(inst_onsets: dict[str, list[int]] | None,
                    time_sig_map: list, tpb: int) -> list[CutEvent]:
    """A melodic instrument resting ≥2 measures while the song goes on → idle/crowd gesture.
    Gives the _NP and crowd-work cuts a PROACTIVE trigger (today only reactive)."""
    out = []
    if not inst_onsets:
        return out
    # Vocals are EXCLUDED: a singer resting >=2 bars between lines is normal, not a moment
    # to film them idle (they get vocal_peak + stagedive instead). Only INSTRUMENTS that
    # drop out while the band carries on justify an idle/crowd-work shot.
    np_cut = {"guitar": ["D_Crowd_Gtr", "D_Gtr_NP"], "bass": ["D_Crowd_Bass", "D_Bass_NP"],
              "keys": ["D_Keys_NP"], "drums": ["D_Drums_NP"]}
    for inst in ("guitar", "bass", "keys", "drums"):
        ons = inst_onsets.get(inst)
        if not ons:
            continue
        for a, b in find_pause_spans(ons, time_sig_map, tpb, min_measures=4):
            # Only film the idle character if the SONG goes on (another instrument is
            # playing in the span); a full-band break is not a moment to film one idler.
            others = [k for k in ("guitar", "bass", "keys", "drums", "vocal") if k != inst]
            if not any(_onsets_in(inst_onsets.get(k), a, b) for k in others):
                continue
            mid = (a + b) // 2
            out.append(CutEvent(mid, "downtime", np_cut[inst], PRIO["downtime"],
                                note=f"{inst} resting >=2 bars"))
    return out


def detect_vocal_peaks(inst_onsets: dict[str, list[int]] | None,
                       accents: list[int], time_sig_map: list, tpb: int) -> list[CutEvent]:
    """Last note of a vocal phrase (proxy for a sustained/peak note) → vocal close-up."""
    out = []
    if not inst_onsets:
        return out
    vocal = inst_onsets.get("_vocal_real") or []
    last_emit = -10 ** 9
    for ps, last, n in _phrases(vocal, time_sig_map, tpb):
        if n < 2:                     # an isolated stab is not a phrase to peak on
            continue
        if last - last_emit < tpb * 4:  # throttle: at most ~1 vocal close-up / 4 beats
            continue
        out.append(CutEvent(_nearest_accent(last, accents, tpb), "vocal_peak",
                            ["D_Vox_CLS", "D_Vox_Cam_PT"], PRIO["vocal_peak"],
                            note="vocal phrase peak"))
        last_emit = last
    return out


def detect_duos(sections: list[Section], inst_onsets: dict[str, list[int]] | None,
                accents: list[int], tpb: int) -> list[CutEvent]:
    """Two instruments co-active in a mid/high section → duo interaction shot."""
    out = []
    if not inst_onsets:
        return out
    has = {k: bool(inst_onsets.get(k)) for k in ("guitar", "bass", "keys", "drums")}
    real_vox = bool(inst_onsets.get("_vocal_real"))
    for s in sections:
        if _RANK[_camera_energy(s)] < 1:          # only mid/high
            continue
        # most-specific available pair
        if real_vox and has["guitar"]:
            cuts = ["D_Duo_Gtr"]
        elif real_vox and has["bass"]:
            cuts = ["D_Duo_Bass"]
        elif has["guitar"] and has["bass"]:
            cuts = ["D_Duo_GB"]
        elif has["keys"] and has["bass"]:
            cuts = ["D_Duo_KB"]
        elif has["keys"] and has["guitar"]:
            cuts = ["D_Duo_KG"]
        else:
            continue
        mid = (s.start + s.end) // 2
        out.append(CutEvent(_nearest_accent(mid, accents, tpb), "duo", cuts,
                            PRIO["duo"], note=f"duo @{s.kind}"))
    return out


def detect_stagedive(sections: list[Section], inst_onsets: dict[str, list[int]] | None,
                     accents: list[int], time_sig_map: list, tpb: int) -> list[CutEvent]:
    """Long vocal gap (≥16 measures with vocals before AND after) → stage dive / crowd surf.
    The book's only safe placement: hit at the gap start, cut away right after."""
    out = []
    if not inst_onsets:
        return out
    vocal = sorted(inst_onsets.get("_vocal_real") or [])
    if len(vocal) < 2:
        return out
    for a, b in zip(vocal, vocal[1:]):
        if b - a >= measure_ticks_at(a, time_sig_map, tpb) * 16:
            out.append(CutEvent(_nearest_accent(a + tpb, accents, tpb), "stagedive",
                                ["D_Stagedive", "D_Crowdsurf"], PRIO["stagedive"],
                                dramatic=True, note=">=16-bar vocal gap"))
    return out


def detect_technical(sections: list[Section], inst_onsets: dict[str, list[int]] | None,
                     accents: list[int], tpb: int) -> list[CutEvent]:
    """Fast/dense melodic passage (small onset spacing) → fretboard close-up."""
    out = []
    if not inst_onsets:
        return out
    for s in sections:
        for inst, cut in (("guitar", "D_Gtr_CLS"), ("bass", "D_Bass_CLS")):
            seg = _onsets_in(inst_onsets.get(inst), s.start, s.end)
            if len(seg) < 8:
                continue
            gaps = sorted(seg[i + 1] - seg[i] for i in range(len(seg) - 1))
            if gaps[len(gaps) // 2] <= tpb // 2:      # median spacing ≤ 1/8
                mid = (s.start + s.end) // 2
                out.append(CutEvent(_nearest_accent(mid, accents, tpb), "technical",
                                    [cut], PRIO["technical"], note=f"fast {inst} @{s.kind}"))
                break
    return out


def detect_events(sections: list[Section],
                  inst_onsets: dict[str, list[int]] | None,
                  accents: list[int] | None,
                  bre_spans: list[tuple[int, int]] | None,
                  time_sig_map: list, tpb: int) -> list[CutEvent]:
    """Full event timeline, sorted by tick (ties broken by priority desc)."""
    acc = sorted(accents) if accents else []
    ev: list[CutEvent] = []
    ev += detect_bre(bre_spans)
    ev += detect_rises(sections, acc, tpb)
    ev += detect_impacts(sections, acc, tpb)
    ev += detect_solos(sections, acc, tpb)
    ev += detect_downtime(inst_onsets, time_sig_map, tpb)
    ev += detect_vocal_peaks(inst_onsets, acc, time_sig_map, tpb)
    ev += detect_duos(sections, inst_onsets, acc, tpb)
    ev += detect_stagedive(sections, inst_onsets, acc, time_sig_map, tpb)
    ev += detect_technical(sections, inst_onsets, acc, tpb)
    ev.sort(key=lambda e: (e.tick, -e.priority))
    return ev

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

from pathlib import Path
import json

from dataclasses import dataclass, field

from .venue import (Section, section_energy, _camera_energy, _energy_tier_at,
                    find_pause_spans, measure_ticks_at)


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

# The leader rank is SONG-RELATIVE (cnt/own-total). That over-rewards a bass that
# is sparse over the whole song: in a riff the guitar and bass play near-identical
# absolute counts, but the bass's smaller denominator makes it "step up" more, so
# riffs were handed to the bassist. The guitar is the visual focal of a riff, so we
# give it a modest multiplier — enough to win riffs it co-plays with the bass while
# leaving genuine vocal/drum leads (which score far higher) untouched.
_GUITAR_LEAD_BIAS = 1.35


def _entry_tier(s: Section) -> str:
    """Energy AT the section's first instant (entry sub-span), not the section peak —
    so a chorus/breakdown that opens calm and only builds later is treated as the calm
    moment it actually is at the downbeat. Drives where dramatic directeds may land."""
    if s.energy_spans:
        return _energy_tier_at(s, s.start)
    return _camera_energy(s)


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
            # Land ON the climb, not the section downbeat: if the section opens calm and
            # only reaches 'high' later, anchor at that first high sub-span so the
            # band-wide kick hits the actual energy rise (not a still-calm intro).
            hi = s.start
            for a, _b, t in (s.energy_spans or []):
                if t == "high":
                    hi = a
                    break
            out.append(CutEvent(_nearest_accent(hi, accents, tpb), "rise",
                                ["D_All_Yeah", "D_All_LT", "D_All_Cam"], PRIO["rise"],
                                dramatic=True, note=f"rise→high @{s.kind}"))
        prev = e
    return out


def detect_impacts(sections: list[Section], accents: list[int], tpb: int) -> list[CutEvent]:
    """Entry of an impact section (intro/chorus/drop/breakdown/outro) → full-band pan.

    GATED ON REAL ENERGY, not the section name: section labels lie (e.g. Elegy tags a
    calm 4-bar lead-in as "Breakdown" and the actual climax as "Bridge" — the names are
    swapped). A dramatic full-band pan only reads as an "impact" if the section is
    genuinely energetic (mid/high); a calm intro/breakdown/outro is a quiet moment the
    framing bed should carry, not a place to throw a band-wide cut. The real high-energy
    entries still get their dramatic cut (here or via detect_rises)."""
    out = []
    for s in sections:
        # Gate on the ENTRY energy, not the section peak: a section that OPENS calm (the
        # high→calm drop into a quiet chorus start) must not get a band-wide pan whose
        # held frame then outlives the energy and lingers into the calm (user's note).
        if s.kind in _IMPACT_KINDS and _RANK[_entry_tier(s)] >= 1:
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


def detect_solos(sections: list[Section], inst_onsets: dict[str, list[int]] | None,
                 accents: list[int], tpb: int) -> list[CutEvent]:
    """Solo section → featured soloist: long pre-roll in, close-up, mid-solo flourish.

    The soloist is taken from the section NAME only when that instrument actually plays
    in the window; otherwise (unnamed solo, or a name that doesn't match the chart) it's
    derived from CONTENT — whoever leads the window song-relative — instead of blindly
    defaulting to guitar. Section labels can't be trusted (see detect_impacts)."""
    out = []
    table = {
        "guitar": ["D_Gtr_Cam_PR", "D_Gtr_CLS", "D_Gtr_Cam_PT"],
        "bass":   ["D_Bass_Cam", "D_Bass_CLS", "D_Bass"],
        "drums":  ["D_Drums_LT", "D_Drums_KD", "D_Drums"],
        "keys":   ["D_Keys_Cam", "D_Keys"],
        "vocal":  ["D_Vox_Cam_PR", "D_Vox_CLS", "D_Vox_Cam_PT"],
    }
    totals = ({k: len(v) for k, v in inst_onsets.items()
               if v and k in ("guitar", "bass", "keys", "drums", "vocal")}
              if inst_onsets else {})
    for s in sections:
        if s.kind != "solo":
            continue
        n = s.name.lower()
        named = next((k for k in ("bass", "drums", "keys", "vocal", "guitar")
                      if k[:4] in n or k in n), None)
        inst = None
        # Trust the name only if that instrument actually plays here.
        if named and _onsets_in(inst_onsets.get(named) if inst_onsets else None,
                                s.start, s.end):
            inst = named
        else:                                          # derive the soloist from content
            leaders = (_section_leaders(inst_onsets, s.start, s.end, totals)
                       if inst_onsets else [])
            inst = leaders[0][0] if leaders else (named or "guitar")
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


# Single-character close-up per instrument (the feature shot for whoever leads).
_FEATURE_CLOSEUP = {
    "guitar": ["D_Gtr_CLS", "D_Gtr"], "bass": ["D_Bass_CLS", "D_Bass"],
    "keys": ["D_Keys_Cam", "D_Keys"], "drums": ["D_Drums_LT", "D_Drums"],
    "vocal": ["D_Vox_CLS", "D_Vox_Cam_PT"],
}
# Interaction shot for two members that CO-lead a section (frozenset of the pair → cut).
_FEATURE_DUO = {
    frozenset({"guitar", "bass"}): "D_Duo_GB",
    frozenset({"keys", "bass"}): "D_Duo_KB",
    frozenset({"keys", "guitar"}): "D_Duo_KG",
    frozenset({"keys", "vocal"}): "D_Duo_KV",
    frozenset({"guitar", "vocal"}): "D_Duo_Gtr",
    frozenset({"bass", "vocal"}): "D_Duo_Bass",
    # NOTE: D_Duo_Drums (drummer-vocalist) is intentionally omitted — it asserts the
    # drummer is the singer, which is almost never the case; a drums+vocal section
    # falls to a vocal close-up instead.
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
            score *= _GUITAR_LEAD_BIAS      # see _GUITAR_LEAD_BIAS: surface riffs
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


def detect_features(sections: list[Section], inst_onsets: dict[str, list[int]] | None,
                    accents: list[int], tpb: int) -> list[CutEvent]:
    """One FEATURE shot per mid/high section that FOLLOWS THE MUSIC: it films whoever
    actually LEADS the section (song-relative density), as an interaction (duo) when two
    members co-lead, otherwise a single-character close-up. The hit lands on a real onset
    of the lead instrument. Candidates are ordered most→least specific so build_camera's
    anti-recency still gives variety — but every option is grounded in what's playing,
    instead of the old blind menu rotation that dropped vox/gtr close-ups on the wrong
    instrument."""
    out = []
    if not inst_onsets:
        return out
    totals = {k: len(v) for k, v in inst_onsets.items()
              if v and k in ("guitar", "bass", "keys", "drums", "vocal")}
    real_vox = bool(inst_onsets.get("_vocal_real"))
    for s in sections:
        if _RANK[_camera_energy(s)] < 1:          # only mid/high
            continue
        mid = (s.start + s.end) // 2
        leaders = _section_leaders(inst_onsets, s.start, s.end, totals)
        if not leaders:
            continue
        lead, lead_score = leaders[0]
        # Candidate order = most→least specific: co-lead duo, then lead close-up, then
        # the second member's close-up. build_camera's anti-recency picks among these,
        # so variety emerges while every option stays grounded in who's actually playing.
        cuts: list[str] = []
        # Co-lead duo: only when a second member is genuinely strong here (>=60% of the
        # lead's density) and forms a real pair — that's a true interaction, not a guess.
        second = None
        if len(leaders) >= 2 and leaders[1][1] >= lead_score * 0.6:
            second = leaders[1][0]
            duo = _FEATURE_DUO.get(frozenset({lead, second}))
            # Vocal duos need REAL vocals (lyrics) for the shot to read as interaction.
            if duo and ("vocal" not in (lead, second) or real_vox):
                cuts.append(duo)
        cuts += _FEATURE_CLOSEUP.get(lead, [])
        if second:
            cuts += _FEATURE_CLOSEUP.get(second, [])
        # Anchor where the LEAD is most active (busiest local onset), not the geometric
        # midpoint — so the feature lands on the moment the player is actually going off,
        # never in a mid-section lull.
        lead_ons = inst_onsets.get(lead) or []
        tick = _busiest_onset(lead_ons, s.start, s.end, tpb, mid) if lead_ons else mid
        out.append(CutEvent(_nearest_accent(tick, accents, tpb), "feature", cuts,
                            PRIO["duo"], note=f"feature {lead}@{s.kind}"))
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
    ev += detect_solos(sections, inst_onsets, acc, tpb)
    ev += detect_downtime(inst_onsets, time_sig_map, tpb)
    ev += detect_vocal_peaks(inst_onsets, acc, time_sig_map, tpb)
    ev += detect_features(sections, inst_onsets, acc, tpb)
    ev += detect_stagedive(sections, inst_onsets, acc, time_sig_map, tpb)
    # detect_technical is defined but not wired yet (Phase 2+: precise fast-run anchor).
    ev.sort(key=lambda e: (e.tick, -e.priority))
    return ev


# ── Density constants ─────────────────────────────────────────────────────────

_DEFAULT_CATEGORY_PCT = {
    "directed_vocals_cam_pt": 10.3,
    "directed_drums_lt": 8.13,
    "directed_guitar_cls": 7.76,
    "directed_vocals_cls": 4.47,
    "directed_all_cam": 4.33,
    "directed_bass_cls": 4.1,
    "directed_duo_gb": 3.93,
    "directed_duo_guitar": 3.77,
    "directed_duo_bass": 3.77,
    "directed_duo_kb": 3.6,
    "directed_crowd": 3.57,
    "directed_vocals": 3.5,
    "directed_vocals_cam_pr": 3.2,
    "directed_all_lt": 3.17,
    "directed_all": 3.1,
    "directed_all_yeah": 2.97,
    "directed_duo_kg": 2.93,
    "directed_drums_kd": 2.93,
    "directed_guitar_cam_pt": 2.8,
    "directed_bass": 2.3,
    "directed_duo_kv": 1.97,
    "directed_drums_pnt": 1.87,
    "directed_bass_cam": 1.4,
    "directed_guitar": 1.37,
    "directed_keys": 1.27,
    "directed_duo_drums": 1.13,
    "directed_keys_cam": 1.1,
    "directed_vocals_np": 1.07,
    "directed_bass_np": 0.97,
    "directed_brej": 0.8,
    "directed_crowd_b": 0.8,
    "directed_guitar_cam_pr": 0.57,
    "directed_guitar_np": 0.4,
    "directed_crowd_g": 0.3,
    "directed_drums": 0.17,
    "directed_keys_np": 0.1,
    "directed_drums_np": 0.07,
    "directed_stagedive": 0.07,
}

_DEFAULT_EVENTS_PER_BEAT = {
    "verse": {"low": 0.0, "mid": 0.026, "high": 0.039},
    "chorus": {"mid": 0.040, "high": 0.070},
    "bridge": {"high": 0.058, "mid": 0.037, "low": 0.0},
    "solo": {"mid": 0.064, "high": 0.074, "very_high": 0.150},
    "intro": {"low": 0.042, "mid": 0.057, "high": 0.052, "very_high": 0.103},
    "outro": {"low": 0.018, "high": 0.070, "mid": 0.058},
    "breakdown": {"high": 0.050, "mid": 0.043, "low": 0.0},
    "prechorus": {"mid": 0.027, "high": 0.042},
    "postchorus": {"mid": 0.047, "high": 0.085},
    "build": {"high": 0.065, "mid": 0.038, "very_high": 0.0},
    "riff": {"high": 0.055, "mid": 0.020},
    "default": {"low": 0.032, "mid": 0.043, "high": 0.052, "very_high": 0.080},
}

_DEFAULT_BEAT_OFFSET_DIST = {
    "verse": {"p5": 0.0, "p25": 0.33, "p50": 0.52, "p75": 0.75, "p95": 1.0},
    "chorus": {"p5": 0.0, "p25": 0.01, "p50": 0.42, "p75": 0.84, "p95": 1.0},
    "bridge": {"p5": 0.0, "p25": 0.10, "p50": 0.39, "p75": 0.64, "p95": 1.0},
    "solo": {"p5": 0.0, "p25": 0.04, "p50": 0.37, "p75": 0.75, "p95": 1.0},
    "intro": {"p5": 0.0, "p25": 0.13, "p50": 0.48, "p75": 0.73, "p95": 1.0},
    "outro": {"p5": 0.0, "p25": 0.0, "p50": 0.39, "p75": 0.66, "p95": 1.0},
    "breakdown": {"p5": 0.0, "p25": 0.13, "p50": 0.48, "p75": 0.71, "p95": 1.0},
    "default": {"p5": 0.0, "p25": 0.17, "p50": 0.47, "p75": 0.74, "p95": 1.0},
}

_DEFAULT_MARKOV = {
    "directed_vocals_cam_pt": {"directed_vocals_cam_pt": 0.422, "directed_drums_lt": 0.063, "directed_guitar_cls": 0.056, "directed_duo_guitar": 0.053, "directed_guitar_cam_pt": 0.040},
    "directed_drums_lt": {"directed_drums_lt": 0.494, "directed_vocals_cam_pt": 0.071, "directed_guitar_cls": 0.054, "directed_duo_guitar": 0.046, "directed_duo_kb": 0.029},
    "directed_guitar_cls": {"directed_guitar_cls": 0.276, "directed_bass_cls": 0.241, "directed_crowd": 0.075, "directed_all_lt": 0.057, "directed_drums_lt": 0.044},
    "directed_duo_guitar": {"directed_duo_bass": 0.283, "directed_vocals_cam_pt": 0.106, "directed_duo_guitar": 0.097, "directed_duo_gb": 0.062, "directed_duo_kb": 0.053},
    "directed_duo_bass": {"directed_duo_kv": 0.177, "directed_vocals_cam_pt": 0.133, "directed_duo_guitar": 0.124, "directed_vocals_cls": 0.080, "directed_all_cam": 0.053},
}


class BeatDensityGenerator:
    """Gera eventos directed baseados em densidade de notas e estatisticas aprendidas.

    Em vez de detetar momentos musicais especificos, distribui eventos directed
    proporcionalmente a densidade de notas dentro da seccao, replicando o padrao
    observado nos dados oficiais.
    """

    def __init__(self, stats_path: str | Path | None = None):
        import json
        if stats_path is None:
            candidate = Path(__file__).parent.parent / "dev" / "camera_stats.json"
            if candidate.exists():
                stats_path = candidate
        if stats_path and Path(stats_path).exists():
            with open(stats_path, encoding="utf-8") as f:
                data = json.load(f)
            self.events_per_beat = data.get("events_per_beat_by_section", _DEFAULT_EVENTS_PER_BEAT)
            self.beat_offsets = data.get("beat_offset_distribution", _DEFAULT_BEAT_OFFSET_DIST)
            self.markov_table = data.get("directed_markov_table", _DEFAULT_MARKOV)
            self.category_pct = data.get("category_distribution_pct", _DEFAULT_CATEGORY_PCT)
        else:
            self.events_per_beat = dict(_DEFAULT_EVENTS_PER_BEAT)
            self.beat_offsets = dict(_DEFAULT_BEAT_OFFSET_DIST)
            self.markov_table = dict(_DEFAULT_MARKOV)
            self.category_pct = dict(_DEFAULT_CATEGORY_PCT)

        # Build reverse lookup: official_directed_name -> internal D_xxx name
        from .venue import DIRECTED_CUTS
        self._official_to_internal_map = {v: k for k, v in DIRECTED_CUTS.items()}

    def _density_bucket(self, notes_per_beat: float) -> str:
        if notes_per_beat < 2:
            return "low"
        elif notes_per_beat < 6:
            return "mid"
        elif notes_per_beat < 15:
            return "high"
        else:
            return "very_high"

    def _sample_from_category(self) -> str:
        import random
        r = random.random() * 100.0
        cumulative = 0.0
        for cut_type, pct in self.category_pct.items():
            cumulative += pct
            if r <= cumulative:
                return cut_type
        return list(self.category_pct.keys())[0] if self.category_pct else "directed_vocals_cam_pt"

    def _pick_cut_type(self, prev_cut_type: str | None) -> str:
        import random
        if prev_cut_type is None or random.random() < 0.10:
            return self._sample_from_category()
        transitions = self.markov_table.get(prev_cut_type, {})
        if transitions:
            r = random.random()
            cumulative = 0.0
            for cut_type, prob in transitions.items():
                cumulative += prob
                if r <= cumulative:
                    return cut_type
        return self._sample_from_category()

    def _sample_positions(self, n_events: int, sec_start: int, sec_end: int,
                          offset_dist: dict, tpb: int) -> list[int]:
        import random
        if n_events <= 0:
            return []
        sec_duration = sec_end - sec_start
        if sec_duration <= 0:
            return []
        p5 = offset_dist.get("p5", 0.0)
        p25 = offset_dist.get("p25", 0.17)
        p50 = offset_dist.get("p50", 0.47)
        p75 = offset_dist.get("p75", 0.74)
        p95 = offset_dist.get("p95", 0.95)
        min_gap = tpb * 2

        candidates = []
        for _ in range(n_events * 5):
            r = random.random()
            if r < 0.05:
                frac = random.uniform(0.01, p5) if p5 > 0.01 else 0.01
            elif r < 0.25:
                frac = random.uniform(p5, p25)
            elif r < 0.50:
                frac = random.uniform(p25, p50)
            elif r < 0.75:
                frac = random.uniform(p50, p75)
            elif r < 0.95:
                frac = random.uniform(p75, p95)
            else:
                frac = random.uniform(p95, 0.99)
            frac = max(0.01, min(0.99, frac))
            tick = sec_start + int(frac * sec_duration)
            candidates.append(tick)
        candidates.sort()

        positions = []
        for c in candidates:
            if not positions or c - positions[-1] >= min_gap:
                positions.append(c)
            if len(positions) >= n_events:
                break
        return positions

    def generate(self, sections: list[Section], inst_onsets: dict[str, list[int]] | None,
                 time_sig_map: list, tpb: int,
                 accents: list[int] | None = None) -> list[CutEvent]:
        import bisect
        out = []
        if not inst_onsets or not sections:
            return out
        prev_cut_type = None
        last_tick = -10 ** 9
        acc = sorted(accents) if accents else []
        for sec in sections:
            sec_beats = (sec.end - sec.start) / tpb
            if sec_beats < 2:
                continue
            n_notes = 0
            for inst, ons in inst_onsets.items():
                if inst.startswith("_") or not ons:
                    continue
                lo = bisect.bisect_left(ons, sec.start)
                hi = bisect.bisect_left(ons, sec.end)
                n_notes += hi - lo
            notes_per_beat = n_notes / max(sec_beats, 0.001)
            d_bucket = self._density_bucket(notes_per_beat)
            kind_epb = self.events_per_beat.get(sec.kind, self.events_per_beat.get("default", {}))
            epb = kind_epb.get(d_bucket, kind_epb.get("mid", 0.04))
            n_events = max(0, int(round(epb * sec_beats)))
            if n_events == 0:
                continue
            offset_dist = self.beat_offsets.get(sec.kind, self.beat_offsets.get("default", {}))
            positions = self._sample_positions(n_events, sec.start, sec.end, offset_dist, tpb)
            for tick in positions:
                # Snap to nearest accent (within 2 beats) or beat
                snapped = tick
                if acc:
                    i = bisect.bisect_left(acc, tick)
                    for ci in (i - 1, i):
                        if 0 <= ci < len(acc) and abs(acc[ci] - tick) <= tpb * 2:
                            if abs(acc[ci] - tick) < abs(snapped - tick):
                                snapped = acc[ci]
                else:
                    # Snap to nearest beat boundary
                    snapped = round(tick / tpb) * tpb
                tick = snapped
                if tick - last_tick < tpb * 2:
                    continue
                cut_type = self._pick_cut_type(prev_cut_type)
                internal_cut = self._official_to_internal_map.get(cut_type)
                if internal_cut is None:
                    continue
                out.append(CutEvent(
                    tick=tick,
                    etype="density",
                    cuts=[internal_cut],
                    priority=25,
                    dramatic=False,
                    note=f"density {sec.kind}/{d_bucket} ({n_notes}n/{sec_beats:.0f}b)"
                ))
                prev_cut_type = cut_type
                last_tick = tick
        return out

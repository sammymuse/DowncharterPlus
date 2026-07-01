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
                    find_pause_spans, measure_ticks_at, _solo_instrument)


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
    "bre": 100, "stagedive": 90, "rise": 80, "solo": 70, "energy_rise": 65,
    "impact": 60, "vocal_peak": 50, "duo": 45, "downtime": 40, "baseline": 30,
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


def detect_energy_transitions(sections: list[Section],
                              energy_env: list[tuple[int, str]] | None,
                              accents: list[int], tpb: int) -> list[CutEvent]:
    """Sub-section energy transitions from audio -> full-band impact cuts.

    The audio energy envelope provides sub-section granularity (calm->mid->high).
    When energy jumps UP significantly within a section (calm->high or mid->high),
    generates a full-band impact cut (D_All_LT, D_All_Cam, D_All_Yeah).

    This catches energy builds that section-level detection misses - e.g., a
    chorus that opens calm and only reaches high energy mid-way through.

    Priority: 65 (between impact=60 and solo=70, since these are genuine audio
    moments that should override structural cuts).
    """
    out = []
    if not energy_env or len(energy_env) < 3:
        return out
    ev = sorted(energy_env)  # [(tick, tier), ...]
    _RANK_ENERGY = {"calm": 0, "mid": 1, "high": 2}
    last_tick = -10 ** 9
    min_gap = tpb * 8  # at most 1 per 8 beats

    for i in range(len(ev) - 1):
        cur_tick, cur_tier = ev[i]
        next_tick, next_tier = ev[i + 1]
        # Only upward jumps: calm->high or mid->high
        if _RANK_ENERGY.get(next_tier, 0) - _RANK_ENERGY.get(cur_tier, 0) < 2:
            continue
        # Position: at the transition point (next_tick) snapped to nearest accent
        tick = _nearest_accent(next_tick, accents, tpb)
        if tick - last_tick < min_gap:
            continue
        out.append(CutEvent(tick, "energy_rise",
                            ["D_All_LT", "D_All_Cam", "D_All_Yeah"],
                            65, dramatic=True,
                            note=f"energy {cur_tier}->{next_tier}"))
        last_tick = tick
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


def detect_duos(sections: list[Section],
                inst_onsets: dict[str, list[int]] | None,
                accents: list[int], tpb: int
                ) -> tuple[list[CutEvent], set[int]]:
    """Co-lead section → duo shot showing both musicians interacting.

    Finds the best duo pair by checking ALL instrument pairs, not just the
    top 2 leaders. This catches vocal+keys duos even when guitar overshadows
    vocal in onset count. Official data: ~19% of directed cuts are duos.

    Returns (events, covered_section_starts) so detect_baseline_cuts can
    skip sections already handled by a duo."""
    out: list[CutEvent] = []
    covered: set[int] = set()
    if not inst_onsets:
        return out, covered
    totals = {k: len(v) for k, v in inst_onsets.items()
              if v and k in ("guitar", "bass", "keys", "drums", "vocal")}
    real_vox = bool(inst_onsets.get("_vocal_real"))
    for s in sections:
        if _RANK[_camera_energy(s)] < 1:          # only mid/high
            continue
        if s.end - s.start < tpb * 4:
            continue
        leaders = _section_leaders(inst_onsets, s.start, s.end, totals)
        if len(leaders) < 2:
            continue
        lead, lead_score = leaders[0]
        if lead_score < 0.02:
            continue
        # Find the BEST duo pair: scan all pairs with score ≥ 40% of lead
        best_duo: tuple[str, str, str] | None = None  # (a, b, D_xxx)
        best_combined = 0.0
        threshold = lead_score * 0.40
        for i, (a, sa) in enumerate(leaders):
            if sa < threshold:
                continue
            for b, sb in leaders[i + 1:]:
                if sb < threshold:
                    continue
                duo = _FEATURE_DUO.get(frozenset({a, b}))
                if duo is None:
                    continue
                if "vocal" in (a, b) and not real_vox:
                    continue
                combined = sa + sb
                if combined > best_combined:
                    best_combined = combined
                    best_duo = (a, b, duo)
        if best_duo is None:
            continue
        a, b, duo = best_duo
        mid = (s.start + s.end) // 2
        # Place on the busier instrument's onset
        ons = inst_onsets.get(a) or []
        tick = (_busiest_onset(ons, s.start, s.end, tpb, mid)
                if ons else mid)
        tick = _nearest_accent(tick, accents, tpb)
        out.append(CutEvent(tick, "duo", [duo], PRIO["duo"],
                            note=f"duo {a}+{b} @{s.kind}"))
        covered.add(s.start)
    return out, covered


def detect_vocal_peaks(inst_onsets: dict[str, list[int]] | None,
                       accents: list[int],
                       time_sig_map: list, tpb: int) -> list[CutEvent]:
    """Last note of a vocal phrase (a sustained/peak note) → vocal close-up.

    Groups vocal onsets into phrases (gap ≥ 1 measure splits phrases),
    then fires a D_Vox_CLS on the last onset of each phrase of ≥2 notes,
    throttled to ≤1 per 4 beats. Official data: vocals are the #1 category
    for first events in a section (29%)."""
    out = []
    if not inst_onsets:
        return out
    vocal = inst_onsets.get("_vocal_real") or []
    if len(vocal) < 5:
        return out
    last_emit = -10 ** 9
    for ps, last, n in _phrases(vocal, time_sig_map, tpb):
        if n < 3:                                  # require substantial phrase
            continue
        if last - last_emit < tpb * 8:              # throttle: ≤1 per 8 beats
            continue
        out.append(CutEvent(_nearest_accent(last, accents, tpb), "vocal_peak",
                            ["D_Vox_CLS", "D_Vox_Cam_PT"], PRIO["vocal_peak"],
                            note="vocal phrase peak"))
        last_emit = last
    return out


def detect_events(sections: list[Section],
                  inst_onsets: dict[str, list[int]] | None,
                  accents: list[int] | None,
                  bre_spans: list[tuple[int, int]] | None,
                  time_sig_map: list, tpb: int,
                  audio_onsets: list[int] | None = None,
                  energy_env: list[tuple[int, str]] | None = None,
                  band_activity: dict[str, list[int]] | None = None) -> list[CutEvent]:
    """Full event timeline. Sorted by tick (ties broken by priority desc)."""
    acc = sorted(accents) if accents else []
    # Merge MIDI + audio accents for richer snap
    if audio_onsets:
        audio = sorted(audio_onsets)
        merged = []
        i = j = 0
        while i < len(acc) and j < len(audio):
            if acc[i] < audio[j]:
                merged.append(acc[i]); i += 1
            elif audio[j] < acc[i]:
                merged.append(audio[j]); j += 1
            else:
                merged.append(acc[i]); i += 1; j += 1
        merged.extend(acc[i:]); merged.extend(audio[j:])
        acc = merged
    ev: list[CutEvent] = []
    ev += detect_bre(bre_spans)
    ev += detect_stagedive(sections, inst_onsets, acc, time_sig_map, tpb)
    ev += detect_downtime(inst_onsets, time_sig_map, tpb)
    ev += detect_vocal_peaks(inst_onsets, acc, time_sig_map, tpb)
    ev += detect_rises(sections, acc, tpb)
    ev += detect_impacts(sections, acc, tpb)
    ev += detect_energy_transitions(sections, energy_env, acc, tpb)
    ev += detect_solos(sections, inst_onsets, acc, tpb)
    # detect_duos runs BEFORE baseline; baseline skips sections where duo fired
    # to avoid double-counting (a section gets either a duo or a closeup, not both).
    duo_events, duo_covered = detect_duos(sections, inst_onsets, acc, tpb)
    ev += duo_events
    ev += detect_baseline_cuts(sections, inst_onsets, acc, time_sig_map, tpb,
                                skip_sections=duo_covered,
                                audio_onsets=audio_onsets,
                                band_activity=band_activity)
    ev.sort(key=lambda e: (e.tick, -e.priority))
    return ev


def detect_baseline_cuts(sections: list[Section],
                         inst_onsets: dict[str, list[int]] | None,
                         accents: list[int] | None,
                         time_sig_map: list, tpb: int,
                         skip_sections: set[int] | None = None,
                         audio_onsets: list[int] | None = None,
                         band_activity: dict[str, list[int]] | None = None
                         ) -> list[CutEvent]:
    """One structural directed cut per qualifying section (mid/high energy).

    Skips sections whose .start is in skip_sections (already covered by
    a higher-priority event like a duo).

    3 categories:
    1. SOLO → instrument close-up (D_Gtr_CLS, D_Bass_CLS, D_Keys_Cam)
    2. DROP/BUILD → full-band cut (D_All_LT, D_All_Cam, D_Drums_LT)
    3. All other mid/high sections → vocal cut (D_Vox_Cam_PT, D_Vocals)

    Placed at the busiest onset of the lead instrument in the section,
    or midsection if no onsets. Priority 30 (below special events).
    """
    out = []
    if not inst_onsets or not sections:
        return out
    skip = skip_sections or set()
    totals = {k: len(v) for k, v in inst_onsets.items()
              if v and k in ("guitar", "bass", "keys", "drums", "vocal")}
    acc = sorted(accents) if accents else []
    # Merge MIDI + audio accents for richer snap
    if audio_onsets:
        audio = sorted(audio_onsets)
        merged = []
        i = j = 0
        while i < len(acc) and j < len(audio):
            if acc[i] < audio[j]:
                merged.append(acc[i]); i += 1
            elif audio[j] < acc[i]:
                merged.append(audio[j]); j += 1
            else:
                merged.append(acc[i]); i += 1; j += 1
        merged.extend(acc[i:]); merged.extend(audio[j:])
        acc = merged

    # Pre-compute audio band activity lookup for identity refinement
    # band_activity = {"bass": [tick, ...], "drums": [...], "lead": [...]}
    # 'lead' ≈ guitar/vocals (300-3000Hz)
    def _active_band_at(tick: int) -> str | None:
        """Which frequency band is most active near `tick` (±1 beat)."""
        if not band_activity:
            return None
        import bisect
        best_band, best_cnt = None, 0
        for band, ticks in band_activity.items():
            if not ticks:
                continue
            i = bisect.bisect_left(ticks, tick)
            cnt = 0
            for ci in (i - 1, i):
                if 0 <= ci < len(ticks) and abs(ticks[ci] - tick) <= tpb:
                    cnt += 1
            if cnt > best_cnt:
                best_band, best_cnt = band, cnt
        return best_band

    for s in sections:
        if _RANK[_camera_energy(s)] < 1:  # skip calm sections
            continue
        if s.start in skip:               # already covered by a duo event
            continue
        if s.end - s.start < tpb * 4:
            continue
        leaders = _section_leaders(inst_onsets, s.start, s.end, totals)
        if not leaders:
            continue
        lead, _ = leaders[0]
        mid = (s.start + s.end) // 2
        # Determine cut type by section kind
        if s.kind == "solo":
            cuts = _FEATURE_CLOSEUP.get(_solo_instrument(s.name), ["D_Gtr_CLS"])
        elif s.kind in ("drop", "build"):
            cuts = ["D_All_LT", "D_All_Cam", "D_Drums_LT"]
        else:
            # Most sections: film the leader
            cuts = _FEATURE_CLOSEUP.get(lead, ["D_Vox_Cam_PT"])
        # Position at busiest onset of lead
        lead_ons = inst_onsets.get(lead) or []
        tick = (_busiest_onset(lead_ons, s.start, s.end, tpb, mid)
                if lead_ons else mid)
        tick = _nearest_accent(tick, acc, tpb) if acc else tick
        # Audio band activity can ADD a secondary cut option
        audio_band = _active_band_at(tick)
        final_cuts = list(cuts)
        if audio_band is not None:
            # Map audio band to appropriate cut
            band_cut = {
                "bass": "D_Bass_CLS",
                "drums": "D_Drums_LT",
                "lead": "D_Vox_Cam_PT",  # lead ≈ vocal/guitar
            }.get(audio_band)
            if band_cut and band_cut not in final_cuts:
                final_cuts.append(band_cut)
        out.append(CutEvent(tick, "baseline", final_cuts, 30,
                            note=f"structural {lead}@{s.kind}" +
                            (f" audio={audio_band}" if audio_band else "")))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#
# Co-occurrence learned from 100 official songs: certain cut types fire
# together at the same tick (e.g. directed_drums_lt + directed_guitar_cls).
# The table below captures P(companion_B | primary_A) ≥ 0.25 (high confidence).

_COMPANION_PAIRS: list[tuple[str, str, float]] = [
    # primary_official      companion_official       prob
    ("directed_duo_guitar", "directed_duo_bass",     0.51),
    ("directed_duo_bass",   "directed_duo_kv",       0.51),
    ("directed_duo_bass",   "directed_duo_guitar",   0.43),
    ("directed_keys",       "directed_bass",         0.54),
    ("directed_guitar",     "directed_bass",         0.65),
    ("directed_guitar_cls", "directed_bass_cls",     0.44),
    ("directed_guitar_cls", "directed_crowd",        0.16),  # < 0.25 but important for crowd
    ("directed_guitar_cls", "directed_all_lt",       0.11),
    ("directed_drums_lt",   "directed_guitar_cls",   0.45),
    ("directed_drums_lt",   "directed_bass_cls",     0.20),
    ("directed_drums_lt",   "directed_guitar_cam_pt", 0.15),
    ("directed_drums_lt",   "directed_bass_cam",     0.10),
    ("directed_duo_kv",     "directed_duo_bass",     0.81),
    ("directed_duo_kb",     "directed_duo_gb",       0.42),
    ("directed_duo_kb",     "directed_duo_kg",       0.42),
    ("directed_duo_kg",     "directed_duo_gb",       0.46),
    ("directed_duo_kg",     "directed_duo_kb",       0.46),
    ("directed_duo_gb",     "directed_duo_kb",       0.47),
    ("directed_duo_gb",     "directed_duo_kg",       0.46),
    ("directed_all_cam",    "directed_all_lt",       0.25),
    ("directed_bass_cls",   "directed_guitar_cls",   0.35),
    ("directed_bass_cam",   "directed_drums_lt",     0.30),
]

# Built once: official_name → D_xxx map
_COMPANION_OFF2INT: dict[str, str] | None = None


def _ensure_companion_map():
    """Build official → internal name map for companion rules."""
    global _COMPANION_OFF2INT
    if _COMPANION_OFF2INT is not None:
        return
    from .venue import DIRECTED_CUTS
    _COMPANION_OFF2INT = {v: k for k, v in DIRECTED_CUTS.items()}


def add_companion_shots(accepted: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Add best companion directed cut at the same tick as accepted primary cuts.

    Uses high-confidence co-occurrence pairs (P ≥ 0.30) learned from 100 songs.
    At most 1 companion per tick, matched to the highest probability pair.

    Returns list of (tick, D_xxx_cut) to merge into the accepted list.
    """
    _ensure_companion_map()

    companions: list[tuple[int, str]] = []
    seen_at: dict[int, set[str]] = {}
    # Build internal → (companion_int, prob) pairs, filtered by P ≥ 0.30
    pair_map: dict[str, list[tuple[str, float]]] = {}
    for p_official, c_official, prob in _COMPANION_PAIRS:
        if prob < 0.30:
            continue
        p_int = _COMPANION_OFF2INT.get(p_official) if _COMPANION_OFF2INT else None
        c_int = _COMPANION_OFF2INT.get(c_official) if _COMPANION_OFF2INT else None
        if not p_int or not c_int:
            continue
        pair_map.setdefault(p_int, []).append((c_int, prob))

    for tick, cut in accepted:
        candidates = pair_map.get(cut)
        if not candidates:
            continue
        # Pick the best companion NOT already seen at this tick
        best: tuple[str, float] | None = None
        for c_int, prob in candidates:
            if tick in seen_at and c_int in seen_at[tick]:
                continue
            if c_int == cut:
                continue
            if best is None or prob > best[1]:
                best = (c_int, prob)
        if best is not None:
            c_int, _prob = best
            if tick not in seen_at:
                seen_at[tick] = set()
            companions.append((tick, c_int))
            seen_at[tick].add(c_int)

    return companions

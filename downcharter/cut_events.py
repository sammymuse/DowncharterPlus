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


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE GENERATOR (Layer 1)
# ══════════════════════════════════════════════════════════════════════════════
#
# Most official directed cuts (57%) are >4 beats from any section boundary,
# placed at a statistical MIDPOINT (mean offset = 0.458). They are NOT "special
# moments" — they are COVERAGE SHOTS following an instrument hierarchy per
# section kind, with ~42% self-repetition rate.
#
# This generator replicates the learned patterns: it distributes directed cuts
# at the right positions, at the right density, with the right instrument focus,
# at low priority (20) so special events (BRE, rises, solos) still override.

_BASELINE_EVENTS_PER_BEAT: dict[str, float] = {
    "chorus": 0.055, "verse": 0.035, "solo": 0.10,
    "intro": 0.050, "outro": 0.035, "breakdown": 0.065,
    "build": 0.060, "bridge": 0.040, "riff": 0.055,
    "prechorus": 0.035, "postchorus": 0.055, "drop": 0.060,
    "default": 0.040,
}

# Mean offset within section to place each baseline event (learned: 0.458).
_BASELINE_OFFSET_MEAN = 0.458
_BASELINE_OFFSET_STD = 0.12   # approximate from p25-p75 spread

# Self-repeat probability (Markov self-transition ~42% in official data).
_SELF_REPEAT_PROB = 0.42

# Instrument-targeted cut list per section kind (top = most common for that section).
# The guard (_guard_directed) filters cuts that don't make sense (absent instrument,
# idle character, no real vocals for crowd, etc.) — so we provide generous options.
_BASELINE_CUTS_BY_KIND: dict[str, list[str]] = {
    "verse":      ["D_Vox_Cam_PT", "D_Vocals", "D_Vox_CLS", "D_Gtr_CLS", "D_Bass_CLS"],
    "chorus":     ["D_Vox_Cam_PT", "D_All_Cam", "D_All", "D_Vocals", "D_Gtr"],
    "prechorus":  ["D_Vocals", "D_Vox_Cam_PT", "D_Gtr", "D_Bass_CLS"],
    "postchorus": ["D_Vox_Cam_PT", "D_All_Cam", "D_All", "D_Vocals"],
    "riff":       ["D_Gtr_CLS", "D_Gtr", "D_Bass_CLS", "D_Drums_LT"],
    "solo":       ["D_Gtr_CLS", "D_Bass_CLS", "D_Keys_Cam", "D_Vox_CLS"],
    "build":      ["D_Drums_LT", "D_Gtr_CLS", "D_All_LT", "D_Bass_CLS"],
    "drop":       ["D_All_LT", "D_Drums_LT", "D_All", "D_All_Cam"],
    "breakdown":  ["D_Gtr_CLS", "D_Drums_LT", "D_Bass_CLS", "D_All_LT"],
    "bridge":     ["D_Vox_CLS", "D_Duo_KV", "D_Keys", "D_Vox_Cam_PT"],
    "intro":      ["D_All_Cam", "D_All", "D_Gtr", "D_Vocals"],
    "outro":      ["D_All", "D_Vox_CLS", "D_All_Yeah", "D_Vox_Cam_PT"],
    "default":    ["D_Vox_Cam_PT", "D_Vocals", "D_Gtr_CLS", "D_All_Cam"],
}


def detect_baseline_cuts(sections: list[Section],
                         inst_onsets: dict[str, list[int]] | None,
                         accents: list[int] | None,
                         time_sig_map: list, tpb: int,
                         events_per_beat: dict[str, float] | None = None
                         ) -> list[CutEvent]:
    """Distribute directed cuts at the statistical pattern learned from 100 songs.

    For each section, computes target event count (events_per_beat × length),
    places them at ∼offset=0.458 (midsection), snaps to nearest accent, and
    selects cut type from the instrument hierarchy per section kind (with
    ~42% self-repeat probability).

    Priority 30 — below special events (40-100) but above nothing.
    """
    import random, bisect
    out: list[CutEvent] = []
    if not inst_onsets or not sections:
        return out
    eb = events_per_beat or _BASELINE_EVENTS_PER_BEAT
    acc = sorted(accents) if accents else []
    prev_cut: str | None = None
    last_tick = -10 ** 9
    min_gap = tpb * 2  # minimum 2 beats between baseline events

    for sec in sections:
        sec_beats = (sec.end - sec.start) / tpb
        if sec_beats < 2:
            continue
        # Target event count
        epb = eb.get(sec.kind, eb.get("default", 0.04))
        n_events = max(1, int(round(epb * sec_beats)))
        # Place events at ~midsection offset, spread evenly
        positions_placed = 0
        for _ in range(n_events * 3):  # oversample to fill min_gap
            if positions_placed >= n_events:
                break
            # Gaussian-ish jitter around mean offset 0.458
            frac = _BASELINE_OFFSET_MEAN + random.gauss(0, _BASELINE_OFFSET_STD)
            frac = max(0.05, min(0.95, frac))
            tick = sec.start + int(frac * (sec.end - sec.start))
            # Gentle snap to nearest accent (within 1 beat) — keeps midsection position
            if acc:
                i = bisect.bisect_left(acc, tick)
                for ci in (i - 1, i):
                    if 0 <= ci < len(acc) and abs(acc[ci] - tick) <= tpb:
                        tick = acc[ci]
                        break
            # Minimum gap from previous event
            if tick - last_tick < min_gap:
                continue
            # Pick cut type: self-repeat or from hierarchy
            cut = _pick_baseline_cut(prev_cut, sec.kind)
            if cut is None:
                continue
            out.append(CutEvent(
                tick=tick, etype="baseline", cuts=[cut],
                priority=30, dramatic=False,
                note=f"baseline {sec.kind} ({positions_placed + 1}/{n_events})"
            ))
            prev_cut = cut
            last_tick = tick
            positions_placed += 1

    return out


def _pick_baseline_cut(prev: str | None, kind: str) -> str | None:
    """Pick a directed cut for a baseline event.

    - 42% chance: repeat previous cut (self-repeat, matching official data)
    - 58% chance: cycle through the section's hierarchy (variety)
    """
    import random
    pool = _BASELINE_CUTS_BY_KIND.get(kind, _BASELINE_CUTS_BY_KIND["default"])
    if not pool:
        return None
    # Self-repeat
    if prev is not None and random.random() < _SELF_REPEAT_PROB:
        return prev
    # Pick from pool avoiding immediate repeat + cycling variety
    if prev is not None and prev in pool:
        # Pick a DIFFERENT cut from the pool (no-immediate-repeat but allow later repeat)
        others = [c for c in pool if c != prev]
        return random.choice(others) if others else pool[0]
    return random.choice(pool)


# ══════════════════════════════════════════════════════════════════════════════
#  (END of Baseline Generator)
# ══════════════════════════════════════════════════════════════════════════════
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

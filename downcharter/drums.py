"""
drums.py — Drum reduction guided by the Rock Band Customs Book (pp.182–186).

Rewritten from scratch (replaces `processor._reduce_drums` and its patches). One
rule per line of the book, easy to audit against the PDF. Principles agreed with
the user:

  • CASCADE — Hard←Expert, Medium←Hard, Easy←Medium (the book's method), never each
    level straight from Expert.
  • Thresholds in playability MS, derived from the book's BPMs (`ms = 60000/BPM/sub`),
    documented inline with the target BPM. Never hard-coded BPM.
  • The BOOK rules; `drum_metrics.py` (pos/padok/count) is only a sanity check.
  • COLOR is irrelevant in drums → no color unification / re-voicing. Y/B/G are
    treated as "hand pads" (cymbal/tom); only KICK / SNARE / PAD are distinguished.

Importance weights (what drops first on collision): snare/crash > kick > tom.

Architecture: convert the Expert events (already post-`_apply_expert_plus`) ONCE
into a gem model, cascade over it, and emit each level back to mido events with the
right note offset (avoids re-pairing note_on/off per level).

    Gem      = (tick, note, dur, vel)         # note ∈ {96,97,98,99,100,101}
    chart    = sorted list of Gems
    markers  = AbsEvents passed intact to every level (tom 110-112, globals)

The `vel` (velocity) carries the DYNAMICS — ghost (low vel) / accent (high vel) — and
is preserved through the cascade and the emission, so the reduced difficulties keep
the Expert ghosts/accents (they are not re-velocitied to a fixed value).
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass

from .constants import (
    DIFF_OFFSET, DRUM_KICK_EXPERT, DRUM_KICK_2X, DRUM_EXPERT_PADS,
    GLOBAL_MARKERS, DRUM_TOM_MARKERS,
)
from .midi_utils import AbsEvent, tick_to_ms, measure_ticks_at, pair_notes

# ── Lane categories ───────────────────────────────────────────────────────────
# Color (which of Y/B/G) is preserved in the output but the LOGIC doesn't use it.
KICK_NOTE  = DRUM_KICK_EXPERT          # 96  (kick2x 95 collapses here)
SNARE_NOTE = DRUM_KICK_EXPERT + 1      # 97
PAD_NOTES  = set(DRUM_EXPERT_PADS) - {SNARE_NOTE}   # 98,99,100,101 (hands: cymbal/tom)

KICK, SNARE, PAD = "kick", "snare", "pad"


def category(note: int) -> str:
    if note == KICK_NOTE:
        return KICK
    if note == SNARE_NOTE:
        return SNARE
    return PAD


# ── Playability MS constants (derived from the book's BPMs) ───────────────────
#
#   ms_of_one_subdivision = 60000 / BPM / subdiv_per_quarter
#   eighth (subdiv=2):  60000 / BPM / 2
#
# EIGHTH_MIN_MS — minimum playable eighth for HAND timekeeping (book: "8th cap").
#   Above the target BPM, the hand switches to quarters. Hard ≈170BPM, Medium ≈140BPM,
#   Easy ≈105BPM → 60000/BPM/2.
EIGHTH_MIN_MS = {
    "hard":   60000 / 170 / 2,   # ≈176 ms  (book Hard: 8th cap at ~170BPM, the top of
                                 #           the book's Hard range. Measured vs the
                                 #           official Hard charts: at 150BPM we collapsed
                                 #           the 8th hi-hat groove of ~155-165BPM songs
                                 #           (outkast/deafheaven/openyourheart) to
                                 #           quarters, losing ~7pp of recall; 170BPM keeps
                                 #           it — recall 82.6→89.6%, precision 85.5→85.1,
                                 #           density 97→105% of official.)
    "medium": 60000 / 140 / 2,   # ≈214 ms  (book Medium: 8th up to ~140BPM)
    "easy":   60000 / 105 / 2,   # ≈286 ms  (book Easy: slower timekeeping)
}

# SIXTEENTH_MIN_MS — minimum sixteenth to ALLOW a 16th roll on Hard
#   (book Hard: 16th rolls only if the 16th is playable). ≈140BPM → 60000/BPM/4.
SIXTEENTH_MIN_MS = {
    "hard": 60000 / 140 / 4,     # ≈107 ms
}

# QUARTER_KICK_MS — above this QUARTER threshold the kick only sits on quarters
#   (book Medium/Easy: "above ~100-110BPM, kick only on quarters"). 110BPM →
#   60000/110 ≈545 ms per quarter.
QUARTER_KICK_MS = {
    "medium": 60000 / 110,       # ≈545 ms
    "easy":   60000 / 100,       # ≈600 ms
}

# KICK_COLLAPSE_MS — DOUBLE-BASH threshold: kicks closer to the previous structural
#   kick than this are double-bass (filler) notes and are dropped, PRESERVING the
#   original groove positions (downbeats + syncopations). There is never a snap to grid.
#   • Hard: 360ms (≈quarter at 167BPM / eighth at 83BPM) — tuned at the user's request
#     to reduce Hard more (≤~1 kick per beat); keeps positions.
#   • Medium/Easy: book's "kick only on quarters" → collapses sub-quarter, leaving at
#     most ~1 kick per beat, but always on the ORIGINAL position (not on the line).
KICK_COLLAPSE_MS = {
    "hard":   360.0,                       # ≈167BPM quarter (reduces Hard more)
    "medium": QUARTER_KICK_MS["medium"]*1.5, # ≈818 ms ≈ DOTTED QUARTER at 110BPM. The
                                           # earlier half-note collapse (×2) was too
                                           # aggressive vs the official Medium charts: it
                                           # dropped real structural kicks, leaving us at
                                           # only 76% of the official kick density (2995
                                           # vs 3953) — the top onset disagreement was
                                           # kick+pad→pad (we drop the kick). The dotted
                                           # quarter collapses only the sub-dotted-quarter
                                           # double-bash: kick recall 41.1→48.8%, density
                                           # 3568 (closer to 3953), cat-acc 70.5→70.9,
                                           # with overall recall/precision and the Easy
                                           # cascade unchanged.
    "easy":   QUARTER_KICK_MS["easy"],     # ≈600 ms (kick on Easy is already rare:
                                           # crash/intensity drop it; collapsing more
                                           # doesn't help — Easy's excess is pad/snare)
}

# KICK_HARD_8TH_FACTOR — the Hard collapse is min(360ms, eighth×THIS factor). At high
#   tempo (metal) the eighth shrinks and the threshold drops, so only SUB-eighth
#   double-bash is collapsed (keeps the real double-bass); at low tempo it stays at
#   360ms (lean Hard).
KICK_HARD_8TH_FACTOR = 1.5

# FILL_MS — COMBINED pad rate below which a run is a fill/roll (sub-eighth). Used to
#   detect fills to collapse. Per level = the target eighth.
FILL_MS = dict(EIGHTH_MIN_MS)


# ── Tempo/measure context (reuses midi_utils) ─────────────────────────────────

@dataclass
class Ctx:
    tempo_map: list
    tpb: int
    time_sig_map: list

    def _ms(self, tick: int) -> float:
        return tick_to_ms(tick, self.tempo_map, self.tpb)

    def span_ms(self, tick: int, span_ticks: int) -> float:
        """Duration in ms of `span_ticks` starting at `tick` (LOCAL tempo)."""
        return self._ms(tick + span_ticks) - self._ms(tick)

    def quarter_ms(self, tick: int) -> float:
        return self.span_ms(tick, self.tpb)

    def eighth_ms(self, tick: int) -> float:
        return self.span_ms(tick, self.tpb // 2)

    def sixteenth_ms(self, tick: int) -> float:
        return self.span_ms(tick, self.tpb // 4)

    def measure_len(self, tick: int) -> int:
        return measure_ticks_at(tick, self.time_sig_map, self.tpb)

    def measure_index(self, tick: int) -> int:
        return tick // max(1, self.measure_len(tick))

    def frac_in_beat(self, tick: int) -> float:
        return (tick % self.tpb) / self.tpb

    def is_onbeat(self, tick: int, tol_ticks: int | None = None) -> bool:
        """True if the tick sits on a QUARTER (main beat)."""
        tol = self.tpb // 16 if tol_ticks is None else tol_ticks
        r = tick % self.tpb
        return min(r, self.tpb - r) <= tol

    def is_offbeat(self, tick: int, tol_ticks: int | None = None) -> bool:
        """True if the tick sits on an off eighth (mid-beat)."""
        tol = self.tpb // 16 if tol_ticks is None else tol_ticks
        return abs((tick % self.tpb) - self.tpb // 2) <= tol


# ── Shared primitives ─────────────────────────────────────────────────────────

def grid_thin(ticks: set[int], gap: int) -> set[int]:
    """Grid-cell thinning: one note per cell of width `gap`, the one nearest the
    grid line (strong beat). Returns the KEPT set."""
    if gap <= 0 or not ticks:
        return set(ticks)
    cells: dict[int, list[int]] = defaultdict(list)
    for t in ticks:
        cells[t // gap].append(t)
    return {min(ts, key=lambda x: abs(x - cell * gap)) for cell, ts in cells.items()}


def adaptive_pad_grid(ticks: set[int], ctx: Ctx, eighth_min_ms: float) -> set[int]:
    """Tempo-adaptive PAD grid (book: 8th timekeeping only up to ~threshold):
    cell = eighth where the local eighth is playable (≥ eighth_min_ms); otherwise
    quarter. Keeps slow hand ostinatos and collapses the fast ones onto the pulse."""
    if not ticks:
        return set()
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for t in ticks:
        gap = (ctx.tpb // 2) if ctx.eighth_ms(t) >= eighth_min_ms else ctx.tpb
        cells[(gap, t // gap)].append(t)
    return {min(ts, key=lambda x: abs(x - cell * gap))
            for (gap, cell), ts in cells.items()}


def snap_to_grid(ticks: set[int], gap: int) -> set[int]:
    """Like grid_thin but meant to align kick/snare to a cymbal grid."""
    return grid_thin(ticks, gap)


def thin_kicks(kick_ticks, ctx: Ctx, fill_ticks: set[int], collapse_ms: float) -> list[int]:
    """Kick thinning that PRESERVES THE ORIGINAL GROOVE (user's decision): keeps the
    Expert positions (downbeats + real syncopations), dropping only the DOUBLE-BASH
    kicks — those closer than `collapse_ms` to the already-kept structural kick. It
    NEVER snaps to the grid line: the phase follows what the drummer played. In a
    collapsed pair it prefers the strong beat (switches to it, but keeps it at its
    original position — it does not move it to the line). Kicks in fills are removed."""
    kept: list[int] = []
    for t in sorted(t for t in kick_ticks if t not in fill_ticks):
        if not kept:
            kept.append(t)
        elif ctx.span_ms(kept[-1], t - kept[-1]) >= collapse_ms:
            kept.append(t)                       # far apart → structural kick, keep
        elif ctx.is_onbeat(t) and not ctx.is_onbeat(kept[-1]):
            kept[-1] = t                         # double-bash: keep the strong beat
        # else: adjacent double-bash → drop (keep the previous, original position)
    return kept


def _prefer_in_pair(prev: int, cur: int, vel, ctx: Ctx, strong: bool) -> int:
    """Of two colliding lane onsets, which to KEEP. The ACCENT wins (louder beats
    ghost — the main hit must survive a ghost grace note). On a velocity tie, the
    STRONG BEAT wins when `strong` (keeps the downbeat over an off-grid neighbour);
    otherwise the earlier (original phase) stays."""
    if vel is not None:
        vp, vc = vel.get(prev, 0), vel.get(cur, 0)
        if vc != vp:
            return cur if vc > vp else prev
    if strong and ctx is not None and ctx.is_onbeat(cur) and not ctx.is_onbeat(prev):
        return cur
    return prev


def thin_lane_strong(ticks, ctx: Ctx, min_gap: int, vel=None) -> list[int]:
    """Like `thin_lane` but in a collapsed pair (gap < min_gap) it prefers the ACCENT
    (louder note — drops ghosts, keeps the main hit) and, on a velocity tie, the STRONG
    BEAT. Keeps the real position. Fixes the snare phase/dynamics in 16th grooves."""
    kept: list[int] = []
    for t in sorted(ticks):
        if not kept or t - kept[-1] >= min_gap:
            kept.append(t)
        else:
            kept[-1] = _prefer_in_pair(kept[-1], t, vel, ctx, strong=True)
    return kept


def thin_lane(ticks, min_gap: int, vel=None, ctx: Ctx | None = None) -> list[int]:
    """Greedy thinning that PRESERVES THE ORIGINAL PHASE: walks in order and keeps a
    tick if it is ≥ `min_gap` from the last kept one. Never snaps to the grid — unlike
    `grid_thin`, it does not force the note onto the beat line, so a backbone on the
    off-beat (half-time feel) survives intact. (Option 1 for snare.) When `vel` is
    given, a collision keeps the ACCENT (louder) over a ghost grace note."""
    kept: list[int] = []
    for t in sorted(ticks):
        if not kept or t - kept[-1] >= min_gap:
            kept.append(t)
        elif vel is not None:
            kept[-1] = _prefer_in_pair(kept[-1], t, vel, ctx, strong=False)
    return kept


def detect_fills(gems: list, ctx: Ctx, max_ms: float) -> list[list[int]]:
    """Runs whose COMBINED pad rate (all hand lanes together, no kick) is faster than
    `max_ms` → unplayable fills/rolls. Returns lists of ticks. Pure function over gems
    (reuses the idea of the old `_collapse_runs`).

    The rate is COMBINED on purpose: a fill `R-Y-B-G` has 1 note per lane, so it's
    never fast within ONE lane — only together is it a roll.

    But the combined rate alone confuses a fill with fast timekeeping (a hi-hat of
    8ths at >150BPM is also "fast"). Two tests tell a FILL apart:
      • IT MOVES BETWEEN LANES — an ostinato stays on a single color; a fill walks the
        kit (R-Y-B-G, R-Y-B) → ≥3 distinct LANES in the run.
      • IT IS SHORT — a transition flourish fits in ~1 measure; a metal 16th section
        runs across several measures as a continuous run (groove, not fill)
        → run span ≤ 1 measure.
    It also requires ≥3 consecutive notes sub-`max_ms`."""
    pad = sorted((t, n) for t, n, *_ in gems if category(n) in (PAD, SNARE))
    pad_ticks = [t for t, _ in pad]
    runs: list[list[int]] = []
    i = 0
    while i < len(pad):
        j = i
        while (j + 1 < len(pad)
               and ctx.span_ms(pad_ticks[j], pad_ticks[j + 1] - pad_ticks[j]) < max_ms):
            j += 1
        run = pad[i:j + 1]
        span = pad_ticks[j] - pad_ticks[i]
        if (len(run) >= 3 and len({n for _, n in run}) >= 3
                and span <= ctx.measure_len(pad_ticks[i])):
            runs.append([t for t, _ in run])
        i = j + 1
    return runs


def collapse_fill(gems: list, fill_ticks: set[int], ctx: Ctx,
                  eighth_min_ms: float, to_quarter: bool = False) -> list[tuple[int, int]]:
    """Collapse a fill ACROSS the lanes (color is irrelevant in the book). The
    per-lane grid misses fills because each color only has 1 hit; here all hand lanes
    are merged and ONE gem is kept per grid cell (the one nearest the line).
    `to_quarter` forces a quarter grid (Medium/Easy); otherwise an adaptive eighth
    grid (Hard). Returns [(tick, note)] — preserves the note that was there."""
    chosen: dict[int, int] = {}
    for t, n, *_ in gems:
        if t in fill_ticks and category(n) in (PAD, SNARE):
            # color irrelevant — keep the first (stable and deterministic)
            chosen.setdefault(t, n)
    ticks = set(chosen)
    if not ticks:
        return []
    if to_quarter:
        kept = grid_thin(ticks, ctx.tpb)
    else:
        kept = adaptive_pad_grid(ticks, ctx, eighth_min_ms)
    return [(t, chosen[t]) for t in kept]


def section_intensity(gems: list, ctx: Ctx) -> dict[int, bool]:
    """Per measure: normalized gem density → "intense" (True) vs "calm".
    Cut = median of the per-measure density (basis of the book's E2)."""
    per_measure: dict[int, int] = defaultdict(int)
    for t, *_ in gems:
        per_measure[ctx.measure_index(t)] += 1
    if not per_measure:
        return {}
    counts = sorted(per_measure.values())
    median = counts[len(counts) // 2]
    return {m: (c > median) for m, c in per_measure.items()}


# ── Gem model ─────────────────────────────────────────────────────────────────

def parse_gems(events: list[AbsEvent]) -> tuple[list, list[AbsEvent]]:
    """Convert Expert events (post-Expert+) into (gems, markers).

    gems    = [(tick, note, dur)] of the playable lanes.
    markers = all other events (tom 110-112, globals, meta) kept intact.

    SOURCE = EXPERT+ (full double bass) whenever it exists: the kick2x lane (95)
    is merged with the main kick (96) below, so EVERY double-bass kick feeds the
    cascade (Hard←Expert+, then Medium←Hard, Easy←Medium). The double-bash is
    later collapsed by `thin_kicks` to the structural positions — but it is the
    Expert+ groove (not a pre-thinned Expert) that is reduced. If a chart has no
    2x notes, 96 alone already IS the full Expert. So the source is always the
    richest kick lane available."""
    notes, others = pair_notes(events)
    gems: list = []
    markers: list[AbsEvent] = list(others)
    for n in notes:
        note = n.note
        if note in (KICK_NOTE, DRUM_KICK_2X):   # 96 + 95(2x) -> reduce from Expert+
            gems.append((n.start, KICK_NOTE, max(1, n.end - n.start), n.velocity))
        elif note == SNARE_NOTE or note in PAD_NOTES:
            gems.append((n.start, note, max(1, n.end - n.start), n.velocity))
        elif note in GLOBAL_MARKERS or note in DRUM_TOM_MARKERS:
            # markers: pass through intact (note_on + note_off)
            from mido import Message
            markers.append(AbsEvent(n.start, Message(
                "note_on", note=note, velocity=n.velocity, channel=n.channel, time=0)))
            markers.append(AbsEvent(n.end, Message(
                "note_off", note=note, velocity=0, channel=n.channel, time=0)))
        # any other (unrecognized) note is discarded (as in the old code)
    gems.sort()
    return gems, markers


def by_tick(gems: list) -> dict[int, dict[int, tuple[int, int]]]:
    """tick → {note: (dur, vel)}. Preserves each gem's velocity (dynamics)."""
    d: dict[int, dict[int, tuple[int, int]]] = defaultdict(dict)
    for t, n, dur, vel in gems:
        d[t][n] = (dur, vel)
    return d


def lanes_by_note(gems: list, pred) -> dict[int, set[int]]:
    """note → set of ticks, for the notes satisfying `pred(note)`."""
    out: dict[int, set[int]] = defaultdict(set)
    for t, n, *_ in gems:
        if pred(n):
            out[n].add(t)
    return out


def rebuild(bt: dict[int, dict[int, tuple[int, int]]],
            keep: set[tuple[int, int]]) -> list:
    """Rebuild the gem list keeping only the (tick, note) pairs in `keep`.
    The original dur/vel (incl. ghost/accent) are preserved."""
    out = []
    for (t, n) in keep:
        dur, vel = bt.get(t, {}).get(n, (1, 100))
        out.append((t, n, dur, vel))
    out.sort()
    return out


# ── HARD (←Expert) — keep grooves, simplify nuances ───────────────────────────

def reduce_hard(gems: list, ctx: Ctx) -> list:
    bt = by_tick(gems)
    keep: set[tuple[int, int]] = set()
    fill_ticks = {t for run in detect_fills(gems, ctx, FILL_MS["hard"]) for t in run}

    # H1/H4/H6 — PADS (hands) OUTSIDE fills: 8th timekeeping only up to ~150BPM,
    # otherwise quarters. Per-lane grid (keeps hand ostinatos).
    for note, ticks in lanes_by_note(gems, lambda n: n in PAD_NOTES).items():
        for t in adaptive_pad_grid(ticks - fill_ticks, ctx, EIGHTH_MIN_MS["hard"]):
            keep.add((t, note))
    # H4 — FILLS: collapse ACROSS the lanes to the eighth (the per-lane grid misses them).
    for t, note in collapse_fill(gems, fill_ticks, ctx, EIGHTH_MIN_MS["hard"]):
        keep.add((t, note))

    # H5 — SNARE: groove backbone, but no consecutive snares closer than 1/8 (removes
    # ~half the accents consistently). The official Hard charts DROP the ghost snares
    # that fall on a sub-16th position (between the eighths) — measured: those onsets
    # were almost all ghosts (vel<100) and were ~1700 false snares. Keeping the snare
    # only on the EIGHTH grid (on-beat OR off-eighth, both survive — math-derived from
    # tpb, no snap) raised snare precision 74.9→89.9% for −3.9pp snare recall (overall
    # Hard precision +5.7pp at −0.4pp recall).
    snare = sorted(t for t, n, *_ in gems if n == SNARE_NOTE and t not in fill_ticks
                   and (ctx.is_onbeat(t) or ctx.is_offbeat(t)))
    svel = {t: bt[t][SNARE_NOTE][1] for t in snare}
    min_gap = ctx.tpb // 2 - ctx.tpb // 16   # 1/8 with 1/16 slack
    for t in thin_lane_strong(snare, ctx, min_gap, vel=svel):
        keep.add((t, SNARE_NOTE))

    # H3 — KICK: ~75% of Expert. Remove ALL kicks from fills (H3); reinforce the
    # quarter grid by dropping extra kicks on adjacent 8th/16th (collapses double-bash).
    # BPM-adaptive collapse: at high tempo the quarter shrinks and the fixed 360ms
    # collapsed REAL double-bass (Beyond ~180BPM lost 43% of the kick). The threshold
    # becomes min(360, eighth*1.5) -> at high tempo it only collapses sub-eighth (keeps
    # metal double-bass), at low tempo it stays at 360ms (keeps lean Hard).
    # NOTE: the old H2 rule ("off-beat crash → drop the kick under it") was REMOVED on
    # Hard — measured vs the official Hard charts, those kicks are KEPT (kick recall
    # 69.4→79.8%, precision unchanged). H2 stays only on Medium (M5) and Easy (E1/E3),
    # which the book simplifies harder. The crash-vs-kick collision is left to Medium.
    _cms = min(KICK_COLLAPSE_MS["hard"], ctx.eighth_ms(0) * KICK_HARD_8TH_FACTOR)
    kick = (t for t, n, *_ in gems if n == KICK_NOTE)
    for t in thin_kicks(kick, ctx, fill_ticks, _cms):
        keep.add((t, KICK_NOTE))

    # H7 — HAND VELOCITY: a hand can't cross to a different pad faster than the playable
    # eighth. Drops sub-eighth cross-kit moves that survived because each lane's grid
    # kept its single note independently. 2-pad chords (same tick) stay.
    keep = _apply_hand_velocity(keep, ctx, EIGHTH_MIN_MS["hard"])

    return rebuild(bt, keep)


# ── MEDIUM (←Hard) — rock skeleton ────────────────────────────────────────────

def reduce_medium(gems: list, ctx: Ctx) -> list:
    bt = by_tick(gems)
    keep: set[tuple[int, int]] = set()
    fill_ticks = {t for run in detect_fills(gems, ctx, FILL_MS["medium"]) for t in run}

    # M2 — no 8th timekeeping above the threshold (~140BPM) → quarters (fewer places
    # for kick). M4 — fills → 8th (or less at high tempo).
    pad_grid = {}
    for note, ticks in lanes_by_note(gems, lambda n: n in PAD_NOTES).items():
        kept = adaptive_pad_grid(ticks - fill_ticks, ctx, EIGHTH_MIN_MS["medium"])
        pad_grid[note] = kept
        for t in kept:
            keep.add((t, note))
    # M4 — FILLS: collapse ACROSS the lanes to the quarter pulse (≤1 note/beat).
    for t, note in collapse_fill(gems, fill_ticks, ctx,
                                 EIGHTH_MIN_MS["medium"], to_quarter=True):
        keep.add((t, note))
        pad_grid.setdefault(note, set()).add(t)

    # M1 — snare backbone PRESERVING THE PHASE (option 1): keeps the drummer's real
    # positions, thinning only by density (≤1/quarter), without snapping to the grid —
    # so as not to destroy off-beat (half-time) grooves like TTFAF/Battery.
    snare = {t for t, n, *_ in gems if n == SNARE_NOTE}
    svel = {t: bt[t][SNARE_NOTE][1] for t in snare}
    for t in thin_lane(snare, ctx.tpb, vel=svel, ctx=ctx):
        keep.add((t, SNARE_NOTE))

    # M3 — kick: preserves the original groove, collapsing sub-quarter double-bash
    # (book: "kick only on quarters" above ~110BPM), without snapping to the grid.
    kick = (t for t, n, *_ in gems if n == KICK_NOTE)
    for t in thin_kicks(kick, ctx, fill_ticks, KICK_COLLAPSE_MS["medium"]):
        # M3b — kick ON-BEAT ONLY: the Medium charter drops off-beat kicks (syncopation
        # is a Hard nuance). Cuts the kick density ~in half without losing the groove.
        if not ctx.is_onbeat(t):
            continue
        # M5 — off-beat crash WITHOUT a kick (downbeat crash keeps its kick).
        if ctx.is_offbeat(t) and any(n in PAD_NOTES for n in bt.get(t, {})):
            continue
        keep.add((t, KICK_NOTE))

    # M6 — HAND VELOCITY: no cross-kit hand move faster than the playable eighth.
    keep = _apply_hand_velocity(keep, ctx, EIGHTH_MIN_MS["medium"])
    # M7 — NO THREE-LIMB HITS: at most 2 simultaneous gems (2 pads allowed; drop the
    # 3rd member, keeping snare+kick backbone over an extra pad/cymbal).
    keep = _enforce_no_three_limbs(keep, bt)

    return rebuild(bt, keep)


# ── EASY (←Medium) — two beat types by intensity ──────────────────────────────

def reduce_easy(gems: list, ctx: Ctx) -> list:
    bt = by_tick(gems)
    intensity = section_intensity(gems, ctx)
    fill_ticks = {t for run in detect_fills(gems, ctx, FILL_MS["easy"]) for t in run}
    keep: set[tuple[int, int]] = set()

    # E4 — fills 8th → quarters at high tempo. Pads to the quarter (slow timekeeping).
    for note, ticks in lanes_by_note(gems, lambda n: n in PAD_NOTES).items():
        kept = adaptive_pad_grid(ticks - fill_ticks, ctx, EIGHTH_MIN_MS["easy"])
        for t in kept:
            keep.add((t, note))
    # E4 — FILLS: collapse ACROSS the lanes to the quarter (at high tempo, ≤1/beat).
    for t, note in collapse_fill(gems, fill_ticks, ctx,
                                 EIGHTH_MIN_MS["easy"], to_quarter=True):
        keep.add((t, note))

    # SNARE backbone PRESERVING THE PHASE (option 1): real positions, ≤1/quarter,
    # without snap — keeps off-beat grooves at high tempo.
    snare = {t for t, n, *_ in gems if n == SNARE_NOTE}
    svel = {t: bt[t][SNARE_NOTE][1] for t in snare}
    for t in thin_lane(snare, ctx.tpb, vel=svel, ctx=ctx):
        keep.add((t, SNARE_NOTE))

    # E4 — kick: preserves the original groove, collapsing sub-quarter double-bash
    # (book: kick only on quarters at high tempo), without snapping to the grid.
    kept_kick = thin_kicks(
        (t for t, n, *_ in gems if n == KICK_NOTE),
        ctx, fill_ticks, KICK_COLLAPSE_MS["easy"])

    # E2 — two base beats by section INTENSITY:
    #   calm section    → kick+snare (keep the kick);
    #   intense section → two-hands without kick (drop the kick).
    # E3/E1 — no kick under a crash; the kick doesn't pair with a crash (kick+snare OK).
    for t in kept_kick:
        if not ctx.is_onbeat(t):
            continue                                          # E: kick only on-beat
        if intensity.get(ctx.measure_index(t), False):
            continue                                          # E2 intense → no kick
        if any(n in PAD_NOTES for n in bt.get(t, {})):
            continue                                          # E3/E1 crash → no kick
        keep.add((t, KICK_NOTE))

    # E1 — no 3 simultaneous members. Weight: snare/crash > kick > tom; on collision
    # the kick is dropped, then it is reduced to a single pad.
    keep = _enforce_max_two(keep, bt)

    # E6 — clean the texture around off-beats: already covered by the quarter grid.
    return rebuild(bt, keep)


def _enforce_max_two(keep: set[tuple[int, int]], bt: dict) -> set[tuple[int, int]]:
    """E1: at most 2 members per tick and never kick+crash. Drops the kick first,
    then the extra pads (keeps 1), preferring to keep the snare."""
    by_t: dict[int, set[int]] = defaultdict(set)
    for (t, n) in keep:
        by_t[t].add(n)
    out: set[tuple[int, int]] = set()
    for t, notes in by_t.items():
        has_snare = SNARE_NOTE in notes
        pads = [n for n in notes if n in PAD_NOTES]
        has_kick = KICK_NOTE in notes
        chosen: set[int] = set()
        if has_snare:
            chosen.add(SNARE_NOTE)
        if pads:
            chosen.add(pads[0])            # a single pad
        if has_kick and not pads and len(chosen) < 2:
            chosen.add(KICK_NOTE)          # kick only without a crash (kick+snare OK)
        for n in chosen:
            out.add((t, n))
    return out


# ── Hand velocity + limb count (Hard & Medium) ────────────────────────────────

def _apply_hand_velocity(keep: set[tuple[int, int]], ctx: Ctx,
                         min_ms: float) -> set[tuple[int, int]]:
    """HAND-VELOCITY limit: a single hand cannot travel to a DIFFERENT pad faster than
    `min_ms`. This bites ONLY between two consecutive LONE CYMBAL/TOM pads (one stick
    moving across the kit) — e.g. a Y on one lane then a B a 16th later, each kept
    independently by its own lane grid, which together is an unplayable cross. It does
    NOT touch:
      • the kick (foot, independent of hand travel);
      • a 2-pad CHORD (two hands striking together — a chord is an anchor, never split);
      • a second hand JOINING/leaving an ongoing pad (e.g. {Y}→{R,Y}: the hi-hat hand
        stays, the snare hand joins — not a cross).
    On a too-fast single→single cross the later note is dropped, unless it sits on a
    stronger beat, in which case it replaces the earlier one (`thin_kicks` preference).
    `min_ms` = the level's playable eighth (EIGHTH_MIN_MS). Returns the filtered `keep`
    (every kick preserved)."""
    hand_by_tick: dict[int, set[int]] = defaultdict(set)
    for (t, n) in keep:
        if n != KICK_NOTE:
            hand_by_tick[t].add(n)

    drop_ticks: set[int] = set()
    last: tuple[int, int] | None = None    # (tick, note) of last kept lone-PAD onset
    for t in sorted(hand_by_tick):
        notes = hand_by_tick[t]
        pads = [n for n in notes if n in PAD_NOTES]
        # Only a LONE cymbal/tom pad is a single-hand travel candidate. A chord, or any
        # onset carrying the snare (its own dedicated hand), resets the anchor and is kept.
        if len(notes) != 1 or len(pads) != 1:
            last = None
            continue
        n = pads[0]
        if last is None:
            last = (t, n)
            continue
        lt, ln = last
        if n == ln or ctx.span_ms(lt, t - lt) >= min_ms:
            last = (t, n)                  # same pad (ostinato) OR hand had time to move
        elif ctx.is_onbeat(t) and not ctx.is_onbeat(lt):
            drop_ticks.discard(t)
            drop_ticks.add(lt)             # too-fast cross → keep the strong beat
            last = (t, n)
        else:
            drop_ticks.add(t)              # hand can't reach in time → drop later cross

    return {(t, nn) for (t, nn) in keep if nn == KICK_NOTE or t not in drop_ticks}


def _enforce_no_three_limbs(keep: set[tuple[int, int]], bt: dict) -> set[tuple[int, int]]:
    """MEDIUM (book): NO THREE-LIMB HITS — at most 2 simultaneous gems (two limbs).
    Two-pad chords ARE allowed (two hands). When 3+ members stack, drop the KICK
    (foot) first — the book's "no kick, snare AND crash at the same time" — keeping
    the hands: snare, then pads. (A bare 2-pad chord, with no kick, stays intact.)"""
    by_t: dict[int, set[int]] = defaultdict(set)
    for (t, n) in keep:
        by_t[t].add(n)
    out: set[tuple[int, int]] = set()
    for t, notes in by_t.items():
        if len(notes) <= 2:
            out |= {(t, n) for n in notes}
            continue
        chosen: list[int] = []
        if SNARE_NOTE in notes:
            chosen.append(SNARE_NOTE)
        for n in sorted(x for x in notes if x in PAD_NOTES):
            if len(chosen) >= 2:
                break
            chosen.append(n)
        if KICK_NOTE in notes and len(chosen) < 2:
            chosen.append(KICK_NOTE)
        out |= {(t, n) for n in chosen}
    return out


# ── Emission ──────────────────────────────────────────────────────────────────

def emit_events(gems: list, offset: int, markers: list[AbsEvent]) -> list[AbsEvent]:
    """Convert gems → mido events with the level's note offset, merging in the
    markers (passed through intact)."""
    import mido
    out: list[AbsEvent] = []
    for t, note, dur, vel in gems:
        out.append(AbsEvent(t, mido.Message(
            "note_on", note=note + offset, velocity=vel, time=0)))
        out.append(AbsEvent(t + dur, mido.Message(
            "note_off", note=note + offset, velocity=0, time=0)))
    for ev in markers:
        out.append(ev.copy())
    out.sort(key=lambda e: e.abs_tick)
    return out


# ── Cascade driver ────────────────────────────────────────────────────────────

def reduce_drums_all(
    expert_events: list[AbsEvent],
    diffs: list[str],
    tempo_map: list,
    tpb: int,
    time_sig_map: list,
) -> dict[str, list[AbsEvent]]:
    """Reduce the Expert drums to all requested levels, in CASCADE
    (Hard←Expert, Medium←Hard, Easy←Medium). Returns {diff: events}."""
    ctx = Ctx(tempo_map, tpb, time_sig_map)
    chart_expert, markers = parse_gems(expert_events)

    charts: dict[str, list] = {"expert": chart_expert}
    charts["hard"]   = reduce_hard(chart_expert, ctx)
    charts["medium"] = reduce_medium(charts["hard"], ctx)
    charts["easy"]   = reduce_easy(charts["medium"], ctx)

    return {diff: emit_events(charts[diff], DIFF_OFFSET[diff], markers)
            for diff in diffs if diff in charts}

"""quality.py — cross-cutting quality guard (Hard/Medium/Easy).

Flags reductions that lost the song's GROOVE, without needing a manual chart:
it measures the fraction of the Expert structural ACCENTS (notes followed by
>= 1 beat of space — the riff's pulse) that the reduction preserved (±TOL). A low
fraction = the reduction does not follow the song's pulse (phase/period off the
grid, badly chosen section). Usable in the production pipeline as a sanity check
before publishing.
"""
from .midi_utils import AbsEvent
from .guitar import pair_notes, note_to_fret
import bisect

TOL = 20

# Warning thresholds per difficulty. Calibrated ~0.10-0.15 below the minimum of
# the GOOD songs in the learn-set (Hard min 95%, Medium min 67%, Easy min 50%), to
# catch real collapses (e.g. Easy-Nightmare on a pure grid dropped to 30%) without
# false positives. Below these the reduction probably lost the groove — inspect it.
GROOVE_FLOOR = {"hard": 0.80, "medium": 0.55, "easy": 0.40}


def expert_accents(events: list[AbsEvent], tpb: int) -> list[int]:
    """Ticks of the Expert structural accents: a note followed by >= 1 beat of space."""
    notes, _ = pair_notes(events)
    starts = sorted({n.start for n in notes
                     if note_to_fret(n.note, "expert") is not None})
    return [t for i, t in enumerate(starts)
            if (starts[i + 1] if i + 1 < len(starts) else t + tpb * 2) - t >= tpb]


def accent_preservation(expert_events: list[AbsEvent],
                        reduced_events: list[AbsEvent],
                        diff: str, tpb: int) -> float:
    """Fraction of Expert accents preserved (±TOL) in the `diff` reduction."""
    accents = expert_accents(expert_events, tpb)
    if not accents:
        return 1.0
    rstarts = sorted({e.abs_tick for e in reduced_events
                      if e.msg.type == "note_on"
                      and getattr(e.msg, "velocity", 0) > 0
                      and note_to_fret(e.msg.note, diff) is not None})
    hit = 0
    for a in accents:
        i = bisect.bisect_left(rstarts, a)
        if any(0 <= ci < len(rstarts) and abs(rstarts[ci] - a) <= TOL
               for ci in (i - 1, i)):
            hit += 1
    return hit / len(accents)


def groove_check(expert_events: list[AbsEvent], reduced_events: list[AbsEvent],
                 diff: str, tpb: int) -> tuple[bool, float]:
    """(ok, score) — ok=False if accent preservation falls below the level's floor."""
    score = accent_preservation(expert_events, reduced_events, diff, tpb)
    return score >= GROOVE_FLOOR.get(diff, 0.45), score

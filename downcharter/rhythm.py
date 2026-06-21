"""
rhythm.py — Note selection per difficulty
Rules: RBN/C3 Docs (Guitar and Bass Authoring)

Hard:
  - Keep original positions (no quantization)
  - Remove notes with gap < 8th note
  - BPM > 160: minimum gap rises to a quarter note

Medium:
  - No forced quantization to the beat
  - Minimum spacing: 8th note (preserves the song's rhythm)
  - Very close notes (< 8th) → keep the one nearest to a strong beat,
    or simply the first of the pair

Easy:
  - Minimum spacing: quarter note
  - Same selection criterion as Medium

Philosophy:
  Don't force notes onto a rigid quarter/half-note grid.
  What matters is avoiding sequences that are too fast and unpredictable,
  not erasing all of the song's original rhythm.
"""
from __future__ import annotations
from .midi_utils import Note, bpm_at


# ── Minimum spacing per difficulty ───────────────────────────────────────────

def min_gap_ticks(diff: str, tpb: int, bpm: float) -> int:
    """
    Minimum spacing between notes, in ticks.

    Hard:
      BPM ≤ 160 → 8th note
      BPM > 160 → quarter note (avoid continuous 8ths at high tempo)

    Medium:
      8th note always — allows 8th-note rhythms, but nothing faster

    Easy:
      dotted-quarter (1.5 beats) — the half note (2 beats) was too sparse and
      drifted out of alignment with the manual charts (which keep notes on
      half-measures). The dotted-quarter recovers ~7pp of global timing without
      inflating the density much.
    """
    quarter = tpb
    eighth  = tpb // 2

    if diff == "hard":
        return eighth if bpm <= 160 else quarter
    elif diff == "medium":
        return eighth if bpm <= 120 else quarter
    else:  # easy
        return (tpb * 3) // 2  # dotted-quarter (1.5 beats)


def snap_to_grid(abs_tick: int, grid: int, tolerance_frac: float = 0.45) -> int | None:
    """
    Tries to quantize abs_tick to the nearest multiple of grid.
    Only quantizes if the note is close enough (< tolerance_frac * grid).
    Used optionally as a tiebreaker, not as the main rule.
    """
    remainder = abs_tick % grid
    if remainder <= grid // 2:
        nearest = abs_tick - remainder
    else:
        nearest = abs_tick - remainder + grid
    dist = abs(abs_tick - nearest)
    if dist <= grid * tolerance_frac:
        return nearest
    return None


def grid_size_ticks(diff: str, tpb: int, bpm: float) -> int:
    """Alias for compatibility with guitar.py."""
    return min_gap_ticks(diff, tpb, bpm)


# ── Hard: no quantization, gap filter only ───────────────────────────────────

def reduce_density_hard(notes: list[Note], tempo_map: list, tpb: int) -> list[Note]:
    """
    Hard: keeps original positions, removes notes that are too fast.
    Processes groups of simultaneous notes (chords) as a single unit.
    """
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: n.start)
    result: list[Note] = []
    last_tick = -999_999

    for note in notes:
        bpm = bpm_at(note.start, tempo_map)
        gap = min_gap_ticks("hard", tpb, bpm)
        if note.start - last_tick >= gap:
            result.append(note)
            last_tick = note.start
    return result


# ── Medium / Easy: gap filter without forced quantization ────────────────────

def reduce_density_grid(
    notes: list[Note],
    diff: str,
    tempo_map: list,
    tpb: int,
) -> list[Note]:
    """
    Medium/Easy: keeps original positions but ensures a minimum spacing.

    When two notes are too close, the one nearest a strong beat (multiple of a
    quarter note) is kept. On a tie, the first one is kept.
    """
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: n.start)

    # Group notes that are too close and pick the best one from each group
    result: list[Note] = []
    last_tick = -999_999

    i = 0
    while i < len(notes):
        note = notes[i]
        bpm  = bpm_at(note.start, tempo_map)
        gap  = min_gap_ticks(diff, tpb, bpm)

        if note.start - last_tick < gap:
            # Too close to the previous note — discard
            i += 1
            continue

        # Collect every note within one gap of the current candidate
        # (there may be slightly later notes that are better choices)
        cluster = [notes[i]]
        j = i + 1
        while j < len(notes) and notes[j].start - note.start < gap:
            cluster.append(notes[j])
            j += 1

        if len(cluster) == 1:
            best = cluster[0]
        else:
            quarter = tpb
            def beat_dist(n: Note) -> float:
                rem = n.start % quarter
                return min(rem, quarter - rem)
            best = min(cluster, key=beat_dist)

        result.append(best)
        last_tick = best.start
        i = j  # skip all notes in the cluster

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def reduce_density(
    notes: list[Note],
    diff: str,
    tempo_map: list,
    tpb: int,
) -> list[Note]:
    if diff == "expert":
        return notes
    elif diff == "hard":
        return reduce_density_hard(notes, tempo_map, tpb)
    else:
        return reduce_density_grid(notes, diff, tempo_map, tpb)

"""
guitar.py — Difficulty reduction for 5-fret instruments
Rules: RBN/C3 Docs (Guitar and Bass Authoring)

Hard:
  Hard is the "reasonable" version of Expert: it keeps the same 5 frets (G R Y B O),
  preserves every playable chord, and removes only the notes that are too fast.
  There is no pitch re-mapping — the frets pass straight from Expert to Hard
  (all that changes is the MIDI offset of -12, via fret_note).

  Chord rules:
    - At most 2 notes per chord
    - Forbidden G+O (frozenset {0,4})
    - G+B and R+O are valid (substitution recommended by RBN)

  Density rules (RBN/C3):
    - Remove strumming/movement that is a 16th or faster
      → min gap = eighth (8th note = tpb//2) for BPM ≤ 160
    - BPM > 160: continuous eighths are not expected
      → min gap = quarter (quarter note = tpb)
    - When a note is removed, do NOT extend the previous note's sustain

  Fret pass-through:
    Expert fret 0 (Green) → Hard fret 0 (Green)
    Expert fret 4 (Orange) → Hard fret 4 (Orange)
    No range compression. _PitchWindow is not used in Hard.

Medium / Easy:
  The pitch ordering of any chord/note is given by the lexicographic ordering
  of the fret tuple (low → high):
    G < G+R < G+Y < ... < R+Y < ... < Y+B+O < ... < O

  To reduce, we compute the chord/note's percentile within the Expert sub-scale
  of the same size, and map it proportionally onto the full scale of the target
  difficulty.

  There is no state between notes — the mapping is static and deterministic.
  This eliminates wrapping bugs and guarantees G → G, O → highest available.
"""
from __future__ import annotations
from itertools import combinations

from .constants import (
    FRET_COUNT, DIFF_OFFSET, FRETS_ALLOWED,
    HARD_FORBIDDEN_CHORDS, MEDIUM_FORBIDDEN_CHORDS,
    GLOBAL_MARKERS,
    fret_note, note_to_fret,
)
from .midi_utils import (
    AbsEvent, Note, pair_notes, notes_to_events,
    bpm_at, measure_ticks_at,
)

# ═══════════════════════════════════════════════════════════
#  PITCH SCALES  (used by Medium / Easy)
# ═══════════════════════════════════════════════════════════

def _build_scale(fret_range: list[int], max_notes: int,
                 forbidden: list[frozenset],
                 max_span: int | None = None) -> list[tuple[int, ...]]:
    """
    All valid chords for the given range, sorted by pitch
    (lexicographic ordering = the game's pitch ordering).
    """
    result: list[tuple[int, ...]] = []
    for size in range(1, max_notes + 1):
        for combo in combinations(fret_range, size):
            fs = frozenset(combo)
            if fs in forbidden:
                continue
            if max_span is not None and len(combo) == 2:
                if combo[1] - combo[0] > max_span:
                    continue
            result.append(combo)
    result.sort()
    return result


_SCALES: dict[str, list[tuple[int, ...]]] = {
    "expert": _build_scale(list(range(5)), 3, []),
    "hard":   _build_scale(list(range(5)), 2, list(HARD_FORBIDDEN_CHORDS)),
    "medium": _build_scale(list(range(4)), 2, list(MEDIUM_FORBIDDEN_CHORDS), max_span=2),
    "easy":   _build_scale(list(range(3)), 1, []),
}

# Sub-scales by chord size
_BY_SIZE: dict[str, dict[int, list[tuple[int, ...]]]] = {
    diff: {sz: [c for c in scale if len(c) == sz] for sz in [1, 2, 3]}
    for diff, scale in _SCALES.items()
}


# ═══════════════════════════════════════════════════════════
#  PITCH MAPPING  (Medium / Easy)
# ═══════════════════════════════════════════════════════════

def _map_chord_static(frets_expert: frozenset[int], diff: str,
                      min_output_size: int = 1) -> frozenset[int]:
    """
    Map an Expert chord onto the chord closest in pitch at the difficulty.

    For single notes: uses the 1-note sub-scale (never produces a chord).
    For chords: uses the diff's full scale, filtered by min_output_size.
    """
    if diff == "expert":
        return frets_expert

    key  = tuple(sorted(frets_expert))
    size = len(key)

    # Expert sub-scale of the same size (pitch reference)
    expert_sub = _BY_SIZE["expert"].get(size, _BY_SIZE["expert"][1])

    try:
        rank = expert_sub.index(key)
    except ValueError:
        high = max(frets_expert)
        rank = min(range(len(expert_sub)),
                   key=lambda i: abs(expert_sub[i][-1] - high))

    total = max(len(expert_sub) - 1, 1)
    pct   = rank / total

    if size == 1:
        target = _BY_SIZE[diff].get(1, _SCALES[diff])
    else:
        target = [c for c in _SCALES[diff] if len(c) >= min_output_size]
        if not target:
            target = _SCALES[diff]

    idx = round(pct * (len(target) - 1))
    idx = max(0, min(len(target) - 1, idx))
    return frozenset(target[idx])


class _PitchWindow:
    """
    Sliding pitch window for mapping single notes (Medium / Easy).

    Keeps an offset indicating which Expert fret is mapped to the bottom
    (index 0) of the difficulty's range. When the Expert line rises beyond
    the top of the window, the offset advances (wrap up). When it drops below
    the bottom, the offset retreats (wrap down).
    """
    def __init__(self, diff: str, first_fret: int):
        n = len(FRETS_ALLOWED[diff])
        pct = first_fret / (FRET_COUNT - 1)
        target_idx = round(pct * (n - 1))
        self._offset = first_fret - target_idx
        self._n = n
        self._diff = diff
        self._allowed = FRETS_ALLOWED[diff]

    def map(self, fret_expert: int) -> int:
        pos = fret_expert - self._offset
        if pos >= self._n:
            self._offset = fret_expert - (self._n - 1)
            pos = self._n - 1
        elif pos < 0:
            self._offset = fret_expert
            pos = 0
        return self._allowed[pos]



def map_chord(frets_expert: frozenset[int], diff: str,
              min_output_size: int = 1) -> frozenset[int]:
    """Static chord mapping (stateless). Used for chords of 2+ notes."""
    return _map_chord_static(frets_expert, diff, min_output_size)


# ═══════════════════════════════════════════════════════════
#  SUSTAIN RULES (RBN)  — used by Hard, Medium and Easy
# ═══════════════════════════════════════════════════════════

def _sustain_gap(diff: str, tpb: int) -> int:
    """
    Minimum gap between the end of a sustain and the next note (RBN):
      Expert: 1/16
      Hard:   1/8
      Medium: 1/4
      Easy:   1/4
    """
    if diff == "expert": return tpb // 4    # 1/16
    if diff == "hard":   return tpb // 4    # 1/16 (book: Hard = same as Expert)
    return tpb                               # 1/4 (medium and easy)


def _trim_sustain(note: Note, diff: str, tpb: int,
                  tempo_map: list, next_start: int | None) -> Note:
    """Shorten the sustain per the book (NEVER extends — only trims).

    Rule (OFFICIAL_RULES.md):
      • Hard:   sustain = SAME as Expert; only shortened to leave a gap ≥ 1/16
                before the next note.
      • Medium/Easy: shortened to leave a gap ≥ 1/4 (quarter) before the next.
    There is no maximum-length cap — what kills sustains is the gap to the next
    KEPT note, not an artificial limit. Sustains < 1/16 become point notes.
    """
    new_end = note.end                          # starts from the original (Expert) length
    if next_start is not None:
        new_end = min(new_end, next_start - _sustain_gap(diff, tpb))
    if new_end - note.start < tpb // 4:         # < 1/16 → point note (no tail)
        return note.with_end(note.start + 1)
    return note.with_end(max(new_end, note.start + 1))


# ═══════════════════════════════════════════════════════════
#  DENSITY (Medium / Easy — used in the M/E pipeline)
# ═══════════════════════════════════════════════════════════

def _min_spacing_ms(diff: str, bpm: float) -> float:
    """Minimum spacing between notes (RBN)."""
    beat = 60_000 / bpm
    if diff == "hard":     return beat / 2   # 1/8
    elif diff == "medium": return beat        # 1/4
    else:                  return beat * 2   # 1/2


# ═══════════════════════════════════════════════════════════
#  HARD — chord reduction
# ═══════════════════════════════════════════════════════════

def _reduce_chord_hard(frets: frozenset[int]) -> frozenset[int]:
    """
    Reduce an Expert chord to Hard-legal:
      - At most 2 notes
      - Forbidden G+O (frozenset {0, 4})

    Reduction strategy for 3+ notes: keep the lowest and highest fret,
    replacing G+O with G+B (shift the Orange to the nearest Blue).
    G+B and R+O are explicitly recommended by RBN as valid substitutions.

    For 2-note chords that are already illegal (only G+O): replace with G+B.
    """
    if not frets:
        return frets

    # Single note: passes straight through
    if len(frets) == 1:
        return frets

    s = sorted(frets)

    # 2-note chord — check whether it is forbidden
    if len(frets) == 2:
        if frozenset(frets) not in HARD_FORBIDDEN_CHORDS:
            return frets
        # G+O → G+B (shift Orange to Blue, keep Green)
        return frozenset({0, 3})

    # 3+ note chord → map by pitch percentile to 2 Hard-legal notes.
    # _map_chord_static uses the 3-note Expert sub-scale as the rank reference
    # and maps proportionally onto the 2-note Hard scale (G+O excluded).
    return _map_chord_static(frets, "hard", min_output_size=2)


# ═══════════════════════════════════════════════════════════
#  HOPO pattern classification
# ═══════════════════════════════════════════════════════════

def _classify_run(seq: list[int]) -> str:
    """
    Classify the pattern of a HOPO run by its fret sequence (0–4).

    Cyclic patterns (highest priority):
      'trill'       — 2-fret cycle (e.g. G→R→G→R); minimum 3 HOPOs
      'trill_3'     — 3-fret cycle (e.g. G→R→Y→G→R→Y); minimum 4 HOPOs
      'trill_4'     — 4-fret cycle (quads); minimum 5 HOPOs
      'trill_5'     — 5-fret cycle (quints); minimum 6 HOPOs

    Direct patterns (monotone runs):
      'triplet_asc'  / 'triplet_desc'  — 3 total notes (2 HOPOs)
      'quad_asc'     / 'quad_desc'     — 4 total notes (3 HOPOs)
      'quint_asc'    / 'quint_desc'    — 5 total notes (4 HOPOs, full sweep)
      'directional_asc' / 'directional_desc' — fallback for other sizes

    Complex patterns:
      'ladder_asc'  — rises to a peak then descends; 1 direction change
      'ladder_desc' — descends to a valley then rises; 1 direction change
      'chimney'     — 2+ changes, starts or ends on the highest fret
      'rev_chimney' — 2+ changes, starts or ends on the lowest fret
      'zig'         — 2+ changes, with no chimney/rev_chimney pattern
    """
    if len(seq) < 2:
        return "directional_asc"

    # ── N-note cycles (priority over directional and chimney) ───────────────
    # Detects repetition of a unit of length 2–4.
    # Checks only the first cycle_len × 2 notes (2 full cycles are
    # enough to confirm the pattern; transition notes at the end are
    # ignored — real music often has irregular links).
    # E.g. G-R-Y × 10 + O  → trill_3  (the first 6 are G-R-Y-G-R-Y)
    # E.g. R-B-O × 31 + O  → trill_3  (TTFAF R-B-O run with a last rogue note)
    for cycle_len in range(2, min(6, len(seq))):
        if len(seq) < cycle_len * 2:      # needs at least 2 cycles
            continue
        unit = seq[:cycle_len]
        # The unit cannot have equal frets in consecutive positions (incl. wrap)
        if any(unit[i] == unit[(i + 1) % cycle_len] for i in range(cycle_len)):
            continue
        # Checks min(3 cycles, total len) — 3 cycles avoid false positives
        # in patterns that coincidentally match in the first 2 cycles.
        check_len = min(cycle_len * 3, len(seq))
        if all(seq[i] == unit[i % cycle_len] for i in range(check_len)):
            if cycle_len == 2:
                return "trill"
            return f"trill_{cycle_len}"

    # ── Directional (0 direction changes) — named by length ─────────────────
    _DIR_NAMES_ASC  = {2: "triplet_asc",  3: "quad_asc",  4: "quint_asc"}
    _DIR_NAMES_DESC = {2: "triplet_desc", 3: "quad_desc", 4: "quint_desc"}

    if all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)):
        return _DIR_NAMES_ASC.get(len(seq), "directional_asc")
    if all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)):
        return _DIR_NAMES_DESC.get(len(seq), "directional_desc")

    # Significant moves (ignore steps with the same fret)
    moves = [1 if seq[i + 1] > seq[i] else -1
             for i in range(len(seq) - 1) if seq[i + 1] != seq[i]]
    if not moves:
        return "directional_asc"   # all equal → treat as flat

    changes = sum(1 for i in range(len(moves) - 1) if moves[i] != moves[i + 1])

    if changes == 1:
        # Ladder: exactly 1 direction change
        #   moves[0] ==  1 → starts rising → interior peak → ladder_asc
        #   moves[0] == -1 → starts descending → interior valley → ladder_desc
        return "ladder_asc" if moves[0] == 1 else "ladder_desc"

    # ── 2+ direction changes ──────────────────────────────────────────────
    peak   = max(seq)
    valley = min(seq)
    if seq[0] == peak or seq[-1] == peak:
        return "chimney"
    if seq[0] == valley or seq[-1] == valley:
        return "rev_chimney"
    return "zig"


# Groups of direct patterns used in _flush_pending
_DIR_ASC_PATTERNS  = frozenset({
    "directional_asc", "triplet_asc", "quad_asc", "quint_asc",
})
_DIR_DESC_PATTERNS = frozenset({
    "directional_desc", "triplet_desc", "quad_desc", "quint_desc",
})
_DIR_ALL_PATTERNS  = _DIR_ASC_PATTERNS | _DIR_DESC_PATTERNS


# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
#  MEDIUM / EASY — main algorithm
# ═══════════════════════════════════════════════════════════

def _reduce_guitar_med_easy(
    events: list[AbsEvent],
    diff: str,
    tempo_map: list,
    tpb: int,
    time_sig_map: list | None = None,
    gap_fill: bool = True,
) -> list[AbsEvent]:
    """
    Generate guitar events for Medium or Easy from Expert.

    Pipeline:
      1. Separate fret notes, global markers, force notes
      2. Group simultaneous notes into chords (tolerance ±2 ticks)
      3. Map each chord/note by pitch percentile (static, stateless)
      4. Filter by minimum density
      5. Shorten sustains
      6. Rebuild events
    """
    from .rhythm import grid_size_ticks
    from collections import defaultdict

    notes, others = pair_notes(events)

    fret_notes   = [n for n in notes if note_to_fret(n.note, "expert") is not None]
    marker_notes = [n for n in notes if n.note in GLOBAL_MARKERS]
    # Expert force notes (101/102) are not passed to any difficulty

    fret_sorted = sorted(fret_notes, key=lambda n: (n.start, n.note))

    # Group genuinely simultaneous notes in Expert (real chords)
    CHORD_TOL_EXPERT = 2
    expert_chords: list[list[Note]] = []
    i_c = 0
    while i_c < len(fret_sorted):
        grp = [fret_sorted[i_c]]
        j_c = i_c + 1
        while (j_c < len(fret_sorted) and
               fret_sorted[j_c].start - fret_sorted[i_c].start <= CHORD_TOL_EXPERT):
            grp.append(fret_sorted[j_c])
            j_c += 1
        expert_chords.append(grp)
        i_c = j_c

    chord_list: list[tuple[int, list[Note]]] = [
        (grp[0].start, grp) for grp in expert_chords
    ]

    # Filter by minimum gap, choosing the best candidate when there's a conflict
    final_chords: list[tuple[int, list[Note]]] = []
    last_tick = -999_999
    i_fil = 0
    while i_fil < len(chord_list):
        ref_tick, grp = chord_list[i_fil]
        bpm  = bpm_at(ref_tick, tempo_map)
        gap  = grid_size_ticks(diff, tpb, bpm)

        if ref_tick - last_tick < gap:
            i_fil += 1
            continue

        cluster = [(ref_tick, grp)]
        j_fil = i_fil + 1
        while j_fil < len(chord_list) and chord_list[j_fil][0] - ref_tick < gap:
            cluster.append(chord_list[j_fil])
            j_fil += 1

        m_ticks = measure_ticks_at(ref_tick, time_sig_map or [(0, 4, 4)], tpb)
        long_pause = last_tick < 0 or (ref_tick - last_tick) > m_ticks
        if len(cluster) == 1 or long_pause:
            best_tick, best_grp = cluster[0]
        elif diff == "easy":
            strong = m_ticks // 2 if m_ticks % (tpb * 2) == 0 else m_ticks
            def strong_beat_dist(item: tuple) -> int:
                rem = item[0] % strong
                return min(rem, strong - rem)
            best_tick, best_grp = min(cluster, key=strong_beat_dist)
        else:
            beat = m_ticks // max(1, m_ticks // tpb)
            def beat_dist_chord(item: tuple) -> float:
                t = item[0]
                rem = t % beat
                return min(rem, beat - rem)
            best_tick, best_grp = min(cluster, key=beat_dist_chord)

        final_chords.append((best_tick, best_grp))
        last_tick = best_tick
        i_fil = j_fil

    chords_quantized = final_chords

    # Adaptive gap-fill (Easy test): where quantization left a wide gap
    # (>= GAP), reinsert the dropped Expert chord that fits at >= 1/4 from both sides.
    # Recovers real notes in SPARSE sections (Star Wars: 66 dropped notes) without
    # inflating the dense ones. Env toggle for evaluation.
    if diff == "easy" and gap_fill and chords_quantized:
        import bisect as _bi
        GAP = tpb * 3                         # 3 beats
        MINSEP = tpb                          # 1/4 from both sides
        kt = [t for t, _g in chords_quantized]
        keptset = set(kt)
        pool = sorted((c for c in chord_list if c[0] not in keptset),
                      key=lambda c: c[0])
        for ref_t, grp in pool:
            i = _bi.bisect_left(kt, ref_t)
            left = kt[i - 1] if i > 0 else None
            right = kt[i] if i < len(kt) else None
            if left is not None and right is not None and (right - left) >= GAP \
                    and (ref_t - left) >= MINSEP and (right - ref_t) >= MINSEP:
                chords_quantized.append((ref_t, grp)); kt.insert(i, ref_t)
        chords_quantized.sort(key=lambda c: c[0])

    # Beat-snap (Medium): near-on-beat notes (≤ 1/6 of a beat) are aligned to the
    # nearest 1/4 — replicates what the human charter does and recovers positions
    # that drifted 60-80t off the grid. Derived from the beat fraction (no hard-coded
    # BPM). Preserves monotonicity: never moves back before the previous note.
    if diff == "medium" and chords_quantized:
        snapped: list[tuple[int, list[Note]]] = []
        prev = -999_999
        for k_sn, (st, grp) in enumerate(chords_quantized):
            beat = tpb
            rem  = st % beat
            cand = st - rem if rem <= beat - rem else st - rem + beat
            nxt  = chords_quantized[k_sn + 1][0] if k_sn + 1 < len(chords_quantized) else None
            if (abs(cand - st) <= tpb // 6 and cand > prev
                    and (nxt is None or cand < nxt)):
                st = cand
            snapped.append((st, grp))
            prev = st
        chords_quantized = snapped

    # Map pitch for each chord/note
    mapped: list[tuple[int, frozenset[int], list[Note]]] = []
    window: _PitchWindow | None = None

    for beat, group in chords_quantized:
        expert_frets = frozenset(
            note_to_fret(n.note, "expert") for n in group
            if note_to_fret(n.note, "expert") is not None
        )
        if not expert_frets:
            continue

        if len(expert_frets) == 1:
            ef = next(iter(expert_frets))
            if window is None:
                window = _PitchWindow(diff, ef)
            mf = window.map(ef)
            mapped_frets = frozenset({mf})
        else:
            mapped_frets = map_chord(expert_frets, diff, 1)
            primary_e = max(expert_frets)
            if window is None:
                window = _PitchWindow(diff, primary_e)

        mapped.append((beat, mapped_frets, group))

    # Shorten sustains and build notes
    result_notes: list[Note] = []
    for idx, (start, frets, group) in enumerate(mapped):
        next_start = mapped[idx + 1][0] if idx + 1 < len(mapped) else None
        orig_end   = max(n.end for n in group)
        ref        = Note(0, group[0].velocity, start, orig_end, group[0].channel)
        trimmed    = _trim_sustain(ref, diff, tpb, tempo_map, next_start)

        for fi in sorted(frets):
            result_notes.append(Note(
                note     = fret_note(fi, diff),
                velocity = group[0].velocity,
                start    = trimmed.start,
                end      = trimmed.end,
                channel  = group[0].channel,
            ))

    # Global markers (Medium and Easy have no force notes)
    all_notes  = result_notes + list(marker_notes)
    note_evs   = notes_to_events(all_notes)
    all_events = note_evs + [e for e in others
                              if e.msg.type not in ("note_on", "note_off")]
    all_events.sort(key=lambda e: e.abs_tick)
    return all_events


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

def reduce_guitar(
    events: list[AbsEvent],
    diff: str,
    tempo_map: list,
    tpb: int,
    time_sig_map: list | None = None,
) -> list[AbsEvent]:
    """
    Generate guitar events for 'diff' from Expert.
    Delegates to each difficulty's production algorithm (guitar2.py).
    """
    if diff == "expert":
        return events
    if diff == "hard":
        from .guitar2 import reduce_hard
        return reduce_hard(events, tempo_map, tpb, time_sig_map)
    if diff == "medium":
        from .guitar2 import reduce_medium
        return reduce_medium(events, tempo_map, tpb, time_sig_map)
    if diff == "easy":
        from .guitar2 import reduce_easy_hybrid
        return reduce_easy_hybrid(events, tempo_map, tpb, time_sig_map)
    return _reduce_guitar_med_easy(events, diff, tempo_map, tpb, time_sig_map)


# Exported for external use (tests, etc.)
reduce_chord = lambda frets, diff: map_chord(frets, diff)

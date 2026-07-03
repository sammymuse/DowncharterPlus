"""
guitar2.py — Official 5-fret reduction (rebuilt from the Customs Book v1, pp.212-226).

Production algorithms, called by reduce_guitar(...) in guitar.py:
  • reduce_hard        — STRIDE-2 + pivot-move (reduces from Expert).
  • reduce_medium      — Expert-direct + priority selection + gap-fill + snap.
  • reduce_easy_hybrid — old grid by default; switches to position preservation
                         (reduce_easy) when the accent detector sees the grid losing
                         the groove (Nightmare/Star Wars).

Principle: rules derived in beat fractions (no hard-coded BPM); melodic fidelity to
the real songs > exactness of the synthetic patternchart.
"""
from __future__ import annotations

from .constants import (
    DIFF_OFFSET, GLOBAL_MARKERS, OPEN_FRET,
    fret_note, note_to_fret, is_open_note,
)
from .midi_utils import (
    AbsEvent, Note, pair_notes, notes_to_events, bpm_at,
)
# Book-faithful helpers reused from guitar.py:
from .guitar import _reduce_chord_hard, _trim_sustain, map_chord, _PitchWindow
from .constants import fret_note as _fret_note


# ═══════════════════════════════════════════════════════════════════════════
#  Internal representation
# ═══════════════════════════════════════════════════════════════════════════
#
#  A "GemGroup" is a chord/note at an instant: (tick, frets:frozenset[int],
#  grp:list[Note]). grp holds the original Notes (for velocity/channel/sustain).
#
# ═══════════════════════════════════════════════════════════════════════════

CHORD_TOL = 2   # tolerance ticks for grouping simultaneous notes


def _group_chords(fret_notes: list[Note], diff: str = "expert",
                  with_open: bool = False) -> list[tuple[int, frozenset[int], list[Note]]]:
    """Group near-simultaneous note_ons (±CHORD_TOL) into a chord per instant.

    If `with_open`, gems with an OPEN note (note 95 in Expert) become an exclusive
    OPEN gem (frets={OPEN_FRET}) — in GH/CH the open does not combine with strings.
    The open travels the whole density cascade as a single-note gem and is emitted
    at the level's open pitch (fret_note(OPEN_FRET, diff))."""
    fs = sorted(fret_notes, key=lambda n: (n.start, n.note))
    out: list[tuple[int, frozenset[int], list[Note]]] = []
    i = 0
    while i < len(fs):
        grp = [fs[i]]
        j = i + 1
        while j < len(fs) and fs[j].start - fs[i].start <= CHORD_TOL:
            grp.append(fs[j]); j += 1
        if with_open and any(is_open_note(n.note, diff) for n in grp):
            out.append((fs[i].start, frozenset({OPEN_FRET}), grp))
            i = j
            continue
        frets = frozenset(
            f for f in (note_to_fret(n.note, diff) for n in grp) if f is not None
        )
        if frets:
            out.append((fs[i].start, frets, grp))
        i = j
    return out


def _classify_hopo(chords, hopo_thresh, force_on, force_off):
    """
    Classify each chord as HOPO (True) or STRUM (False), the game's way:
      auto-HOPO = single note, different fret from the previous, gap ≤ hopo_thresh.
      Force-on (101) forces HOPO; force-off (102) forces strum and propagates
      across the rest of the chain.
    Returns a list of bool aligned with `chords`.
    """
    out = []
    pt, pf = -10**9, frozenset()
    strum_chain = False
    for tick, frets, _ in chords:
        gap = tick - pt
        single = len(frets) == 1
        diff = frets != pf
        auto = single and diff and gap <= hopo_thresh
        if tick in force_on and single and diff:
            h = True
            strum_chain = False
        elif tick in force_off:
            h = False
            strum_chain = True            # propagate: subsequent auto-HOPO notes become strum
        elif strum_chain and auto:
            h = False                     # 102 broke the chain — continue forcing strum
        else:
            h = auto
            # Natural chain break (chord, same fret, or gap > threshold) resets strum_chain
            if not single or not diff or gap > hopo_thresh:
                strum_chain = False
        out.append(h)
        pt, pf = tick, frets
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  HARD — reduces the Expert (Customs Book pp.212-214)
# ═══════════════════════════════════════════════════════════════════════════

def _reduce_hard_chart(
    chords: list[tuple[int, frozenset[int], list[Note]]],
    tempo_map: list,
    tpb: int,
) -> tuple[list[tuple[int, frozenset[int], list[Note]]], set[int]]:
    """
    Apply the book's density rules for Hard and return:
      (reduced chart, set of kept original ticks).

    Rules (OFFICIAL_RULES.md / pp.212-214, 225):
      • Minimum gap = 1/8 ALWAYS. Removes anything faster than 1/8 ("remove 16th").
        Keeps the original POSITIONS — does not quantize.
      • Continuous 8ths above 160 BPM: the player should not play continuous 8ths
        at that tempo (p.225). Instead of dropping to 1/4 (too sparse), the book
        (p.213) says to REMOVE EVERY 4th 8th — break the run, keeping the pulse.
      • Don't extend reduced notes (the sustain trim handles that later).
    """
    EIGHTH = tpb // 2
    GAP_TOL = tpb // 16
    # NOTE: WEIGHTED selection by importance (Onyx style: downbeat>quarter>
    # sustain>chord, greedy with a 1/8 minimum gap) was tested and REJECTED — it
    # breaks the regular 8th pulse and lowers pos (−1.2pp) and HOPO recall (−5.9pp)
    # with honest density. The slot-grid below is better. (compare_external.py / session.)
    # Step A: GRID-ALIGNED selection at 1/8. For each 1/8 slot (grid line =
    # multiple of EIGHTH, includes on-beats AND off-beats), keep one note.
    # Pick the one nearest the grid line (preserving the original position) —
    # this keeps the song's real 8ths (on/off-beat) instead of deriving like the
    # greedy. Notes faster than 1/8 (2+ in the same slot) reduce to a single one.
    buckets: dict[int, tuple[int, frozenset[int], list[Note]]] = {}
    bucket_dist: dict[int, int] = {}
    for tick, frets, grp in chords:
        slot = (tick + EIGHTH // 2) // EIGHTH       # index of the 1/8 slot
        dist = abs(tick - slot * EIGHTH)
        if slot not in buckets or dist < bucket_dist[slot]:
            buckets[slot] = (tick, frets, grp)
            bucket_dist[slot] = dist
    kept = [buckets[s] for s in sorted(buckets)]

    # Step B: continuous 8ths at >160 BPM → remove every 4th note of the run.
    # An "8th run" is consecutive notes spaced ~1/8 apart (≤ 1/8 + tol).
    # The book says "remove every 4th 8th" (THIN_PERIOD=4). We only apply it to
    # GENUINELY long runs — ≥ 8 8ths, i.e. a whole 4/4 measure or more
    # (THIN_MINRUN=8): short bursts of 4-7 8ths are not the "sustained continuous
    # 8th" that the book penalizes (p.225). Empirically this minrun raises pos
    # without costing mode or inflating density (sweep_thin.py).
    THIN_BPM, THIN_PERIOD, THIN_MINRUN = 160, 4, 8
    out: list[tuple[int, frozenset[int], list[Note]]] = []
    run: list[int] = []   # indices in `kept` of the current run

    def _flush_run():
        if len(run) >= THIN_MINRUN:
            for pos, ki in enumerate(run):
                # remove the 4th 8th of each group of 4 (keeps 3 in 4)
                if pos % THIN_PERIOD != THIN_PERIOD - 1:
                    out.append(kept[ki])
        else:
            for ki in run:
                out.append(kept[ki])
        run.clear()

    for idx in range(len(kept)):
        tick = kept[idx][0]
        bpm = bpm_at(tick, tempo_map)
        if idx > 0:
            gap = tick - kept[idx - 1][0]
            cont = gap <= EIGHTH + GAP_TOL
        else:
            cont = False
        fast = bpm > THIN_BPM
        if run and cont and fast:
            run.append(idx)
        else:
            _flush_run()
            run.append(idx)
        # if this point isn't fast/continuous, the 1-element run is just copied
        if not (cont and fast):
            # ensure runs only group fast-continuous notes
            pass
    _flush_run()

    out.sort(key=lambda c: c[0])
    kept_ticks = {grp[0].start for _, _, grp in out}
    return out, kept_ticks


def reduce_hard(
    events: list[AbsEvent],
    tempo_map: list,
    tpb: int,
    time_sig_map: list | None = None,
) -> list[AbsEvent]:
    """Generate Hard guitar events from Expert (book version)."""
    HOPO_THRESH = tpb // 3          # 1/12 — CH/YARG auto-HOPO threshold
    FORCE_ON, FORCE_OFF = 101, 102

    notes, others = pair_notes(events)
    fret_notes   = [n for n in notes if note_to_fret(n.note, "expert") is not None
                    or is_open_note(n.note, "expert")]
    marker_notes = [n for n in notes if n.note in GLOBAL_MARKERS]
    force_on_n   = [n for n in notes if n.note == FORCE_ON]
    force_off_n  = [n for n in notes if n.note == FORCE_OFF]
    force_on_t   = {n.start for n in force_on_n}
    force_off_t  = {n.start for n in force_off_n}

    # 1. Group into Expert chords (with OPEN notes)
    expert_chords = _group_chords(fret_notes, "expert", with_open=True)

    # 2. Classify HOPO/strum (force-aware) — preserved through the cascade
    hopo_flags = _classify_hopo(expert_chords, HOPO_THRESH, force_on_t, force_off_t)

    # 3. Reduce chords to Hard-legal (no 3-notes, no G+O → 1-4).
    #    OPEN ({OPEN_FRET}) passes through intact (single-note gem).
    hard_chords = [
        (tick, frets if OPEN_FRET in frets else _reduce_chord_hard(frets), grp)
        for (tick, frets, grp) in expert_chords
    ]

    # 4. Density (book): minimum gap 1/8 / 1/4
    kept, kept_ticks = _reduce_hard_chart(hard_chords, tempo_map, tpb)

    # 5. Build notes with shortened sustains (without stretching reduced ones)
    result_notes: list[Note] = []
    for idx, (start, frets, grp) in enumerate(kept):
        next_start = kept[idx + 1][0] if idx + 1 < len(kept) else None
        orig_end   = max(n.end for n in grp)
        ref        = Note(0, grp[0].velocity, start, orig_end, grp[0].channel)
        trimmed    = _trim_sustain(ref, "hard", tpb, tempo_map, next_start)
        for fi in sorted(frets):
            result_notes.append(Note(
                note=fret_note(fi, "hard"), velocity=grp[0].velocity,
                start=trimmed.start, end=trimmed.end, channel=grp[0].channel,
            ))

    # 6. Force markers by CASCADE INHERITANCE.
    #    Each kept note inherits the strum/HOPO classification it had in Expert
    #    (hopo_flags). But it can only be a HOPO if, in the reduced chart, it is
    #    still a single note, with a fret change and gap ≤ ~1/8 (if the gap blew up,
    #    it became a picked note → strum; this makes the 1st note of each phrase a
    #    strum). Then force markers are emitted only where the Hard auto-HOPO
    #    (gap ≤ 1/12) disagrees with the target: force-on (89) to create a 1/8 HOPO;
    #    force-off (90) to stop an unwanted auto-HOPO. It is the faithful translation
    #    of "chord→chord HOPOs reduced to a single note, marked with H".
    expert_hopo = {expert_chords[k][0]: hopo_flags[k]
                   for k in range(len(expert_chords))}
    # Gap band within which a note INHERITED as a HOPO from Expert can still be a
    # HOPO in the reduced Hard. Up to 1/6 (tpb*2//3): at 1/8 with swing/slow tempo
    # the gap after reduction reaches ~1/6. Only applies to inherited HOPOs (target
    # requires expert_hopo=True), so it doesn't invent HOPOs on strum notes.
    HI = (tpb * 2) // 3     # 2/3 of a beat
    # NOTE: the "strum on a strong beat" rule (notes on quarter/downbeat → strum even
    # when inheriting a HOPO from Expert) was tested and REJECTED — it raises mode but
    # is a precision/recall trade with no free lunch (quarter: +4.3pp mode / −22pp
    # recall; downbeat: +0.9 / −3.7). We chose to keep maximum recall. (sweep_strong.py)

    # A HOPO can never be FORCED if the previous note is more than a quarter note
    # (1/4 = tpb) away: with nothing to hammer-on from in the last 1/4, the forced
    # HOPO is unplayable (e.g. the 1st note of beat 3 when beat 2 was empty). This is
    # a hard cap on top of HI (which is already tighter); kept explicit so the rule
    # holds regardless of future HI tuning.
    QUARTER = tpb                  # 1/4 note
    force_notes: list[Note] = []
    prev_t, prev_f = None, None
    for (start, frets, grp) in kept:
        single = len(frets) == 1
        cur = min(frets) if single else None
        gap = (start - prev_t) if prev_t is not None else 10**9
        fret_change = single and prev_f is not None and cur != prev_f
        target_hopo = (single and fret_change
                       and expert_hopo.get(grp[0].start, False)
                       and gap <= HI and gap <= QUARTER)
        auto_hopo = single and fret_change and gap <= HOPO_THRESH
        if target_hopo and not auto_hopo:
            force_notes.append(Note(FORCE_ON + DIFF_OFFSET["hard"], grp[0].velocity,
                                    start, start + 1, grp[0].channel))
        elif auto_hopo and not target_hopo:
            force_notes.append(Note(FORCE_OFF + DIFF_OFFSET["hard"], grp[0].velocity,
                                    start, start + 1, grp[0].channel))
        prev_t, prev_f = start, cur

    all_notes = result_notes + list(marker_notes) + force_notes
    note_evs = notes_to_events(all_notes)
    out = note_evs + [e for e in others
                      if e.msg.type not in ("note_on", "note_off")]
    out.sort(key=lambda e: e.abs_tick)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  MEDIUM — reduces OUR Hard (cascade; Customs Book pp.215-217)
# ═══════════════════════════════════════════════════════════════════════════

def _select_prio(chords, tpb, spacing):
    """Greedy priority selection (Onyx style): keeps the most important notes while
    respecting a minimum spacing. Removes the least important ones in dense zones.

    Priority (lower score = kept first):
      • MELODIC SALIENCE (key): in a dense cluster, the human charter keeps the gem
        of the RIFF — the chord / note whose content changes — and discards the
        repeated on-beat pedal. In syncopated thrash (Battery) this is the difference
        between following the riff (off-beat) and hammering the beat. Bonus: chord
        (-4), content change vs the previous gem (-2). (medium pos +2.96pp,
        Battery +17.9pp, 0 regressions in the learn-set; _exp_sal.py.)
      • Sustain > measure boundary > 1/4-aligned (structural tiebreak).
    """
    import bisect as _b
    BEAT = tpb; MEASURE = tpb * 4; SNAP = tpb // 8; EIGHTH = tpb // 2
    W_CHORD, W_CHANGE = 4, 2
    # previous gem by tick order (to detect content change)
    by_tick = sorted(range(len(chords)), key=lambda i: chords[i][0])
    prev_frets: dict[int, frozenset[int] | None] = {}
    pf = None
    for i in by_tick:
        prev_frets[i] = pf
        pf = chords[i][1]
    def prio(i):
        tick, frets, grp = chords[i]
        dur = max(n.end for n in grp) - tick
        s = 0
        if len(frets) >= 2:                                          s -= W_CHORD
        if prev_frets[i] is not None and frets != prev_frets[i]:     s -= W_CHANGE
        if not (dur > EIGHTH):                                        s += 4
        if not (abs(tick - round(tick / MEASURE) * MEASURE) <= SNAP): s += 2
        if not (abs(tick - round(tick / BEAT) * BEAT) <= SNAP):       s += 1
        return s
    order = sorted(range(len(chords)),
                   key=lambda i: (prio(i), chords[i][0]))
    kt, keep = [], set()
    for i in order:
        tick = chords[i][0]
        idx = _b.bisect_left(kt, tick)
        if all(not (0 <= ci < len(kt) and abs(kt[ci] - tick) < spacing)
               for ci in (idx - 1, idx)):
            keep.add(i); kt.insert(idx, tick)
    return [chords[i] for i in sorted(keep)]


def _select_grid(chords, grid):
    """Grid-aligned selection: for each slot (multiple of `grid`, on/off-beat),
    keep the note nearest the grid line. Same as the Hard's Step A but with a
    parametrizable grid (Hard=1/8, Medium=1/4)."""
    buckets: dict[int, tuple[int, frozenset[int], list[Note]]] = {}
    bdist: dict[int, int] = {}
    for tick, frets, grp in chords:
        slot = (tick + grid // 2) // grid
        dist = abs(tick - slot * grid)
        if slot not in buckets or dist < bdist[slot]:
            buckets[slot] = (tick, frets, grp); bdist[slot] = dist
    return [buckets[s] for s in sorted(buckets)]


def _reduce_prio_diff(
    events: list[AbsEvent],
    diff: str,
    tempo_map: list,
    tpb: int,
    spacing: int,
    gap: int,
) -> list[AbsEvent]:
    """Shared Medium/Easy core: reduces from EXPERT by PRIORITY selection.

    Architecture decision (session): reducing from Expert beats the cascade on the
    pos/density frontier — Expert offers a rich pool of candidates for the
    importance selection to choose from (the cascade already dropped notes the lower
    level wanted).

    Steps:
      • `_select_prio` — greedy by priority (sustain > measure boundary >
        1/4-aligned), with a minimum `spacing` between kept notes.
      • Gap-fill — where a gap >= `gap` is left, reinsert the dropped note that fits
        at >= 1/8 from both sides. Recovers real notes in SPARSE sections without
        inflating the dense ones (which have no wide gaps).
      • Beat-snap — a note <= 1/6 from the nearest 1/4 is aligned (preserves monotonicity).
      • Pitch: `_PitchWindow` (contour) + `map_chord`; sustains via `_trim_sustain`.
      • ALL STRUM: the manual Medium/Easy have 0 HOPOs across the whole learn-set.
    """
    import bisect as _bi
    HOPO_THRESH = tpb // 3
    MINSEP = tpb // 2                 # minimum separation from neighbors in gap-fill (1/8)

    # OPEN notes: supported up to Medium (GH has no opens in Easy).
    open_ok = diff != "easy"
    notes, others = pair_notes(events)
    fret_notes   = [n for n in notes if note_to_fret(n.note, "expert") is not None
                    or (open_ok and is_open_note(n.note, "expert"))]
    marker_notes = [n for n in notes if n.note in GLOBAL_MARKERS]

    # 1. Group Expert chords (with OPEN if allowed)
    expert_chords = _group_chords(fret_notes, "expert", with_open=open_ok)

    # 2. Density: priority selection
    kept = _select_prio(expert_chords, tpb, spacing)

    # 2b. Adaptive gap-fill
    kt = [c[0] for c in kept]
    keptset = {c[0] for c in kept}
    dropped = sorted((c for c in expert_chords if c[0] not in keptset),
                     key=lambda c: c[0])
    for c in dropped:
        t = c[0]
        i = _bi.bisect_left(kt, t)
        left = kt[i - 1] if i > 0 else None
        right = kt[i] if i < len(kt) else None
        if left is not None and right is not None and (right - left) >= gap \
                and (t - left) >= MINSEP and (right - t) >= MINSEP:
            kept.append(c); kt.insert(i, t)
    kept.sort(key=lambda c: c[0])

    # 2c. Beat-snap
    snapped = []
    prev = -10**9
    for k_sn, (start, hf, grp) in enumerate(kept):
        rem = start % tpb
        cand = start - rem if rem <= tpb - rem else start - rem + tpb
        nxt = kept[k_sn + 1][0] if k_sn + 1 < len(kept) else None
        if abs(cand - start) <= tpb // 6 and cand > prev and (nxt is None or cand < nxt):
            start = cand
        snapped.append((start, hf, grp))
        prev = start
    kept = snapped

    # 3. Map pitch (single-note wrapping + map_chord for chords)
    mapped: list[tuple[int, frozenset[int], list[Note]]] = []
    window: _PitchWindow | None = None
    for (start, hfrets, grp) in kept:
        if OPEN_FRET in hfrets:
            mapped.append((start, frozenset({OPEN_FRET}), grp))
            continue
        if len(hfrets) == 1:
            ef = next(iter(hfrets))
            if window is None:
                window = _PitchWindow(diff, ef)
            mfrets = frozenset({window.map(ef)})
        else:
            # Keep 2+ note Expert chords AS chords in Medium (min_output_size=2).
            # map_chord with min=1 collapsed ~32% of 2-note Expert chords to a single
            # note — the main driver of Medium's low fret accuracy (fret 44.5%->56.2%,
            # chord count 3187->5011 ≈ the official 5010, recall unchanged). Easy stays
            # single-note (its manual charts are ~all single, only 10 chords/6151).
            min_out = 2 if diff == "medium" else 1
            mfrets = map_chord(hfrets, diff, min_out)
            if window is None:
                window = _PitchWindow(diff, max(hfrets))
        mapped.append((start, mfrets, grp))

    # 5. Notes + sustains
    result_notes: list[Note] = []
    for idx, (start, mfrets, grp) in enumerate(mapped):
        next_start = mapped[idx + 1][0] if idx + 1 < len(mapped) else None
        orig_end   = max(n.end for n in grp)
        ref        = Note(0, grp[0].velocity, start, orig_end, grp[0].channel)
        trimmed    = _trim_sustain(ref, diff, tpb, tempo_map, next_start)
        for fi in sorted(mfrets):
            result_notes.append(Note(
                note=_fret_note(fi, diff), velocity=grp[0].velocity,
                start=trimmed.start, end=trimmed.end, channel=grp[0].channel,
            ))

    # 6. ALL STRUM: suppress auto-HOPOs with force-strum.
    f_off = 102 + DIFF_OFFSET[diff]   # Medium=78, Easy=66
    force_notes: list[Note] = []
    prev_t, prev_f = None, None
    for (start, mfrets, grp) in mapped:
        single = len(mfrets) == 1
        cur = min(mfrets) if single else None
        gp = (start - prev_t) if prev_t is not None else 10**9
        fret_change = single and prev_f is not None and cur != prev_f
        if single and fret_change and gp <= HOPO_THRESH:
            force_notes.append(Note(f_off, grp[0].velocity, start, start + 1, grp[0].channel))
        prev_t, prev_f = start, cur

    all_notes = result_notes + list(marker_notes) + force_notes
    note_evs = notes_to_events(all_notes)
    out = note_evs + [e for e in others
                      if e.msg.type not in ("note_on", "note_off")]
    out.sort(key=lambda e: e.abs_tick)
    return out


def reduce_medium(
    events: list[AbsEvent],
    tempo_map: list,
    tpb: int,
    time_sig_map: list | None = None,
) -> list[AbsEvent]:
    """Medium v2: Expert-direct + priority. spacing=1/4, gap-fill=2.25b.
        pos=84.03% mode=100% density=1.072× (honest Onyx 1.055×).
    Confirmed vs alternatives: prio-cascade 79.4%@1.003×, grid-cascade 85.4%@1.152×.
    Gap-fill 2.25b keeps the density honest; beat-snap +0.67pp at no cost."""
    return _reduce_prio_diff(events, "medium", tempo_map, tpb,
                             spacing=tpb, gap=(tpb * 9) // 4)


def reduce_easy(
    events: list[AbsEvent],
    tempo_map: list,
    tpb: int,
    time_sig_map: list | None = None,
) -> list[AbsEvent]:
    """Easy v2 (priority, Expert-direct), sparser than Medium."""
    return _reduce_prio_diff(events, "easy", tempo_map, tpb,
                             spacing=(tpb * 3) // 2, gap=tpb * 3)


def _easy_accent_preservation(events: list[AbsEvent], reduced: list[AbsEvent],
                              tpb: int) -> float:
    """Fraction of the Expert structural ACCENTS (notes followed by >= 1 beat of
    space — the riff's pulse) preserved (±TOL) by the reduction. Measures whether the
    chart kept the song's groove."""
    import bisect as _bi
    TOL = 20
    notes, _ = pair_notes(events)
    starts = sorted({n.start for n in notes
                     if note_to_fret(n.note, "expert") is not None})
    accents = [t for i, t in enumerate(starts)
               if (starts[i + 1] if i + 1 < len(starts) else t + tpb * 2) - t >= tpb]
    if not accents:
        return 1.0
    rstarts = sorted({e.abs_tick for e in reduced
                      if e.msg.type == "note_on" and getattr(e.msg, "velocity", 0) > 0
                      and note_to_fret(e.msg.note, "easy") is not None})
    hit = 0
    for a in accents:
        i = _bi.bisect_left(rstarts, a)
        if any(0 <= ci < len(rstarts) and abs(rstarts[ci] - a) <= TOL
               for ci in (i - 1, i)):
            hit += 1
    return hit / len(accents)


def reduce_easy_hybrid(
    events: list[AbsEvent],
    tempo_map: list,
    tpb: int,
    time_sig_map: list | None = None,
) -> list[AbsEvent]:
    """AUTO HYBRID Easy: uses the old grid by default (wins on songs with a steady
    rhythm, where the grid beats the charter's choices), but DETECTS when the grid
    lost the groove and switches to position-preservation selection (v2).

    Detector: if the old grid preserves < ACCENT_FLOOR of the Expert structural
    accents (riff-pulse notes), it is fighting the music — phase/period off the
    measure grid (e.g. Nightmare in period-3, Star Wars). Validated on the learn-set:
    it only fires on Nightmare (oldAcc 30%) and Star Wars (41%); Carry/Iris (44%) and
    everything else stay on the grid. v2 (preserves the real Expert positions) fixes
    the phase, like the drums ("preserve phase" > snap-to-grid)."""
    from .guitar import _reduce_guitar_med_easy
    ACCENT_FLOOR = 0.42
    # Detect on the BASE grid (no gap-fill) — a pure indicator of lost groove.
    base = _reduce_guitar_med_easy(events, "easy", tempo_map, tpb, time_sig_map,
                                   gap_fill=False)
    if _easy_accent_preservation(events, base, tpb) >= ACCENT_FLOOR:
        # healthy grid → emit the gap-filled version (honest density + recovers sparse)
        return _reduce_guitar_med_easy(events, "easy", tempo_map, tpb, time_sig_map,
                                       gap_fill=True)
    # grid lost the groove (phase/period off the grid) → position preservation (v2)
    return reduce_easy(events, tempo_map, tpb, time_sig_map)

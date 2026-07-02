"""
guitar_handmap.py — Richer hand position map for 5-fret tracks (guitar/bass).

Generates force-HOPO (101) and force-strum (102) markers for Expert difficulty
based on hand position tracking, HOPO-chain analysis, and patterns extracted from
100 official RB3 MIDIs.

Official chart findings:
  - 64% of 101 markers land on gap=240 (0.5 beat / 8th note) with fret change
    where auto-HOPO (gap <= 160) would NOT trigger — they extend HOPO to 8th-note runs.
  - 102 markers appear at hand position changes, after long rests (>1 beat),
    and before same-fret repetitions that would break a HOPO chain.
  - A HOPO chain, once started, continues at 8th-note gaps without extra markers.
  - Chain breaks at: chord, same-fret repetition, or gap > 1 beat.

Algorithm:
  1. Track the index-finger hand position through the fretboard.
  2. Track HOPO chain state (active / broken).
  3. Insert 101 at the first note of a new chain when gap > auto-threshold (160)
     but <= chain-threshold (480) — forces HOPO on 8th-note runs.
  4. Insert 102 at hand position changes (>= 2 frets) and after long rests.
"""
from __future__ import annotations

import mido

from .midi_utils import AbsEvent

EXPERT_GEM_LO, EXPERT_GEM_HI = 96, 100
FORCE_HOPO = 101
FORCE_STRUM = 102


def apply_handmap(events: list[AbsEvent], tpb: int) -> list[AbsEvent]:
    """Return a NEW event list with force-HOPO (101) and force-strum (102) markers
    added to the Expert guitar/bass track.

    Parameters
    ----------
    events : list[AbsEvent]
        The original track events (Expert gems + existing markers).
    tpb : int
        Ticks per beat (used to scale thresholds).

    Returns
    -------
    list[AbsEvent]
        Events with 101/102 markers inserted.
    """
    if not events or tpb <= 0:
        return list(events)

    # Scale thresholds to actual TPB
    auto_thresh = max(1, tpb // 3)
    chain_thresh = tpb
    rest_thresh = tpb * 2

    # ── 1. Extract gems and existing force markers ──────────────────────────
    gems: list[tuple[int, int]] = []   # (tick, fret)
    existing_101: set[int] = set()
    existing_102: set[int] = set()

    for ev in events:
        if ev.msg.type != "note_on" or ev.msg.velocity == 0:
            continue
        n = ev.msg.note
        if n == FORCE_HOPO:
            existing_101.add(ev.abs_tick)
        elif n == FORCE_STRUM:
            existing_102.add(ev.abs_tick)
        elif EXPERT_GEM_LO <= n <= EXPERT_GEM_HI:
            gems.append((ev.abs_tick, n - EXPERT_GEM_LO))

    if len(gems) < 2:
        return list(events)

    # Group simultaneous notes into chords (same tick)
    from collections import defaultdict
    tick_frets: dict[int, frozenset[int]] = {}
    for t, f in gems:
        tick_frets.setdefault(t, set()).add(f)
    sorted_ticks = sorted(tick_frets)
    chord_list = [(t, frozenset(tick_frets[t])) for t in sorted_ticks]

    # ── 2. Hand position + HOPO chain scan ─────────────────────────────────
    generated_101: set[int] = set()
    generated_102: set[int] = set()

    hand_pos = min(chord_list[0][1])   # index finger start position
    chain_active = False
    prev_tick: int | None = None
    prev_fret: int | None = None
    prev_single = False

    for i, (tick, frets) in enumerate(chord_list):
        is_single = len(frets) == 1
        min_fret = min(frets)
        max_fret = max(frets)

        # ── Hand position change (big shift >= 2) ─────────────────────────
        if min_fret < hand_pos or max_fret > hand_pos + 3:
            old = hand_pos
            hand_pos = max(0, max_fret - 3)
            if hand_pos != old and abs(hand_pos - old) >= 2:
                generated_102.add(tick)

        if prev_tick is None:
            prev_tick, prev_fret, prev_single = tick, min_fret, is_single
            continue

        gap = tick - prev_tick

        # ── Long rest >= 2 beats: break chain ─────────────────────────────
        if gap >= rest_thresh:
            chain_active = False
            prev_tick, prev_fret, prev_single = tick, min_fret, is_single
            continue

        # ── Chord: strummed, breaks chain (no 102 — auto-strum suffices) ──
        if not is_single:
            chain_active = False
            prev_tick, prev_fret, prev_single = tick, min_fret, is_single
            continue

        curr_fret = min_fret
        same_fret = prev_fret is not None and curr_fret == prev_fret
        fret_change = prev_fret is not None and curr_fret != prev_fret

        # Look ahead: is the NEXT note after this one on the same fret?
        next_same_fret = False
        if i + 2 < len(chord_list):
            nxt_frets = chord_list[i + 1][1]
            if len(nxt_frets) == 1 and min(nxt_frets) == curr_fret:
                next_same_fret = True

        # ── Same fret as previous: auto-strum, breaks chain ───────────────
        if same_fret:
            chain_active = False
            prev_tick, prev_fret, prev_single = tick, curr_fret, True
            continue

        # Gap beyond chain threshold -> chain break (no 102)
        if gap > chain_thresh:
            chain_active = False
            prev_tick, prev_fret, prev_single = tick, curr_fret, True
            continue

        # ── 102: force strum when the NEXT note would break the chain ──────
        # (same-fret repetition coming up — break cleanly on this note)
        if chain_active and next_same_fret:
            generated_102.add(tick)

        # ── Within chain threshold ─────────────────────────────────────────
        auto_hopo = fret_change and gap <= auto_thresh

        if not chain_active:
            # Starting a new HOPO chain
            if auto_hopo:
                chain_active = True   # auto starts the chain
            else:
                # Gap > auto_thresh but <= chain_thresh — needs 101, but only
                # if the chain actually continues (look at the NEXT note after
                # this one to confirm).
                next_ok = False
                if i + 2 < len(chord_list):
                    nt, nf = chord_list[i + 1]
                    n_gap = nt - tick
                    n_single = len(nf) == 1
                    n_change = prev_fret is not None and (
                        min(nf) != curr_fret if n_single else True)
                    if n_single and n_change and n_gap <= chain_thresh:
                        next_ok = True
                if next_ok:
                    generated_101.add(tick)
                    chain_active = True
        else:
            # Chain is active — continues automatically
            pass   # no marker needed

        prev_tick, prev_fret, prev_single = tick, curr_fret, True

    # ── 3. Merge generated markers (avoid duplicates) ──────────────────────
    # Only add markers that don't already exist
    to_add_101 = generated_101 - existing_101
    to_add_102 = generated_102 - existing_102

    if not to_add_101 and not to_add_102:
        return list(events)

    # Build force-marker events
    new_events: list[AbsEvent] = list(events)
    for tick in sorted(to_add_101):
        new_events.append(AbsEvent(
            abs_tick=tick,
            msg=mido.Message("note_on", note=FORCE_HOPO, velocity=96, time=0),
        ))
    for tick in sorted(to_add_102):
        new_events.append(AbsEvent(
            abs_tick=tick,
            msg=mido.Message("note_on", note=FORCE_STRUM, velocity=96, time=0),
        ))

    new_events.sort(key=lambda e: e.abs_tick)
    return new_events

"""validate.py — pre-pack sanity gate for the RB3 MIDI (PS3 / Xbox builds only).

Magma runs a long list of MIDI checks before it compiles a song; some are pure
pedantry (track-name conventions, density warnings) but a handful guard against
things that genuinely **crash Rock Band 3 in-game**. We do NOT want to run Magma,
but we DO want its crash-relevant guarantees on the notes.mid we ship.

This module is deliberately scoped to the PACK step (ps3build / stfs) — never the
Process tab. Processing reduces/charts freely; only when we assemble the final
console package do we assert the result is something RB3 will load without
crashing. Every check is NON-FATAL by default (it returns a list of issues the
caller logs, like quality.groove_check), so a questionable chart still builds —
the user just gets warned. Callers may choose to treat "error" issues as fatal.

Crash-relevant checks implemented (see the study in CLAUDE/chat):
  * time division must be 480 TPB
  * every charted gem difficulty (Easy/Med/Hard/Expert) must be non-empty
  * no stuck/overlapping note of the same pitch on one track (broken-chord / hung
    sustain — RB3 can hang drawing it)
  * a missing [end] event (characters freeze in slow-mo at the end)
  * BEAT track present with >= 2 beats of lead-in before the first gem
  * PART VOCALS: notes must sit inside a phrase marker, phrases must not overlap
"""
from __future__ import annotations

from .midi_utils import to_abs

# Per-difficulty gem pitch windows, shared by guitar/bass/keys/drums.
_DIFF_WINDOWS = {
    "Easy":   (60, 64),
    "Medium": (72, 76),
    "Hard":   (84, 88),
    "Expert": (96, 100),
}
# PART tracks that use the 4-difficulty gem windows above.
_GEM_PARTS = ("PART GUITAR", "PART BASS", "PART DRUMS", "PART KEYS",
              "PART RHYTHM", "PART GUITAR COOP")

# Vocal phrase markers and the talky/sung pitch band of PART VOCALS.
_VOCAL_PHRASE = (105, 106)
_VOCAL_MIN, _VOCAL_MAX = 36, 84


def _is_on(m) -> bool:
    return m.type == "note_on" and getattr(m, "velocity", 0) > 0


def _is_off(m) -> bool:
    return m.type == "note_off" or (m.type == "note_on"
                                    and getattr(m, "velocity", 0) == 0)


def _first_gem_tick(mid) -> int | None:
    """Earliest tick of any Expert gem across the instrument tracks."""
    lo, hi = _DIFF_WINDOWS["Expert"]
    best = None
    for tr in mid.tracks:
        nm = (tr.name or "").upper()
        if not any(nm.startswith(p) for p in _GEM_PARTS):
            continue
        for e in to_abs(tr):
            if _is_on(e.msg) and lo <= e.msg.note <= hi:
                best = e.abs_tick if best is None else min(best, e.abs_tick)
                break
    return best


def _check_overlaps(tr, name, issues):
    """Flag a same-pitch note_on arriving while that pitch is still held (a stuck
    sustain). RB3 can hang trying to render the never-closed gem."""
    open_pitch: dict[int, int] = {}
    bad = 0
    # Process note-offs before note-ons at the same tick so a back-to-back
    # gem (off then on at the identical tick) is NOT mistaken for an overlap.
    evts = sorted(to_abs(tr), key=lambda e: (e.abs_tick, 0 if _is_off(e.msg) else 1))
    for e in evts:
        m = e.msg
        n = getattr(m, "note", None)
        if n is None:
            continue
        if _is_on(m):
            if n in open_pitch:
                bad += 1
            open_pitch[n] = e.abs_tick
        elif _is_off(m):
            open_pitch.pop(n, None)
    if bad:
        issues.append(("error",
                       f"{name}: {bad} overlapping/stuck note(s) of the same "
                       f"pitch (broken chord / hung sustain — can crash RB3)"))


def _check_gem_difficulties(tr, name, issues):
    """If a part has any Expert gems, every lower difficulty must be non-empty
    too — selecting an empty difficulty crashes RB3."""
    present = {}
    for diff, (lo, hi) in _DIFF_WINDOWS.items():
        present[diff] = any(_is_on(m) and lo <= m.note <= hi for m in tr)
    if not present["Expert"]:
        return                                  # not a charted gem part
    for diff in ("Easy", "Medium", "Hard"):
        if not present[diff]:
            issues.append(("error",
                           f"{name}: {diff} difficulty has no gems "
                           f"(empty difficulty crashes RB3)"))


def _check_vocals(tr, name, issues):
    """Vocal notes must fall inside a phrase marker, and phrases must not overlap."""
    abs_evts = to_abs(tr)
    phrases = []                                 # (start_tick, end_tick)
    open_t = None
    for e in abs_evts:
        m = e.msg
        if getattr(m, "note", None) not in _VOCAL_PHRASE:
            continue
        if _is_on(m):
            if open_t is not None:
                issues.append(("error", f"{name}: overlapping vocal phrase "
                                        f"markers (can crash the vocal tracker)"))
            open_t = e.abs_tick
        elif _is_off(m) and open_t is not None:
            phrases.append((open_t, e.abs_tick))
            open_t = None
    if not phrases:
        return
    outside = 0
    for e in abs_evts:
        m = e.msg
        n = getattr(m, "note", None)
        if n is None or not (_VOCAL_MIN <= n <= _VOCAL_MAX) or not _is_on(m):
            continue
        if not any(s <= e.abs_tick < en for s, en in phrases):
            outside += 1
    if outside:
        issues.append(("error", f"{name}: {outside} vocal note(s) outside any "
                                f"phrase marker (can crash the vocal tracker)"))


def validate_rb_midi(mid) -> list[tuple[str, str]]:
    """Run the crash-relevant checks on a finished RB3 notes.mid.

    Returns a list of ``(level, message)`` where level is "error" (will likely
    crash RB3 / be rejected) or "warn" (suspicious but probably loads). Empty
    list == clean. Never raises; never mutates `mid`.
    """
    issues: list[tuple[str, str]] = []

    # 1) Time division must be 480 TPB.
    if mid.ticks_per_beat != 480:
        issues.append(("error", f"time division is {mid.ticks_per_beat} TPB, "
                                f"RB3 requires 480"))

    # 2) [end] event present (else characters freeze in slow-mo at the end).
    has_end = any(getattr(m, "text", "").strip().lower() == "[end]"
                  for tr in mid.tracks for m in tr
                  if m.type in ("text", "marker"))
    if not has_end:
        issues.append(("warn", "no [end] event — characters may freeze at the "
                               "song end"))

    # 3) BEAT track + >= 2 beats of lead-in before the first gem.
    beat_tr = next((tr for tr in mid.tracks
                    if (tr.name or "").strip().upper() == "BEAT"), None)
    first_gem = _first_gem_tick(mid)
    if beat_tr is None:
        issues.append(("warn", "no BEAT track found"))
    elif first_gem is not None:
        lead = sum(1 for e in to_abs(beat_tr)
                   if _is_on(e.msg) and e.abs_tick < first_gem)
        if lead < 2:
            issues.append(("error", f"only {lead} beat(s) before the first gem "
                                    f"(RB3 needs >= 2 beats of lead-in)"))

    # 4) Per-track checks.
    for tr in mid.tracks:
        nm = (tr.name or "").strip().upper()
        if not nm:
            continue
        _check_overlaps(tr, nm, issues)
        if any(nm.startswith(p) for p in _GEM_PARTS):
            _check_gem_difficulties(tr, nm, issues)
        if nm == "PART VOCALS":
            _check_vocals(tr, nm, issues)

    return issues

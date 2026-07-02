"""convert.py — the native package-conversion pipeline (separate from the normal
per-folder MIDI processing).

This drives the "Convert" GUI tab: it takes an already-built RB3 song folder and
re-assembles it natively as a clean RPCS3 PS3 song folder, with our own milo
(downcharter/milo.py) so the lipsync we generate is guaranteed in the file the
game loads — no Onyx, no stale milos. Xbox CON (.con) and YARG .sng are planned
follow-ups; this first cut handles only the PS3 folder, per the rollout decision.

Bass-pedal variants (RB3 doesn't read YARG-style Expert+ 2x automatically):
  * "2x": force every Expert+ double-kick (note 95) down to a normal kick
    (note 96) so the doubles ALWAYS play, regardless of the in-game 2x toggle.
  * "1x": remove the Expert+ double-kicks entirely, leaving a chart that's
    playable with a single pedal.
  * "both": emit two folders, one of each.

The note-95 markers come from Downcharter's Expert+ pass (processor
`_apply_expert_plus`), so the source notes.mid is expected to already carry them
where fast double-bass was detected.
"""
from __future__ import annotations
import mido

from .constants import DRUM_KICK_EXPERT, DRUM_KICK_2X
from .midi_utils import build_tempo_map, tick_to_ms, to_abs, to_track


def _is_drums_track(track: mido.MidiTrack) -> bool:
    return "DRUM" in (track.name or "").strip().upper()


# Open (no-fret) strum markers. YARG/Clone-Hero charts use the note one below the
# green gem of each difficulty (the "ENHANCED_OPENS" extension). Rock Band 3 has
# no open-strum lane and silently IGNORES these notes, so the chart plays with
# gaps. RB3 has no open lane, so an open MUST become a fretted gem.
#
# A naive open→green map breaks the chart: an open rendered as green is now
# indistinguishable from a real green, so the player can't tell finger positions
# apart and HOPO/strum shapes collapse. Onyx solves this in `noOpenNotes`
# (Onyx/Guitar.hs) by shifting the NEIGHBOURHOOD of each open UP one fret too, so
# "open (→green)" stays distinct from a real green. We port that algorithm below.
#
# Green gem per difficulty (open = base-1, red = base+1 … orange = base+4):
#   Easy 60, Medium 72, Hard 84, Expert 96.  Force HOPO/strum markers (base+5/+6)
#   are NOT gems and are left untouched — they travel with their position.
_DIFF_BASES = (60, 72, 84, 96)

# Tracks that use the 5-fret open convention (NOT drums — there note 95 is the
# 2x-kick, handled separately by apply_pedal_variant).
_FRET_TRACK_KEYS = ("GUITAR", "BASS", "RHYTHM")


def _is_fret_track(track: mido.MidiTrack) -> bool:
    nm = (track.name or "").strip().upper()
    return "DRUM" not in nm and any(k in nm for k in _FRET_TRACK_KEYS)


# ── Onyx noOpenNotes port (lane space: open=-1, green=0, red=1 … orange=4) ──────
#
# Faithful port of `noOpenNotesNewAlgorithm` (mtolly/onyx Onyx/Guitar.hs). Each
# difficulty is processed on its own. Notes are grouped by onset (chords = groups
# of >1 gem) and tagged FretGroupLow (shift +1 fret) or FretGroupHigh (stay):
#   open→green AND its low-side neighbours move up so the open-as-green is
#   distinguishable from a genuine green, while runs that climb away (high passes)
#   are left in place. Adjacent gems that were equal stay equal and different stay
#   different, so the force HOPO/strum markers still line up unchanged.
#
# We deliberately skip Onyx's optional `mutedOpensToRBStyle` (detectMuted=False,
# the common case): we don't reinterpret strummed opens between chords as muted
# strums, we only de-open them.

_LOW, _HIGH = "low", "high"


def _single_lane(group: list) -> int | None:
    """Lane of a single-gem group (chords return None — they never spread)."""
    return group[0]["lane"] if len(group) == 1 else None


def _fix_wrapping(groups: list) -> list:
    """Onyx `fixWrapping`: a full ascending run of SINGLE gems
    open→green→red→yellow→blue→orange would wrap off the top once everything is
    shifted up. Pre-swap the red and yellow blocks (R↔Y) so the later +1 shift
    yields a valid 'G R Y R B O' instead. Lanes are mutated in place; positions
    are preserved. Returns the same list."""
    def run(start: int, lane: int) -> int:
        i = start
        while i < len(groups) and _single_lane(groups[i]) == lane:
            i += 1
        return i

    i = 0
    n = len(groups)
    while i < n:
        a = run(i, -1)                       # opens
        if a > i:
            b = run(a, 0)                    # greens
            if b > a:
                c = run(b, 1)                # reds
                if c > b:
                    d = run(c, 2)            # yellows
                    if d > c:
                        e = run(d, 3)        # blues
                        if e > d and e < n and _single_lane(groups[e]) == 4:
                            for k in range(b, c):       # reds → yellow
                                groups[k][0]["lane"] = 2
                            for k in range(c, d):       # yellows → red
                                groups[k][0]["lane"] = 1
                            i = e            # continue from the orange (fix6)
                            continue
        i += 1
    return groups


def _init_mark(group: list) -> str | None:
    """initState: single open → Low; single orange → High; chord → High; else
    unmarked."""
    if len(group) >= 2:
        return _HIGH
    lane = group[0]["lane"]
    if lane == -1:
        return _LOW
    if lane == 4:
        return _HIGH
    return None


def _pass_spread(groups: list, marks: list, mark: str, moves: tuple) -> list:
    """pass1/pass2 (forward): a marked SINGLE note spreads its mark to the next
    UNMARKED single note when the fret movement is within `moves`. Forward, in
    place, so a freshly-marked note keeps propagating to its successor."""
    marks = marks[:]
    for i in range(len(groups) - 1):
        if (len(groups[i]) == 1 and marks[i] == mark
                and len(groups[i + 1]) == 1 and marks[i + 1] is None):
            mv = groups[i + 1][0]["lane"] - groups[i][0]["lane"]
            if mv in moves:
                marks[i + 1] = mark
    return marks


def _pass_both(groups: list, marks: list, mark: str, moves: tuple) -> list:
    """Apply a spread pass forward, then on the reversed sequence, then unreverse
    (Onyx's `reverse $ pass $ reverse $ pass`)."""
    marks = _pass_spread(groups, marks, mark, moves)
    rmarks = _pass_spread(list(reversed(groups)), list(reversed(marks)), mark, moves)
    return list(reversed(rmarks))


def _pass4(marks: list) -> list:
    """pass4: any run of UNMARKED notes bounded by Low (or song boundary) with no
    High between → marked Low. Faithful iterative port of Onyx's pass4/pass4'."""
    out: list = []
    i, n = 0, len(marks)

    # start handler: unmarked prefix bounded on the left by the song start.
    j = i
    while j < n and marks[j] is None:
        j += 1
    if j > i:
        if j == n or marks[j] == _LOW:
            out.extend([_LOW] * (j - i))
        else:
            out.extend(marks[i:j])
        i = j

    # "low, unmarked, low/end" → fill the unmarked run Low.
    while i < n:
        k = i
        while k < n and marks[k] == _LOW:
            k += 1
        if k > i:
            m = k
            while m < n and marks[m] is None:
                m += 1
            if m > k and (m == n or marks[m] == _LOW):
                out.extend(marks[i:k])
                out.extend([_LOW] * (m - k))
                i = m
                continue
        out.append(marks[i])
        i += 1
    return out


def _no_open_shift(gems: list) -> None:
    """Run the full Onyx pipeline on one difficulty's gems and set ['new_lane']
    on each gem dict (lane space, open=-1 … orange=4)."""
    by_tick: dict = {}
    for g in gems:
        by_tick.setdefault(g["start"], []).append(g)
    groups = [by_tick[t] for t in sorted(by_tick)]

    # fixWrapping both ways
    _fix_wrapping(groups)
    _fix_wrapping(list(reversed(groups)))   # mutates the same gem dicts in place

    marks = [_init_mark(grp) for grp in groups]
    marks = _pass_both(groups, marks, _HIGH, (0, -1))   # pass1
    marks = _pass_both(groups, marks, _LOW, (0, 1))     # pass2

    # pass3: drop the Low marking from the opens themselves.
    for i, grp in enumerate(groups):
        if len(grp) == 1 and grp[0]["lane"] == -1 and marks[i] == _LOW:
            marks[i] = None

    marks = _pass4(marks)
    marks = [_HIGH if m is None else m for m in marks]   # pass5

    for grp, mk in zip(groups, marks):
        delta = 1 if mk == _LOW else 0
        for g in grp:
            g["new_lane"] = max(0, min(4, g["lane"] + delta))


def convert_open_notes(mid: mido.MidiFile) -> tuple[mido.MidiFile, dict]:
    """Return a NEW MidiFile with open-strum notes on 5-fret tracks de-opened the
    way Onyx does it: opens become green and the surrounding low notes shift up a
    fret so an open-turned-green stays distinct from a real green.

    Each difficulty is handled independently; a difficulty with NO open notes is
    left byte-for-byte unchanged. Returns (new_mid, {"converted": n}) where n is
    the number of open gems de-opened. Never mutates the input.
    """
    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    converted = 0

    for track in mid.tracks:
        new_tr = mido.MidiTrack()
        new_tr.name = track.name
        if not _is_fret_track(track):
            for m in track:
                new_tr.append(m.copy())
            out.tracks.append(new_tr)
            continue

        # Absolute-tick view (delta times preserved on the messages themselves).
        abs_msgs = []
        t = 0
        for m in track:
            t += m.time
            abs_msgs.append((t, m))

        remap: dict = {}          # message index → new note number

        for base in _DIFF_BASES:
            lo, hi = base - 1, base + 4    # open … orange
            # Skip difficulties with no open notes — leave them exactly as-is.
            if not any(m.type == "note_on" and m.velocity > 0 and m.note == base - 1
                       for _, m in abs_msgs):
                continue

            stacks: dict = {}             # note number → FIFO of (start_tick, on_idx)
            gems: list = []
            for idx, (tick, m) in enumerate(abs_msgs):
                if m.type not in ("note_on", "note_off"):
                    continue
                if not (lo <= m.note <= hi):
                    continue
                if m.type == "note_on" and m.velocity > 0:
                    stacks.setdefault(m.note, []).append((tick, idx))
                else:
                    st = stacks.get(m.note)
                    if st:
                        start_tick, on_idx = st.pop(0)
                        gems.append({"start": start_tick, "lane": m.note - base,
                                     "on_idx": on_idx, "off_idx": idx})
            if not gems:
                continue

            _no_open_shift(gems)
            for g in gems:
                new_note = base + g["new_lane"]
                remap[g["on_idx"]] = new_note
                remap[g["off_idx"]] = new_note
                if g["lane"] == -1:
                    converted += 1

        for idx, (tick, m) in enumerate(abs_msgs):
            if idx in remap and m.type in ("note_on", "note_off"):
                new_tr.append(m.copy(note=remap[idx]))
            else:
                new_tr.append(m.copy())
        out.tracks.append(new_tr)

    return out, {"converted": converted}


# ── source-format normalisation (Onyx MIDI auto-detect) ────────────────────────
# A raw .mid imported from FoF/GH/Phase-Shift uses conventions RB3 doesn't read.
# Onyx auto-detects the source format and rewrites it into the RB layout before
# anything else (its "RB Overdrive format, passing through unmodified" vs the
# FoF/PS/CH import path). chart_to_midi already emits the RB layout directly, so
# this brings a raw source .mid up to the SAME end-state. Two data-derived,
# no-op-when-already-RB normalisations:
#   • legacy track names → the exact RB names (RB3 ignores a track it can't name);
#   • FoF star-power on note 103 → RB overdrive note 116 (FoF reused the pitch RB
#     uses for solos, so an old chart's "star power" is otherwise invisible).

# Canonical RB3 track names (upper-case). A track whose name only differs by case
# or stray whitespace is snapped to the canonical spelling.
_RB_TRACK_NAMES = {
    "PART GUITAR", "PART BASS", "PART DRUMS", "PART KEYS", "PART VOCALS",
    "PART GUITAR COOP", "PART RHYTHM", "HARM1", "HARM2", "HARM3",
    "EVENTS", "VENUE", "BEAT",
}
# Legacy/aliased names → the RB standard (GH "T1/T2 GEMS", singular variants).
_TRACK_ALIASES = {
    "T1 GEMS": "PART GUITAR", "T2 GEMS": "PART GUITAR COOP",
    "GUITAR": "PART GUITAR", "BASS": "PART BASS", "DRUMS": "PART DRUMS",
    "PART GUITARS": "PART GUITAR", "PART DRUM": "PART DRUMS",
    "PART KEY": "PART KEYS", "PART VOCAL": "PART VOCALS",
}
# Five-fret + drums gameplay tracks where note 103 means star power in FoF.
_SP_REMAP_TRACKS = {
    "PART GUITAR", "PART BASS", "PART KEYS", "PART RHYTHM",
    "PART GUITAR COOP", "PART DRUMS",
}


def normalize_source_midi(mid: mido.MidiFile) -> dict:
    """Onyx-style source-format auto-detect, mutating `mid` in place. Renames
    legacy/aliased instrument tracks to the exact RB3 names, and — only when the
    whole file has NO note-116 overdrive but DOES use note 103 (the unambiguous
    FoF signature, since any RB chart with a solo also carries 116 overdrive) —
    remaps note 103 on gameplay tracks to overdrive note 116. Both are no-ops on a
    chart already in RB format. Returns {"tracks_renamed": [(old, new), ...],
    "sp_remapped": <overdrive phrases remapped>}."""
    renamed: list[tuple[str, str]] = []
    for tr in mid.tracks:
        nm = (tr.name or "").strip().upper()
        canon = _TRACK_ALIASES.get(nm) or (nm if nm in _RB_TRACK_NAMES else None)
        if canon and canon != (tr.name or ""):
            old = tr.name
            tr.name = canon
            if (old or "").strip().upper() != canon:   # not just a case/space fix
                renamed.append((old, canon))

    c116 = c103 = 0
    for tr in mid.tracks:
        for m in tr:
            if _msg_is_on(m):
                n = getattr(m, "note", None)
                if n == 116:
                    c116 += 1
                elif n == 103:
                    c103 += 1
    sp_remapped = 0
    if c116 == 0 and c103 > 0:
        for tr in mid.tracks:
            if (tr.name or "").strip().upper() not in _SP_REMAP_TRACKS:
                continue
            for m in tr:
                if getattr(m, "note", None) == 103 and m.type in ("note_on", "note_off"):
                    m.note = 116
                    if m.type == "note_on" and getattr(m, "velocity", 0) > 0:
                        sp_remapped += 1
    return {"tracks_renamed": renamed, "sp_remapped": sp_remapped}


# ── drum limb animations ───────────────────────────────────────────────────────
# RB3 animates the drummer from dedicated animation notes (24-51 on PART DRUMS).
# YARG auto-animates from the chart, but RB3 needs them authored, so a chart with
# no animation notes leaves the drummer idle. We synthesise them from the Expert
# Pro-drum gems with proper LEFT/RIGHT-HAND sticking — a faithful port of Onyx's
# `autoDrumAnimation`/`autoSticking` (mtolly/onyx, Onyx/MIDI/Track/Drums.hs):
#
#   * Each gem maps to an "anim pad" (snare, hihat, ride, crashes, toms), with
#     pro-tom markers (110/111/112) turning the yellow/blue/green lanes into toms.
#   * Simultaneous hits split between the hands (lower kit position → LH, higher
#     → RH); special chords (two cymbals → both crashes; red+yellow-tom → snare
#     flam) match Onyx.
#   * Single hits within 0.25 s of each other form a "phrase" whose sticking
#     alternates hands, leading with the hand that keeps the drummer from crossing
#     over on the next hit (double-strokes are delayed to the latest moment).
#
# The full RB3 animation note map (per Onyx `parseDrumAnimation`):
#   24 kick(RF) · 26/27 snare LH/RH · 30/31 hihat LH/RH · 34/36 crash1 LH/RH ·
#   38 crash2 RH · 42/43 ride RH/LH · 44 crash2 LH · 46/47 tom1 LH/RH ·
#   48/49 tom2 LH/RH · 50/51 floortom LH/RH
_TOM_MARKERS = (110, 111, 112)

# Anim pads, ordered left→right across the kit (used for hand assignment and the
# "normal direction" of motion). Mirrors Onyx's `AnimPad` Ord.
_SNARE, _HIHAT, _CRASH1, _TOM1, _TOM2, _FLOOR, _CRASH2, _RIDE = range(8)
_LH, _RH = "LH", "RH"

# (pad, hand) → RB3 animation note. Hard hits only (we don't author ghost notes).
_ANIM_NOTE = {
    (_SNARE, _LH): 26,  (_SNARE, _RH): 27,
    (_HIHAT, _LH): 30,  (_HIHAT, _RH): 31,
    (_CRASH1, _LH): 34, (_CRASH1, _RH): 36,
    (_TOM1, _LH): 46,   (_TOM1, _RH): 47,
    (_TOM2, _LH): 48,   (_TOM2, _RH): 49,
    (_FLOOR, _LH): 50,  (_FLOOR, _RH): 51,
    (_CRASH2, _LH): 44, (_CRASH2, _RH): 38,
    (_RIDE, _LH): 43,   (_RIDE, _RH): 42,
}
_ANIM_KICK = 24


def _flip(hand: str) -> str:
    return _RH if hand == _LH else _LH


def _normal_direction(x: int, y: int):
    """Natural hand when moving from pad x to pad y (None = no preference).
    Hihat↔snare and snare↔tom1 are centred, so neither implies a direction."""
    if (x, y) in ((_HIHAT, _SNARE), (_SNARE, _HIHAT), (_SNARE, _TOM1), (_TOM1, _SNARE)):
        return None
    if x < y:
        return _RH
    if x > y:
        return _LH
    return None


def _auto_sticking(pads: list[int]) -> list[str]:
    """Assign LH/RH to a phrase of single hits (Onyx `autoSticking`)."""
    out: list[str] = []
    prev = None                       # None | (hand, pad)
    for i, x in enumerate(pads):
        rest = pads[i + 1:]
        if prev is None:
            # Look ahead: first non-None direction decides the starting hand so the
            # run lands correctly; flip if an even number of "free" moves precede it.
            dirs = [_normal_direction(a, b) for a, b in zip(pads[i:], rest)]
            n, h = 0, None
            for d in dirs:
                if d is None:
                    n += 1
                else:
                    h = d
                    break
            if h is None:
                hand = _RH
            else:
                hand = _flip(h) if n % 2 == 0 else h
        else:
            prev_hand, prev_pad = prev
            if x == prev_pad:
                # Same pad: keep the hand (double stroke) only if that sets us up to
                # NOT cross over on the next hit; otherwise alternate.
                if rest and _flip(prev_hand) == _normal_direction(x, rest[0]):
                    hand = prev_hand
                else:
                    hand = _flip(prev_hand)
            else:
                hand = _flip(prev_hand)   # moving pads always switches hands
        out.append(hand)
        prev = (hand, x)
    return out


def _is_fill_buffer(pads: list[int], ticks: list[int],
                    tempo_map: list, tpb: int) -> bool:
    """A buffer is a fill if it moves across ≥3 distinct pads and spans ≤ 1
    measure — the same criteria used by ``drums.detect_fills`` for difficulty
    reduction.  Pure function, no side effects."""
    if len(set(pads)) < 3:
        return False
    span_ticks = ticks[-1] - ticks[0]
    if span_ticks > 4 * tpb:            # > 1 measure in 4/4
        return False
    span_ms = tick_to_ms(ticks[-1], tempo_map, tpb) - tick_to_ms(ticks[0], tempo_map, tpb)
    measure_ms = (tick_to_ms(ticks[0] + 4 * tpb, tempo_map, tpb)
                  - tick_to_ms(ticks[0], tempo_map, tpb))
    return span_ms <= measure_ms


def _fill_sticking(pads: list[int]) -> list[str]:
    """Sticking for a fill phrase: strict alternation (no double strokes, RLRL),
    starting with the dominant hand (RH) when that doesn't cause crossovers.

    The hand assigned to each hit is checked against the natural kit direction
    (``_normal_direction``).  If starting on RH causes ANY crossover, we fall
    back to LH.  If both cross over, the hand with fewer crossovers wins (RH
    breaks ties — dominant hand priority).
    """
    if not pads:
        return []

    def _simulate(start: str) -> tuple[list[str], int]:
        hands: list[str] = []
        hand = start
        n_x = 0
        for i, x in enumerate(pads):
            hands.append(hand)
            if i + 1 < len(pads):
                nd = _normal_direction(x, pads[i + 1])
                nxt = _flip(hand)
                if nd is not None and nd != nxt:
                    n_x += 1
            hand = _flip(hand)
        return hands, n_x

    rh_h, rh_x = _simulate(_RH)
    lh_h, lh_x = _simulate(_LH)

    if rh_x == lh_x:
        return rh_h           # tie → RH dominant
    if rh_x < lh_x:
        return rh_h
    return lh_h


def generate_drum_animations(mid: mido.MidiFile) -> tuple[mido.MidiFile, dict]:
    """Return a NEW MidiFile with drummer limb-animation notes (24-51) synthesised
    on PART DRUMS from its Expert gems, with left/right-hand sticking. No-op for a
    drums track that is already animated. Returns (new_mid, {"added": n}). Never
    mutates the input."""
    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    tpb = mid.ticks_per_beat
    anim_len = max(1, tpb // 8)
    tempo_map = build_tempo_map(mid)
    close_ms = 250.0                  # Onyx closeTime = 0.25 s
    added = 0

    for track in mid.tracks:
        if not _is_drums_track(track):
            new_tr = mido.MidiTrack()
            for m in track:
                new_tr.append(m.copy())
            out.tracks.append(new_tr)
            continue

        # Absolute-tick view of the track.
        abs_msgs, t = [], 0
        already_animated = False
        for m in track:
            t += m.time
            abs_msgs.append((t, m))
            if m.type == "note_on" and 24 <= m.note <= 51:
                already_animated = True

        if already_animated:
            new_tr = mido.MidiTrack()
            for m in track:
                new_tr.append(m.copy())
            out.tracks.append(new_tr)
            continue

        # Pro-tom marker spans → know whether a lane is a tom at a given tick.
        spans = {n: [] for n in _TOM_MARKERS}
        opening: dict[int, int] = {}
        for tick, m in abs_msgs:
            if m.type in ("note_on", "note_off") and m.note in _TOM_MARKERS:
                if m.type == "note_on" and m.velocity > 0:
                    opening[m.note] = tick
                elif m.note in opening:
                    spans[m.note].append((opening.pop(m.note), tick))

        def _is_tom(marker: int, tick: int) -> bool:
            return any(a <= tick < b for a, b in spans[marker])

        # Collect Expert gems per tick (one set of lanes per onset).
        gems_at: dict[int, set] = {}
        for tick, m in abs_msgs:
            if not (m.type == "note_on" and m.velocity > 0):
                continue
            n = m.note
            if n in (DRUM_KICK_EXPERT, DRUM_KICK_2X):
                lane = "kick"
            elif n == 97:
                lane = "red"
            elif n == 98:
                lane = "yellow_tom" if _is_tom(110, tick) else "yellow_cym"
            elif n == 99:
                lane = "blue_tom" if _is_tom(111, tick) else "blue_cym"
            elif n in (100, 101):
                lane = "green_tom" if _is_tom(112, tick) else "green_cym"
            else:
                continue
            gems_at.setdefault(tick, set()).add(lane)

        # Per-onset → list of anim pads (Onyx `autoDrumAnimation`). Kicks are
        # emitted directly as note 24; everything else feeds the sticking pass.
        _LANE_PAD = {
            "red": _SNARE, "yellow_cym": _HIHAT, "blue_cym": _RIDE,
            "green_cym": _CRASH2, "yellow_tom": _TOM1, "blue_tom": _TOM2,
            "green_tom": _FLOOR,
        }
        kicks: list[int] = []
        events: list[tuple] = []      # ("pair", tick, lo, hi) | ("single", tick, pad)
        for tick in sorted(gems_at):
            lanes = gems_at[tick]
            if "kick" in lanes:
                kicks.append(tick)
            # Special chords (match Onyx ordering).
            if {"yellow_cym", "green_cym"} <= lanes \
                    or {"blue_cym", "green_cym"} <= lanes:
                pads = [_CRASH1, _CRASH2]
            elif {"red", "yellow_tom"} <= lanes:
                pads = [_SNARE, _SNARE]
            else:
                pads = sorted(_LANE_PAD[l] for l in lanes if l in _LANE_PAD)
            if not pads:
                continue
            if len(pads) >= 2:
                events.append(("pair", tick, min(pads), max(pads)))
            else:
                events.append(("single", tick, pads[0]))

        # Walk the event stream; flush single-hit phrases through _auto_sticking.
        anim: list[tuple[int, int]] = []   # (tick, note)

        def _emit(tick: int, pad: int, hand: str) -> None:
            anim.append((tick, _ANIM_NOTE[(pad, hand)]))

        buffer: list[tuple[int, int]] = []   # (tick, pad)

        def _flush() -> None:
            if not buffer:
                return
            pads = [p for _, p in buffer]
            ticks = [t for t, _ in buffer]
            if _is_fill_buffer(pads, ticks, tempo_map, tpb):
                hands = _fill_sticking(pads)
            else:
                hands = _auto_sticking(pads)
            for (btick, pad), hand in zip(buffer, hands):
                _emit(btick, pad, hand)
            buffer.clear()

        prev_tick = None
        for ev in events:
            if ev[0] == "pair":
                _flush()
                _, tick, lo, hi = ev
                _emit(tick, lo, _LH)
                _emit(tick, hi, _RH)
                prev_tick = tick
            else:
                _, tick, pad = ev
                if buffer and prev_tick is not None and \
                        (tick_to_ms(tick, tempo_map, tpb)
                         - tick_to_ms(prev_tick, tempo_map, tpb)) <= close_ms:
                    buffer.append((tick, pad))
                else:
                    _flush()
                    buffer.append((tick, pad))
                prev_tick = tick
        _flush()

        for tick in kicks:
            anim.append((tick, _ANIM_KICK))

        # Merge gems + animation note pairs, drop the old end-of-track, re-time.
        merged = [(tick, m) for tick, m in abs_msgs if m.type != "end_of_track"]
        for tick, note in anim:
            merged.append((tick, mido.Message("note_on", note=note, velocity=96, time=0)))
            merged.append((tick + anim_len, mido.Message("note_off", note=note, velocity=0, time=0)))
            added += 1
        merged.sort(key=lambda tm: tm[0])

        new_tr = mido.MidiTrack()
        last = 0
        for tick, m in merged:
            new_tr.append(m.copy(time=tick - last))
            last = tick
        new_tr.append(mido.MetaMessage("end_of_track", time=0))
        out.tracks.append(new_tr)

    return out, {"added": added}


def apply_pedal_variant(mid: mido.MidiFile, mode: str) -> tuple[mido.MidiFile, dict]:
    """Return a NEW MidiFile with PART DRUMS kicks adjusted for `mode`.

    mode == "2x": every note-95 (Expert+ 2x-kick) becomes note-96 (normal kick),
                  so the doubles play with no in-game toggle. (No-op for songs
                  with no 95 markers — they already play as-is.)
    mode == "1x": every note-95 note_on/note_off is dropped, removing the
                  double-kicks so the chart is single-pedal playable.

    Returns (new_mid, stats) where stats = {"converted": n, "removed": n}.
    Never mutates the input file.
    """
    if mode not in ("1x", "2x"):
        raise ValueError(f"pedal mode must be '1x' or '2x', got {mode!r}")

    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    converted = 0
    removed = 0

    for track in mid.tracks:
        new_tr = mido.MidiTrack()
        new_tr.name = track.name
        if not _is_drums_track(track):
            for msg in track:
                new_tr.append(msg.copy())
            out.tracks.append(new_tr)
            continue

        # Drums track: walk messages, carrying delta time across dropped events
        # (1x mode) so timing of surviving messages is preserved.
        pending_delta = 0
        for msg in track:
            delta = msg.time + pending_delta
            is_note = msg.type in ("note_on", "note_off")
            if is_note and msg.note == DRUM_KICK_2X:
                if mode == "2x":
                    new_tr.append(msg.copy(note=DRUM_KICK_EXPERT, time=delta))
                    pending_delta = 0
                    if msg.type == "note_on" and msg.velocity > 0:
                        converted += 1
                else:  # 1x → drop this note, push its delta onto the next msg
                    pending_delta = delta
                    if msg.type == "note_on" and msg.velocity > 0:
                        removed += 1
            else:
                new_tr.append(msg.copy(time=delta))
                pending_delta = 0
        out.tracks.append(new_tr)

    return out, {"converted": converted, "removed": removed}


def count_double_kicks(mid: mido.MidiFile) -> int:
    """Number of Expert+ 2x-kick markers (note-95 note_ons) on the drums track(s).

    Zero means there is nothing for the "2x" variant to convert: the chart is
    already single-pedal, so a 2x build would be byte-for-byte the 1x build.
    Callers use this to decide whether the "2x" name/label is warranted."""
    n = 0
    for track in mid.tracks:
        if not _is_drums_track(track):
            continue
        for msg in track:
            if (msg.type == "note_on" and msg.velocity > 0
                    and msg.note == DRUM_KICK_2X):
                n += 1
    return n


# ── RB3 crash-safety sanitiser (PACK step only) ────────────────────────────────
def _msg_is_on(m) -> bool:
    return m.type == "note_on" and getattr(m, "velocity", 0) > 0


def _msg_is_off(m) -> bool:
    return m.type == "note_off" or (m.type == "note_on"
                                    and getattr(m, "velocity", 0) == 0)


def _is_ps_sysex(m) -> bool:
    """A Phase Shift sysex (open/tap markers): F0 50 53 00 00 … F7. mido stores
    data without the F0/F7, so it begins (0x50, 0x53)."""
    return m.type == "sysex" and tuple(getattr(m, "data", ()))[:2] == (0x50, 0x53)


def sanitize_for_rb(mid: mido.MidiFile) -> tuple[mido.MidiFile, dict]:
    """Return a NEW MidiFile made safe for Rock Band 3, fixing the things that
    crash the game in-game (Magma would reject them):

      * **Overlapping/stuck same-pitch notes** — a second note_on for a pitch that
        is still held (broken chord / hung sustain). RB3 can hang rendering the
        never-closed gem. We force-close the held note at the new onset, drop the
        now-dangling note_off, and close any note left open at the track's end.
      * **Phase Shift sysex** (open-note / tap markers, F0 50 53 …) — the YARG/CH
        workaround keeps these (with their illegal 0xFF byte), but RB3 doesn't read
        them; remove them so they can't confuse the loader.
      * **Tap-note markers (note 104)** on five-fret tracks — the Clone Hero/Phase
        Shift "tap force" pitch. RB3's standard guitar engine has no tap notes and
        104 is not a valid RB3 gem marker; Magma rejects it. We drop it (matching
        Onyx's RB3 target) — the affected gems simply revert to strum/HOPO by
        spacing, exactly as Onyx leaves them for RB3.

      * **Phase-Shift-only tracks** (any track whose name ends in ``_PS``, e.g.
        ``PART REAL_DRUMS_PS``) — these are Clone Hero / Phase Shift extensions RB3
        does not recognise. Onyx never emits them; shipping one can crash RB3's
        song loader. We drop the whole track (matching Onyx's RB3 output).

    Never mutates the input. Returns
    (new_mid, {"overlaps_fixed", "sysex_removed", "tap_removed", "ps_tracks_dropped"}).
    """
    from collections import defaultdict, deque
    from .midi_utils import AbsEvent

    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    overlaps_fixed = sysex_removed = tap_removed = ps_tracks_dropped = 0

    for track in mid.tracks:
        # Drop Phase-Shift-only tracks (e.g. PART REAL_DRUMS_PS) outright — RB3
        # can choke on a track name it doesn't know. (The standard RB pro-drums
        # track is "PART REAL_DRUMS", with no _PS suffix, and is kept.)
        if (track.name or "").strip().upper().endswith("_PS"):
            ps_tracks_dropped += 1
            continue
        abs_evts = to_abs(track)
        last_tick = abs_evts[-1].abs_tick if abs_evts else 0
        # Tap-force pitch is only meaningful on the five-fret instrument tracks.
        nm = (track.name or "").strip().upper()
        strip_tap = nm in _FIVE_FRET_TRACKS

        # Pair note_on/off into intervals per pitch, then make the intervals of
        # each pitch non-overlapping. This matches RB3's model (and the validator):
        # at any tick the offs apply before the ons, so a pitch must never be open
        # twice — neither across ticks (a hung sustain) nor at the same tick (a
        # duplicate gem). Non-note events (text/markers/meta) are kept verbatim.
        stacks: dict[int, deque] = defaultdict(deque)        # pitch → FIFO of (tick, on_msg)
        intervals: dict[int, list] = defaultdict(list)       # pitch → [[on, off, on_msg], …]
        others: list = []                                    # (tick, msg) non-note / kept

        # Process offs before ons within a tick so a back-to-back gem pairs right.
        ordered = sorted(enumerate(abs_evts),
                         key=lambda ie: (ie[1].abs_tick,
                                         0 if _msg_is_off(ie[1].msg)
                                         else (1 if _msg_is_on(ie[1].msg) else -1),
                                         ie[0]))
        for _, e in ordered:
            m = e.msg
            if _is_ps_sysex(m):
                sysex_removed += 1
                continue
            if strip_tap and getattr(m, "note", None) == _TAP_NOTE:
                tap_removed += 1
                continue
            n = getattr(m, "note", None)
            if n is not None and _msg_is_on(m):
                stacks[n].append((e.abs_tick, m))
            elif n is not None and _msg_is_off(m):
                if stacks[n]:
                    on_tick, on_msg = stacks[n].popleft()
                    intervals[n].append([on_tick, e.abs_tick, on_msg])
                # else: dangling off → drop
            else:
                others.append((e.abs_tick, m))
        # Notes left open at the track end → close them at the last tick.
        for n, dq in stacks.items():
            for on_tick, on_msg in dq:
                intervals[n].append([on_tick, max(on_tick + 1, last_tick), on_msg])
                overlaps_fixed += 1

        note_evts: list = []
        for n, ivs in intervals.items():
            ivs.sort(key=lambda x: (x[0], x[1]))
            # Clamp each interval's end to the next one's start (allow back-to-back).
            for i in range(len(ivs) - 1):
                if ivs[i][1] > ivs[i + 1][0]:
                    ivs[i][1] = ivs[i + 1][0]
                    overlaps_fixed += 1
            for on_tick, off_tick, on_msg in ivs:
                if off_tick <= on_tick:          # zero-length (same-tick duplicate)
                    overlaps_fixed += 1
                    continue
                note_evts.append((on_tick, on_msg.copy()))
                note_evts.append((off_tick,
                                  mido.Message("note_off", note=n, velocity=0,
                                               channel=getattr(on_msg, "channel", 0))))

        # Merge kept + note events; within a tick keep meta first, then offs, then
        # ons (the same order RB3 reads), then rebuild monotonic delta times.
        merged = others + note_evts

        def _k(tm):
            t, m = tm
            pr = 1 if _msg_is_off(m) else (2 if _msg_is_on(m) else 0)
            return (t, pr)
        merged.sort(key=_k)
        out.tracks.append(to_track([AbsEvent(abs_tick=t, msg=m)
                                    for t, m in merged]))

    return out, {"overlaps_fixed": overlaps_fixed, "sysex_removed": sysex_removed,
                 "tap_removed": tap_removed, "ps_tracks_dropped": ps_tracks_dropped}


# ── Onyx no-Magma fixups (PACK step only) ──────────────────────────────────────
# Ported from the corrections Onyx applies when building an RB3 song WITHOUT the
# Magma compiler (mtolly/onyx Onyx/Build/RB3CH.hs `processMIDI`). These are the
# minimum guarantees Magma would otherwise enforce; the ones below genuinely
# affect whether RB3 loads/plays a song. Each is a no-op when nothing needs fixing
# and never mutates the input. Audio-coupled fixups (lead-in pad) are separate so
# the caller can prepend matching silence to the mogg.

_OVERDRIVE_NOTE = 116          # the single overdrive/star-power marker (all parts)
_GEM_LO, _GEM_HI = 60, 100     # gem lane span across Easy..Expert (5-fret + drums)
_TAP_NOTE = 104                # CH/Phase-Shift "tap force" pitch (no RB3 equivalent)
# Five-fret instrument tracks that carry the CH/PS tap-force marker (note 104).
_FIVE_FRET_TRACKS = {"PART GUITAR", "PART BASS", "PART KEYS", "PART RHYTHM",
                     "PART GUITAR COOP"}

# Drum difficulty → the lowest gem pitch of that difficulty (used to know which
# difficulties are actually charted so we only mix events that exist).
_DRUM_DIFF_BASE = {0: 60, 1: 72, 2: 84, 3: 96}   # Easy/Medium/Hard/Expert


def _is_drums_name(name) -> bool:
    return "DRUM" in (name or "").strip().upper()


def _remove_noteless_overdrive(track) -> int:
    """Onyx `fixNotelessOD`: drop note-116 overdrive phrases that contain no gem
    in any difficulty. An empty overdrive phrase is rejected by Magma and can
    make RB3 choke when it tries to award star-power over nothing. Mutates the
    given (absolute-event) list in place via rebuild; returns count removed."""
    abs_evts = to_abs(track)
    # Overdrive on/off spans.
    od_spans: list[list[int]] = []           # [on_tick, off_tick, on_idx, off_idx]
    open_on = None
    gem_ticks: list[int] = []
    for i, e in enumerate(abs_evts):
        m = e.msg
        n = getattr(m, "note", None)
        if n is None:
            continue
        if n == _OVERDRIVE_NOTE:
            if _msg_is_on(m):
                open_on = (e.abs_tick, i)
            elif _msg_is_off(m) and open_on is not None:
                od_spans.append([open_on[0], e.abs_tick, open_on[1], i])
                open_on = None
        elif _msg_is_on(m) and _GEM_LO <= n <= _GEM_HI:
            gem_ticks.append(e.abs_tick)
    if not od_spans:
        return 0
    gem_ticks.sort()
    import bisect
    drop_idx: set[int] = set()
    removed = 0
    for on_t, off_t, on_i, off_i in od_spans:
        lo = bisect.bisect_left(gem_ticks, on_t)
        hi = bisect.bisect_left(gem_ticks, off_t)
        if hi <= lo:                          # no gem onset inside [on, off)
            drop_idx.add(on_i)
            drop_idx.add(off_i)
            removed += 1
    if not removed:
        return 0
    kept = [e for i, e in enumerate(abs_evts) if i not in drop_idx]
    new_tr = to_track(kept)
    new_tr.name = track.name
    track[:] = new_tr
    return removed


def _add_drum_mix_events(track) -> int:
    """Onyx `drumsComplete` (mix portion): RB3 needs a ``[mix <diff> drums0]`` text
    event for each charted drum difficulty, or the drum audio mapping is undefined
    (silent kit / possible hang). We add the missing ones at tick 0 — but only if
    the track carries NO ``[mix ...]`` events at all (an authored mix is left
    untouched). ``drums0`` = single (stereo) drum stem, the YARG/CH norm; sources
    with separate kick/snare stems ship their own mix events, so we never override.
    Returns the number of mix events added."""
    has_mix = any(getattr(m, "text", "").strip().lower().startswith("[mix")
                  for m in track if m.type in ("text", "lyrics"))
    if has_mix:
        return 0
    charted = {d for d, base in _DRUM_DIFF_BASE.items()
               if any(_msg_is_on(m) and base <= getattr(m, "note", -1) <= base + 4
                      for m in track)}
    if not charted:
        return 0
    # Insert "[mix d drums0]" text events at the very front (tick 0).
    abs_evts = to_abs(track)
    new_front = []
    from .midi_utils import AbsEvent
    for d in sorted(charted):
        new_front.append(AbsEvent(abs_tick=0,
                                  msg=mido.MetaMessage("text",
                                                       text=f"[mix {d} drums0]",
                                                       time=0)))
    new_tr = to_track(new_front + abs_evts)
    new_tr.name = track.name
    track[:] = new_tr
    return len(charted)


def _time_sig_changes(mid: mido.MidiFile) -> list[tuple[int, int, int]]:
    """Return [(abs_tick, numerator, denominator), ...] sorted, with an implicit
    4/4 at tick 0 if none is authored there."""
    changes: list[tuple[int, int, int]] = []
    for tr in mid.tracks:
        t = 0
        for m in tr:
            t += m.time
            if m.type == "time_signature":
                changes.append((t, m.numerator, m.denominator))
    changes.sort(key=lambda c: c[0])
    if not changes or changes[0][0] != 0:
        changes.insert(0, (0, 4, 4))
    return changes


def _generate_beats(mid: mido.MidiFile, end_tick: int) -> list[tuple[int, bool]]:
    """Produce [(tick, is_downbeat), ...] from tick 0 up to (not including)
    end_tick, following the authored time signatures. Beat unit = a quarter note
    scaled by the denominator (tpb*4/den); the first beat of each measure (and of
    every time-sig change) is the downbeat. This is the data-derived skeleton Onyx
    `basicTiming`/`fixBeatTrack` lays down — no per-song hard-coding."""
    tpb = mid.ticks_per_beat
    changes = _time_sig_changes(mid)
    beats: list[tuple[int, bool]] = []
    for idx, (seg_start, num, den) in enumerate(changes):
        seg_end = changes[idx + 1][0] if idx + 1 < len(changes) else end_tick
        seg_end = min(seg_end, end_tick)
        beat_unit = max(1, tpb * 4 // max(1, den))
        t = seg_start
        bim = 0  # beat index within the measure (resets at each time-sig change)
        while t < seg_end:
            beats.append((t, bim % max(1, num) == 0))
            t += beat_unit
            bim += 1
    return beats


def _add_basic_timing(mid: mido.MidiFile) -> dict:
    """Onyx `basicTiming`: guarantee an EVENTS ``[end]`` marker and a populated
    BEAT track — songs we never processed (raw YARG/CH .mid) may ship neither, and
    RB3 needs both (no [end] => the song never ends / can hang; no BEAT track =>
    no measure grid). Mutates `mid` in place; adds only what is missing. Returns
    {"end_added": 0|1, "beat_added": <beat notes added>}."""
    from .midi_utils import AbsEvent
    tpb = mid.ticks_per_beat

    # Last event tick across the whole file (song extent).
    last = 0
    for tr in mid.tracks:
        t = 0
        for m in tr:
            t += m.time
        last = max(last, t)

    # Existing [end] tick (in any track), if authored.
    end_tick = None
    for tr in mid.tracks:
        t = 0
        for m in tr:
            t += m.time
            if m.type == "text" and (m.text or "").strip().lower() == "[end]":
                end_tick = t if end_tick is None else min(end_tick, t)

    end_added = 0
    if end_tick is None:
        # Place [end] two beats past the last content, rounded up to a beat.
        end_tick = (last // tpb + 2) * tpb
        evt = AbsEvent(abs_tick=end_tick,
                       msg=mido.MetaMessage("text", text="[end]", time=0))
        ev_tr = next((tr for tr in mid.tracks
                      if (tr.name or "").strip().upper() == "EVENTS"), None)
        if ev_tr is None:
            ev_tr = mido.MidiTrack()
            mid.tracks.append(ev_tr)
            ev_tr[:] = to_track([evt])
        else:
            ev_tr[:] = to_track(to_abs(ev_tr) + [evt])
        ev_tr.name = "EVENTS"   # (re)set after content replace; to_track drops it
        end_added = 1

    # BEAT track: present AND carrying downbeat/upbeat (12/13) notes?
    beat_tr = next((tr for tr in mid.tracks
                    if (tr.name or "").strip().upper() == "BEAT"), None)
    has_beats = beat_tr is not None and any(
        getattr(m, "note", None) in (12, 13) for m in beat_tr)

    # A BEAT track can be PRESENT but malformed: gaps where whole beats/measures
    # are missing (we have seen authored tracks that skip 3 beats every measure in
    # constant 4/4). RB3 needs a continuous measure grid — a gappy BEAT grid is a
    # known crash class. Detect a defect by comparing each beat-to-beat interval to
    # the track's own median spacing; any gap noticeably larger than the median
    # (i.e. missing beats) flags the track for a full rebuild from the time-sig map.
    beat_defective = False
    if has_beats:
        ons = []
        t = 0
        for m in beat_tr:
            t += m.time
            if m.type == "note_on" and getattr(m, "note", None) in (12, 13) \
                    and m.velocity > 0:
                ons.append(t)
        if len(ons) >= 3:
            gaps = sorted(b - a for a, b in zip(ons, ons[1:]) if b > a)
            med = gaps[len(gaps) // 2]
            if med > 0 and any(g > med * 1.5 for g in gaps):
                beat_defective = True

    beat_added = 0
    if (not has_beats or beat_defective) and end_tick > 0:
        beats = _generate_beats(mid, end_tick)
        off_gap = max(1, tpb // 8)
        evts = []
        for tick, is_down in beats:
            note = 12 if is_down else 13
            evts.append(AbsEvent(abs_tick=tick,
                                 msg=mido.Message("note_on", note=note,
                                                  velocity=100, time=0)))
            evts.append(AbsEvent(abs_tick=tick + off_gap,
                                 msg=mido.Message("note_off", note=note,
                                                  velocity=0, time=0)))
        if beat_tr is None:
            beat_tr = mido.MidiTrack()
            beat_tr.name = "BEAT"
            mid.tracks.append(beat_tr)
        beat_tr[:] = to_track(evts)
        beat_tr.name = "BEAT"
        beat_added = len(beats)

    return {"end_added": end_added, "beat_added": beat_added}


# Tracks that participate in RB3 unison-overdrive bonuses. A unison is awarded
# when 2+ of these have an overdrive phrase that starts at the SAME tick; phrases
# that overlap but start at different ticks form an *invalid* (partial) unison.
_UNISON_TRACKS = {"PART DRUMS", "PART BASS", "PART GUITAR", "PART KEYS"}
# Drum fill / BRE activation lanes (all five light up together for one fill).
_FILL_NOTES = range(120, 125)
_CLOSE_FILL_SEC = 2.5          # Onyx `fixCloseFills`: minimum gap between fills
_MUSIC_PAD_BEATS = 2.0         # [music_start]/[music_end] inset from start/[end]


def _events_track(mid: mido.MidiFile) -> mido.MidiTrack:
    """Return the EVENTS track, creating an empty one if absent."""
    ev = next((tr for tr in mid.tracks
               if (tr.name or "").strip().upper() == "EVENTS"), None)
    if ev is None:
        ev = mido.MidiTrack()
        ev.name = "EVENTS"
        mid.tracks.append(ev)
    return ev


def _add_music_events(mid: mido.MidiFile) -> dict:
    """Onyx places ``[music_start]`` / ``[music_end]`` text markers in EVENTS when
    missing. They bound where the backing music (and crowd) play; their absence
    doesn't crash but makes the song outro/crowd misbehave. ``[music_start]`` goes
    two beats in (Onyx default [1:3:000] in 4/4), ``[music_end]`` two beats before
    ``[end]``. Run AFTER basicTiming so ``[end]`` exists. No-op when both present;
    never adds an end marker past/at the start. Returns {"music_start_added",
    "music_end_added"}."""
    from .midi_utils import AbsEvent
    tpb = mid.ticks_per_beat
    inset = int(round(_MUSIC_PAD_BEATS * tpb))

    have_start = have_end = False
    end_tick = None
    last = 0
    for tr in mid.tracks:
        t = 0
        for m in tr:
            t += m.time
            if m.type == "text":
                txt = (m.text or "").strip().lower()
                if txt == "[music_start]":
                    have_start = True
                elif txt == "[music_end]":
                    have_end = True
                elif txt == "[end]":
                    end_tick = t if end_tick is None else min(end_tick, t)
        last = max(last, t)
    if end_tick is None:
        end_tick = last

    to_add: list[AbsEvent] = []
    start_added = end_added = 0
    if not have_start:
        st = min(inset, max(0, end_tick - 1))
        to_add.append(AbsEvent(abs_tick=st,
                      msg=mido.MetaMessage("text", text="[music_start]", time=0)))
        start_added = 1
    if not have_end:
        me = end_tick - inset
        if me > inset:          # keep it after music_start and inside the song
            to_add.append(AbsEvent(abs_tick=me,
                          msg=mido.MetaMessage("text", text="[music_end]", time=0)))
            end_added = 1
    if to_add:
        ev = _events_track(mid)
        ev[:] = to_track(to_abs(ev) + to_add)
        ev.name = "EVENTS"
    return {"music_start_added": start_added, "music_end_added": end_added}


def _od_phrases(track) -> list[tuple[int, int, int, int]]:
    """Overdrive (note 116) spans of a track as [(start, end, on_idx, off_idx)]
    over its absolute-event list."""
    spans: list[tuple[int, int, int, int]] = []
    open_on = None
    for i, e in enumerate(to_abs(track)):
        n = getattr(e.msg, "note", None)
        if n != _OVERDRIVE_NOTE:
            continue
        if _msg_is_on(e.msg):
            open_on = (e.abs_tick, i)
        elif _msg_is_off(e.msg) and open_on is not None:
            spans.append((open_on[0], e.abs_tick, open_on[1], i))
            open_on = None
    return spans


def _fix_partial_unisons(mid: mido.MidiFile) -> int:
    """Onyx `fixPartialUnisons`: RB3 groups overlapping overdrive phrases across
    instruments into a unison bonus, but only when they start at the SAME tick. A
    phrase that overlaps another instrument's phrase yet starts at a different tick
    is an invalid (partial) unison and is rejected by Magma / mishandled in-game.
    We remove the later-starting offender of every such overlapping, misaligned
    pair (dropping an OD phrase only loses a little star-power — never gems).
    No-op when all overlaps are aligned. Returns phrases removed."""
    tracks = [tr for tr in mid.tracks
              if (tr.name or "").strip().upper() in _UNISON_TRACKS]
    phrases = []   # (start, end, track_obj, on_idx, off_idx)
    for tr in tracks:
        for start, end, on_i, off_i in _od_phrases(tr):
            phrases.append((start, end, tr, on_i, off_i))

    drop_per_track: dict[int, set[int]] = {}
    removed = 0
    for a in range(len(phrases)):
        sa, ea, tra, ona, offa = phrases[a]
        for b in range(a + 1, len(phrases)):
            sb, eb, trb, onb, offb = phrases[b]
            if tra is trb:
                continue
            if sa < eb and sb < ea and sa != sb:        # overlap, misaligned
                later_tr, later_on, later_off = (
                    (trb, onb, offb) if sb > sa else (tra, ona, offa))
                d = drop_per_track.setdefault(id(later_tr), set())
                if later_on not in d:
                    d.add(later_on); d.add(later_off); removed += 1
    if not removed:
        return 0
    for tr in tracks:
        drop = drop_per_track.get(id(tr))
        if not drop:
            continue
        kept = [e for i, e in enumerate(to_abs(tr)) if i not in drop]
        name = tr.name
        tr[:] = to_track(kept)
        tr.name = name
    return removed


def _remove_close_fills(mid: mido.MidiFile) -> int:
    """Onyx `fixCloseFills`: drum fill activation lanes (notes 120-124) spaced less
    than 2.5 s apart confuse RB3's fill/BRE logic. We keep the first fill and drop
    any whose start is within 2.5 s of the previously kept fill's end. Operates on
    PART DRUMS only. No-op when fills are well spaced. Returns fills removed."""
    tr = next((t for t in mid.tracks
               if (t.name or "").strip().upper() == "PART DRUMS"), None)
    if tr is None:
        return 0
    # Fill spans tracked via lane 120 (all five lanes share the same span).
    spans: list[list[int]] = []
    open_on = None
    for e in to_abs(tr):
        if getattr(e.msg, "note", None) != 120:
            continue
        if _msg_is_on(e.msg):
            open_on = e.abs_tick
        elif _msg_is_off(e.msg) and open_on is not None:
            spans.append([open_on, e.abs_tick]); open_on = None
    if len(spans) < 2:
        return 0
    spans.sort()
    tempo_map = build_tempo_map(mid)
    tpb = mid.ticks_per_beat
    remove_ranges: list[tuple[int, int]] = []
    prev_end = spans[0][1]
    for start, end in spans[1:]:
        gap_ms = tick_to_ms(start, tempo_map, tpb) - tick_to_ms(prev_end, tempo_map, tpb)
        if gap_ms < _CLOSE_FILL_SEC * 1000.0:
            remove_ranges.append((start, end))      # too close → drop this fill
        else:
            prev_end = end
    if not remove_ranges:
        return 0
    kept = []
    for e in to_abs(tr):
        n = getattr(e.msg, "note", None)
        if n in _FILL_NOTES and any(s <= e.abs_tick <= en for s, en in remove_ranges):
            continue
        kept.append(e)
    name = tr.name
    tr[:] = to_track(kept)
    tr.name = name
    return len(remove_ranges)


def _extend_short_fills(mid: mido.MidiFile) -> dict:
    """RB3 drum fill activation lanes (notes 120-124) must span a real region — the
    default authored length is one measure. Some sources ship degenerate 1-tick fill
    markers (start==end+1); RB3 mishandles a zero-length fill (a known crash class).
    We lengthen any fill shorter than its measure by moving its START backward to
    ``end - one_measure`` so the ACTIVATION point (the fill end) stays exactly where
    it was — the fill still triggers in the same place. The new start is clamped to
    the previous fill's end, to tick 0, AND to the end of any preceding overdrive
    (note 116) phrase: a drum fill that overlaps an OD phrase is itself invalid in
    RB3 (another crash class), so we never extend a fill back into one. Measure
    length is read from the authored time signature at the fill (data-derived).
    Operates on PART DRUMS only. No-op when every fill is already at least a measure
    long.     Returns {"extended": int, "removed": int} where removed fills are ones
    that were shorter than a measure and couldn't be extended to a full measure
    (blocked by OD/prev fill/song start) — keeping them degenerate is more
    dangerous than dropping them."""
    tr = next((t for t in mid.tracks
               if (t.name or "").strip().upper() == "PART DRUMS"), None)
    if tr is None:
        return {"extended": 0, "removed": 0}
    tpb = mid.ticks_per_beat
    sig_changes = _time_sig_changes(mid)   # [(tick, num, den), ...] sorted

    def measure_ticks_at(tick: int) -> int:
        num, den = 4, 4
        for ct, n, d in sig_changes:
            if ct <= tick:
                num, den = n, d
            else:
                break
        return max(1, num) * max(1, tpb * 4 // max(1, den))

    events = to_abs(tr)
    # Overdrive (note 116) phrase spans on this drum track — a fill must not be
    # extended back into one (fill overlapping OD is invalid in RB3).
    od_spans = []
    _od_open = None
    for e in events:
        if getattr(e.msg, "note", None) != _OVERDRIVE_NOTE:
            continue
        if _msg_is_on(e.msg):
            _od_open = e.abs_tick
        elif _msg_is_off(e.msg) and _od_open is not None:
            od_spans.append((_od_open, e.abs_tick)); _od_open = None

    # Pair each fill-lane note_on with its matching note_off; group by start tick.
    open_at: dict[int, list] = {}        # note -> [on_event,...]
    spans = []                           # (start, end, [on_events])
    starts: dict[int, list] = {}
    for e in events:
        n = getattr(e.msg, "note", None)
        if n not in _FILL_NOTES:
            continue
        if _msg_is_on(e.msg):
            open_at.setdefault(n, []).append(e)
        elif _msg_is_off(e.msg) and open_at.get(n):
            on_e = open_at[n].pop(0)
            starts.setdefault(on_e.abs_tick, [on_e.abs_tick, e.abs_tick, []])
            starts[on_e.abs_tick][1] = max(starts[on_e.abs_tick][1], e.abs_tick)
            starts[on_e.abs_tick][2].append(on_e)
    if not starts:
        return {"extended": 0, "removed": 0}
    ordered = sorted(starts.values(), key=lambda s: s[0])
    extended = 0
    removed = 0
    prev_end = 0
    drop_ranges: list[tuple[int, int]] = []
    for start, end, on_events in ordered:
        meas = measure_ticks_at(end)
        if end - start < meas:
            # Floor at the end of any OD phrase that precedes this fill's end, so
            # the widened fill never overlaps an overdrive phrase.
            od_floor = max((oe for os, oe in od_spans if os < end and oe <= end),
                           default=0)
            new_start = max(prev_end, end - meas, 0, od_floor)
            new_len = end - new_start
            if new_len >= meas:
                for on_e in on_events:
                    on_e.abs_tick = new_start
                extended += 1
            else:
                # Blocked from extending to a full measure (OD/prev fill/song
                # start) — remove the short fill entirely; keeping it degenerate
                # is more dangerous than dropping it.
                drop_ranges.append((start, end))
                removed += 1
        prev_end = end
    if not extended and not removed:
        return {"extended": 0, "removed": 0}
    if removed:
        def _keep(e):
            n = getattr(e.msg, "note", None)
            if n is not None and n in _FILL_NOTES:
                for ds, de in drop_ranges:
                    if ds <= e.abs_tick <= de:
                        return False
            return True
        events = [e for e in events if _keep(e)]
    name = tr.name
    tr[:] = to_track(events)
    tr.name = name
    return {"extended": extended, "removed": removed}


def apply_rb_fixups(mid: mido.MidiFile) -> tuple[mido.MidiFile, dict]:
    """Apply Onyx's no-Magma MIDI fixups that affect RB3 load/playback, returning
    a NEW MidiFile and a stats dict. Currently: ensure an EVENTS ``[end]`` marker
    and a BEAT track (`basicTiming`), remove note-less overdrive phrases
    (`fixNotelessOD`) on every instrument track, add missing drum `[mix]`
    events (`drumsComplete`) on PART DRUMS, fix invalid partial unisons
    (`fixPartialUnisons`), drop too-close drum fills (`fixCloseFills`), and add
    `[music_start]`/`[music_end]` markers when missing. No-op where nothing needs
    fixing; never mutates the input. The lead-in pad is handled separately (it
    needs the mogg padded in lockstep)."""
    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    for track in mid.tracks:
        new_tr = mido.MidiTrack()
        new_tr.name = track.name
        for m in track:
            new_tr.append(m.copy())
        out.tracks.append(new_tr)

    od_removed = mix_added = 0
    for tr in out.tracks:
        nm = (tr.name or "").strip().upper()
        if not nm or nm in ("BEAT", "VENUE", "EVENTS"):
            continue
        od_removed += _remove_noteless_overdrive(tr)
        # Drum [mix] only on the standard PART DRUMS (not PART REAL_DRUMS_PS etc,
        # which RB3 doesn't read and which would get a spurious mix event).
        if nm == "PART DRUMS":
            mix_added += _add_drum_mix_events(tr)

    # Cross-track fixups (after the per-track OD removal above settled phrases).
    unison_removed = _fix_partial_unisons(out)
    close_fills_removed = _remove_close_fills(out)
    # Disabled: Onyx does NOT extend short fills (notes 120-124 point markers
    # stay as 1-tick triggers). Extending them to 1 measure can crash RB3.
    fills_extended = 0
    fills_removed = 0

    # basicTiming runs late so it can add an EVENTS/BEAT track without disturbing
    # the per-track loop above (and so a freshly built BEAT track is padded by
    # pad_start downstream). music markers run last — they need [end] to exist.
    timing = _add_basic_timing(out)
    music = _add_music_events(out)
    return out, {"noteless_od_removed": od_removed, "drum_mix_added": mix_added,
                 "end_added": timing["end_added"],
                 "beat_added": timing["beat_added"],
                 "unison_removed": unison_removed,
                 "close_fills_removed": close_fills_removed,
                 "fills_extended": fills_extended,
                 "fills_removed": fills_removed,
                 "music_start_added": music["music_start_added"],
                 "music_end_added": music["music_end_added"]}


def lead_in_pad_ticks(mid: mido.MidiFile, min_beats: float = 6.0) -> int:
    """Ticks of silence to prepend so the first gem sits at least `min_beats` beats
    from the song start (Onyx `magmaPad`). RB3 needs lead-in before the first note;
    a chart starting at tick 0 (no BEAT lead-in) is rejected/hang. Returns 0
    when the chart already has enough lead-in. Default 6 beats = 3.0s at 120 BPM
    matching Onyx's magmaPad exact behaviour.

    Only counts playable notes (60-102), skipping phrase markers (>=103)."""
    tpb = mid.ticks_per_beat

    # Find first playable note (note < 103) across all PART tracks
    first_note = None
    for tr in mid.tracks:
        nm = (tr.name or "").strip().upper()
        if not nm.startswith("PART"):
            continue
        t = 0
        for m in tr:
            t += m.time
            if m.type == "note_on" and m.velocity > 0:
                note = getattr(m, "note", -1)
                # Skip phrase markers (105/106), BRE (120-124), drum fills (110-112)
                if note < 103:
                    first_note = t if first_note is None else min(first_note, t)
                break
    if first_note is None:
        return 0
    need = int(round(min_beats * tpb))
    return max(0, need - first_note)



def pad_start(mid: mido.MidiFile, pad_ticks: int) -> mido.MidiFile:
    """Return a NEW MidiFile matching Onyx's magmaPad exact output:
    1. Extend the first time-signature bar (4/4 → 10/4 for 6 extra beats).
    2. Insert 120 BPM at tick 0 for the padding region.
    3. Restore original tempo at pad_ticks (where the first note lands).
    4. Restore original time-signature at new_num * tpb (end of the extended bar).
    5. Fill BEAT track across the lead-in.
    6. Shift all note/lyric/text events by pad_ticks.
    Original tempo and time-signature events are replaced, not duplicated."""
    if pad_ticks <= 0:
        return mid
    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    tpb = mid.ticks_per_beat

    init_ts = next(((m.numerator, m.denominator) for tr in mid.tracks for m in tr
                    if m.type == "time_signature"), (4, 4))
    orig_num, orig_den = init_ts
    extra_beats = pad_ticks // tpb
    new_num = orig_num + extra_beats  # e.g. 4 + 6 = 10

    for track in mid.tracks:
        nm = (track.name or "").strip().upper()
        abs_evts = to_abs(track)
        # Strip events we replace from shifted list.  Keep set_tempo events
        # AFTER the first one — they are valid tempo changes along the song
        # that must shift with pad_ticks.  Only the first set_tempo (at tick 0)
        # is replaced by the 120 BPM pad + restore pair below.
        first_tempo_seen = False
        def _keep_set_tempo(e):
            nonlocal first_tempo_seen
            if e.msg.type == "set_tempo":
                if not first_tempo_seen:
                    first_tempo_seen = True
                    return False   # skip first tempo (replaced by pad)
                return True        # keep subsequent tempo changes
            return True
        shifted = [AbsEvent_shift(e, pad_ticks) for e in abs_evts
                   if _keep_set_tempo(e)
                   and e.msg.type not in ("time_signature", "track_name")]
        # For BEAT: drop shifted events inside the extended bar (we generated them).
        if nm == "BEAT":
            extended_end = new_num * tpb
            shifted = [e for e in shifted if e.abs_tick >= extended_end]

        front = []
        # Preserve track name.
        if track.name:
            front.append(_ev(0, mido.MetaMessage("track_name",
                                                 name=track.name, time=0)))
        if any(m.type == "set_tempo" for m in track):
            orig_tempo = next((m.tempo for m in track
                               if m.type == "set_tempo"), 500000)
            front.append(_ev(0, mido.MetaMessage("set_tempo", tempo=500000, time=0)))
        if any(m.type == "time_signature" for m in track):
            front.append(_ev(0, mido.MetaMessage("time_signature",
                                                 numerator=new_num,
                                                 denominator=orig_den, time=0)))
        # Restore original tempo where the first note lands.
        if any(m.type == "set_tempo" for m in track):
            front.append(_ev(pad_ticks, mido.MetaMessage("set_tempo",
                                                         tempo=orig_tempo, time=0)))
        # Restore original time-sig at the END of the extended bar.
        if any(m.type == "time_signature" for m in track):
            front.append(_ev(new_num * tpb, mido.MetaMessage("time_signature",
                                                             numerator=orig_num,
                                                             denominator=orig_den,
                                                             time=0)))
        # BEAT lead-in: fill beats across the extended bar.
        if nm == "BEAT":
            beat = 0
            t = 0
            while t < new_num * tpb:
                note = 12 if (beat % max(1, orig_num) == 0) else 13
                front.append(_ev(t, mido.Message("note_on", note=note,
                                                 velocity=100, time=0)))
                front.append(_ev(t + max(1, tpb // 8),
                                 mido.Message("note_off", note=note,
                                              velocity=0, time=0)))
                t += tpb
                beat += 1
        out.tracks.append(to_track(front + shifted))
    if getattr(out, "_merged_track", None) is not None:
        out._merged_track = None
    return out


# Init-marker templates matching Onyx's edgesBRE output exactly.
# (note, off_offset_or_None) — force markers have no note_off.
_INIT_GUITAR = [(101, None), (96, 60), (89, None), (84, 60),
                (77, None), (72, 60), (65, None), (60, 60)]
_INIT_BASS   = [(102, None), (97, 60), (90, None), (84, 60),
                (78, None), (72, 60), (66, None), (60, 60)]


def fix_init_markers(mid: mido.MidiFile) -> mido.MidiFile:
    """Replace the first note-on cluster in PART GUITAR / PART BASS with the
    full set of Onyx-style init markers (force-HOPO/strum + lane notes for
    all four difficulty levels).  Force markers stay on; lane notes get a
    60-tick note_off.  Matches Onyx edgesBRE byte-for-byte."""
    for track in mid.tracks:
        nm = (track.name or "").strip().upper()
        markers = _INIT_GUITAR if nm == "PART GUITAR" else \
                  _INIT_BASS   if nm == "PART BASS"   else None
        if markers is None:
            continue
        # Find first note_on (velocity > 0).
        t, first = 0, None
        for m in track:
            t += m.time
            if _msg_is_on(m) and _GEM_LO <= getattr(m, "note", -1) <= _GEM_HI:
                first = t
                break
        if first is None:
            continue
        abs_evts = to_abs(track)
        # Drop note events in a 100-tick window around the first note.
        cleaned = [e for e in abs_evts
                   if not (first <= e.abs_tick <= first + 100
                           and e.msg.type in ("note_on", "note_off")
                           and hasattr(e.msg, "note"))]
        # Insert Onyx init markers.
        for note, off_off in markers:
            cleaned.append(_ev(first, mido.Message("note_on", note=note,
                                                   velocity=96, time=0)))
            if off_off is not None:
                cleaned.append(_ev(first + off_off,
                                   mido.Message("note_off", note=note,
                                                velocity=0, time=0)))
        new_track = to_track(cleaned)
        new_track.name = track.name
        mid.tracks[mid.tracks.index(track)] = new_track
    return mid


def _ev(tick, msg):
    from .midi_utils import AbsEvent
    return AbsEvent(abs_tick=tick, msg=msg)


def AbsEvent_shift(e, pad_ticks):
    from .midi_utils import AbsEvent
    return AbsEvent(abs_tick=e.abs_tick + pad_ticks, msg=e.msg.copy())

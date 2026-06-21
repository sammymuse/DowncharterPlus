"""
constants.py — MIDI note map for Clone Hero / YARG
Source: TheNathannator/GuitarGame_ChartFormats + RBN/C3 Docs
"""

# ── 5-fret: playable notes per difficulty ─────────────────────────────────────
#
#  Expert  96-100  (G=96 R=97 Y=98 B=99 O=100)  Open=95  FH=101 FS=102
#  Hard    84-88   (G=84 R=85 Y=86 B=87 O=88)   Open=83  FH=89  FS=90
#  Medium  72-75   (G=72 R=73 Y=74 B=75)         Open=71  FH=77  FS=78
#  Easy    60-62   (G=60 R=61 Y=62)              Open=59  FH=65  FS=66
#
# RBN rule: Medium focuses on G R Y B (up to blue);  Easy focuses on G R Y (up to yellow)
# Notes outside the range are wrapped (mapped) onto the allowed frets

FRET_NAMES = {
    0: "Green",
    1: "Red",
    2: "Yellow",
    3: "Blue",
    4: "Orange",
}

# Expert notes for each fret (0=Green … 4=Orange)
EXPERT_BASE = 96   # Green
FRET_COUNT  = 5

# Difficulty offsets relative to Expert
DIFF_OFFSET = {
    "expert": 0,
    "hard":  -12,
    "medium":-24,
    "easy":  -36,
}

# Allowed fret notes per difficulty (index 0-4)
FRETS_ALLOWED = {
    "expert": list(range(5)),      # 0-4: G R Y B O
    "hard":   list(range(5)),      # 0-4: G R Y B O  (Hard has all 5)
    "medium": list(range(4)),      # 0-3: G R Y B
    "easy":   list(range(3)),      # 0-2: G R Y
}

# Sentinel fret for the OPEN note (strum with no strings). 1 below Green.
# fret_note(OPEN_FRET, diff) = EXPERT_BASE-1+offset → 95/83/71/59 (Expert/Hard/Med/Easy).
OPEN_FRET = -1

def fret_note(fret_idx: int, diff: str) -> int:
    """Return the MIDI note for the fret (0=Green…4=Orange, -1=Open) at the given difficulty."""
    return EXPERT_BASE + fret_idx + DIFF_OFFSET[diff]

def note_to_fret(note: int, diff: str) -> int | None:
    """Convert MIDI note → fret index (0-4). None if not a playable fret."""
    off = note - EXPERT_BASE - DIFF_OFFSET[diff]
    if 0 <= off < FRET_COUNT:
        return off
    return None

def is_fret_note(note: int, diff: str) -> bool:
    return note_to_fret(note, diff) is not None

def is_open_note(note: int, diff: str) -> bool:
    """Open note: 1 below Green at the given difficulty."""
    return note == (EXPERT_BASE - 1 + DIFF_OFFSET[diff])

def is_force_note(note: int, diff: str) -> bool:
    """ForceHOPO (Green+5) ou ForceStrum (Green+6)."""
    base = EXPERT_BASE + DIFF_OFFSET[diff]
    return note in (base + 5, base + 6)

# ── Global markers (no offset between difficulties) ───────────────────────────
GLOBAL_MARKERS = set(range(103, 128))
# Includes: 103=Solo, 116=StarPower, 120-124=BRE, 126=Tremolo, 127=Trill

# ── Sustain rules (RBN docs) ──────────────────────────────────────────────────
# "Short" = has no tail  → length ≤ 1/16 (any BPM)
# Expert: minimum sustain = dotted-8th if BPM>100, else 8th
# Hard: same as Expert
# Medium: sustain trimmed by +1/8 (dotted-8th → 16th); minimum gap = 1/4
# Easy: sustain trimmed by an extra +1/16 vs Medium; minimum gap = 1/4

# ── Chord rules (RBN docs) ───────────────────────────────────────────────────
# Expert: all chords allowed; forbidden G+O+anything (3-note with G and O)
# Hard:   no 3-note chords; no G+O chord
# Medium: only adjacent 2-note chords (1-2); no G+B, G+O, R+O
# Easy:   NO chords

# Chords forbidden in Hard (pairs of fret indices)
HARD_FORBIDDEN_CHORDS = frozenset([
    frozenset({0, 4}),   # G+O
])
HARD_MAX_CHORD_SIZE = 2

# Chords forbidden in Medium (besides size > 2)
MEDIUM_FORBIDDEN_CHORDS = frozenset([
    frozenset({0, 3}),   # G+B
    frozenset({0, 4}),   # G+O
    frozenset({1, 4}),   # R+O
])
MEDIUM_MAX_CHORD_SIZE = 2

# ── Drums ─────────────────────────────────────────────────────────────────────
DRUM_KICK        = {diff: 96 + DIFF_OFFSET[diff]      for diff in DIFF_OFFSET}
DRUM_KICK_EXPERT = 96
DRUM_KICK_2X     = 95   # Expert+ only
DRUM_PADS_EXPERT = {97, 98, 99, 100, 101}   # red,yellow,blue,green,green5
DRUM_EXPERT_PADS = DRUM_PADS_EXPERT          # alias (used in processor.py)
DRUM_TOM_MARKERS = {110, 111, 112}

# Track names → type
TRACK_TYPES: dict[str, str] = {}
for _n in ("PART GUITAR", "PART GUITAR COOP", "PART BASS",
           "PART RHYTHM", "PART KEYS", "T1 GEMS"):
    TRACK_TYPES[_n] = "guitar"
TRACK_TYPES["PART DRUMS"] = "drums"
TRACK_TYPES["PART DRUM"]  = "drums"

"""venue.py — VENUE track generation (camera + lights + post-proc).

Generates an EXPLICIT venue as text events in the modern C3/venuegen format,
derived from the song's structure and accents (see VENUE_SPEC.md, Customs Book
ch. 20). Project philosophy: rules derived mathematically from the tempo_map /
sections / accents — never hard-coded for specific BPMs.

Pipeline:
  parse_sections (EVENTS [section *]) → classify → lighting/postproc per section
  → camera cuts (2-4s pacing from the tempo) → directed cuts + BRE + RANDOM
  → text events sorted on the VENUE track.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import re

import mido
from .midi_utils import AbsEvent, tick_to_ms, ms_to_ticks, measure_ticks_at
from .venue_director import plan_venue, VenueDesign, CLIMAX_PP_FACTOR

# ── Vocabulary (VENUE_SPEC.md) ────────────────────────────────────────────────

# Standard camera cuts → text event [coop_*]
CAMERA_CUTS = {
    # 4 characters (most generic)
    "All_Behind": "coop_all_behind", "All_Far": "coop_all_far",
    "All_Near": "coop_all_near",
    # 3 (no drums)
    "Front_Behind": "coop_front_behind", "Front_Near": "coop_front_near",
    # 1 standard character
    "D_Behind": "coop_d_behind", "D_Near": "coop_d_near",
    "V_Behind": "coop_v_behind", "V_Near": "coop_v_near",
    "B_Behind": "coop_b_behind", "B_Near": "coop_b_near",
    "G_Behind": "coop_g_behind", "G_Near": "coop_g_near",
    "K_Behind": "coop_k_behind", "K_Near": "coop_k_near",
    # 1 character closeup
    "D_Hand": "coop_d_closeup_hand", "D_Head": "coop_d_closeup_head",
    "V_Closeup": "coop_v_closeup",
    "B_Hand": "coop_b_closeup_hand", "B_Head": "coop_b_closeup_head",
    "G_Head": "coop_g_closeup_head", "G_Hand": "coop_g_closeup_hand",
    "K_Hand": "coop_k_closeup_hand", "K_Head": "coop_k_closeup_head",
    # 2 characters (most specific)
    "DV_Near": "coop_dv_near", "BD_Near": "coop_bd_near", "DG_Near": "coop_dg_near",
    "BV_Behind": "coop_bv_behind", "BV_Near": "coop_bv_near",
    "GV_Behind": "coop_gv_behind", "GV_Near": "coop_gv_near",
    "KV_Behind": "coop_kv_behind", "KV_Near": "coop_kv_near",
    "BG_Behind": "coop_bg_behind", "BG_Near": "coop_bg_near",
    "BK_Behind": "coop_bk_behind", "BK_Near": "coop_bk_near",
    "GK_Behind": "coop_gk_behind", "GK_Near": "coop_gk_near",
}

# Directed cuts → text event [directed_*]
DIRECTED_CUTS = {
    "D_All": "directed_all", "D_All_Cam": "directed_all_cam",
    "D_All_LT": "directed_all_lt", "D_All_Yeah": "directed_all_yeah",
    "D_BRE": "directed_brej", "D_BRE_Jump": "directed_brej",
    "D_Drums": "directed_drums", "D_Drums_LT": "directed_drums_lt",
    "D_Drums_NP": "directed_drums_np", "D_Drums_Point": "directed_drums_pnt",
    "D_Drums_KD": "directed_drums_kd",
    "D_Bass": "directed_bass", "D_Bass_NP": "directed_bass_np",
    "D_Bass_Cam": "directed_bass_cam", "D_Bass_CLS": "directed_bass_cls",
    "D_Gtr": "directed_guitar", "D_Gtr_NP": "directed_guitar_np",
    "D_Gtr_CLS": "directed_guitar_cls",
    "D_Gtr_Cam_PR": "directed_guitar_cam_pr", "D_Gtr_Cam_PT": "directed_guitar_cam_pt",
    "D_Keys": "directed_keys", "D_Keys_NP": "directed_keys_np",
    "D_Keys_Cam": "directed_keys_cam",
    "D_Vocals": "directed_vocals", "D_Vox_NP": "directed_vocals_np",
    "D_Vox_CLS": "directed_vocals_cls",
    "D_Vox_Cam_PR": "directed_vocals_cam_pr", "D_Vox_Cam_PT": "directed_vocals_cam_pt",
    "D_Stagedive": "directed_stagedive", "D_Crowdsurf": "directed_crowdsurf",
    "D_Crowd": "directed_crowd", "D_Crowd_Gtr": "directed_crowd_g",
    "D_Crowd_Bass": "directed_crowd_b",
    "D_Duo_Gtr": "directed_duo_guitar", "D_Duo_Bass": "directed_duo_bass",
    "D_Duo_Drums": "directed_duo_drums", "D_Duo_GB": "directed_duo_gb",
    "D_Duo_KV": "directed_duo_kv", "D_Duo_KB": "directed_duo_kb",
    "D_Duo_KG": "directed_duo_kg",
}

# Lighting presets (name → text event [lighting (x)])
LIGHTING_MANUAL = {"verse", "chorus", "manual_cool", "manual_warm", "dischord", "stomp"}
LIGHTING_AUTO = {
    "frenzy", "harmony", "loop_cool", "loop_warm", "silhouettes", "silhouettes_spot",
    "searchlights", "sweep", "strobe_slow", "strobe_fast",
    "blackout_slow", "blackout_fast", "blackout_spot", "flare_slow", "flare_fast", "bre",
}

# Post-processing (.pp)
POSTPROCS = {
    # basics
    "ProFilm_a", "ProFilm_b", "video_a", "film_sepia_ink", "film_silvertone",
    "shitty_tv", "bloom", "film_16mm", "film_b+w", "video_bw", "contrast_a",
    "video_security", "film_blue_filter", "desat_blue", "photocopy",
    # special
    "bright", "posterize", "clean_trails", "film_contrast", "film_contrast_blue",
    "flicker_trails", "desat_posterize_trails", "video_trails", "film_contrast_green",
    "film_contrast_red", "horror_movie_special", "space_woosh", "ProFilm_mirror_a",
    "ProFilm_psychedelic_blue_red", "photo_negative",
}

# ── Section classification ─────────────────────────────────────────────────────

# Canonical section types (test order matters: most specific first)
_SECTION_PATTERNS = [
    ("intro",     ("intro",)),
    ("outro",     ("outro", "end")),
    ("prechorus", ("prechorus", "pre_chorus", "pre-chorus")),
    ("postchorus",("postchorus", "post_chorus")),
    ("chorus",    ("chorus", "refrain", "hook")),
    ("verse",     ("verse",)),
    ("bridge",    ("bridge",)),
    ("solo",      ("solo", "lead")),
    ("build",     ("build", "buildup", "rise")),
    ("drop",      ("drop",)),
    ("breakdown", ("breakdown", "break")),
    ("riff",      ("riff", "main_riff")),
]


def classify_section(name: str) -> str:
    """Classify a practice section name into a canonical type."""
    n = name.lower()
    for canon, keys in _SECTION_PATTERNS:
        if any(k in n for k in keys):
            return canon
    return "default"


@dataclass
class Section:
    start: int          # abs_tick
    end: int            # abs_tick (start of the next section / end of the song)
    name: str           # original name
    kind: str           # canonical type
    energy: str | None = None   # 'calm'/'mid'/'high' — refined by audio; None=structural
    warmth: str | None = None   # 'warm'/'cool' — timbre (audio brightness); None=neutral
    energy_spans: list | None = None   # [(start,end,tier)] sub-section energy (audio); None=use mean


def section_energy(s: "Section") -> str:
    """Effective section energy: the audio-refined one if it exists, otherwise the
    structural one (derived from the type). Centralizes all energy decisions."""
    return s.energy or SECTION_ENERGY.get(s.kind, "calm")


# ── Density-driven fallbacks (no sections / unknown names) ────────────────────
#
# When the song has no practice sections, or the names don't classify, we can't
# rely on the name. We measure the real note DENSITY (derived from the tempo,
# never from BPM) and let the song decide the energy. Project philosophy: rules
# derived mathematically from the song itself.

_BLOCK_MEASURES = 8                 # block size when synthesizing sections
_ENERGY_KIND = {"calm": "verse", "mid": "prechorus", "high": "chorus"}


def _count_onsets(onsets: list[int], a: int, b: int) -> int:
    """Number of onsets in the interval [a, b). `onsets` must be sorted."""
    import bisect
    return bisect.bisect_left(onsets, b) - bisect.bisect_left(onsets, a)


def _tier_thresholds(densities: list[float]) -> tuple[float, float]:
    """Thirds (33%/66%) of the song's density distribution."""
    s = sorted(densities)
    if not s:
        return (0.0, 0.0)
    return s[len(s) // 3], s[2 * len(s) // 3]


def _tier(d: float, lo: float, hi: float) -> str:
    if d <= lo:
        return "calm"
    if d < hi:
        return "mid"
    return "high"


def synthesize_sections(song_end: int, onsets: list[int],
                        time_sig_map: list, tpb: int) -> list[Section]:
    """No markers: slice the song into blocks of _BLOCK_MEASURES measures,
    classify each block by its note density (calm/medium/dense →
    verse/prechorus/chorus) and merge consecutive blocks of the same type."""
    onsets = sorted(onsets)
    blocks: list[tuple[int, int, float]] = []
    t = 0
    while t < song_end:
        mt = measure_ticks_at(t, time_sig_map, tpb)
        end = min(t + mt * _BLOCK_MEASURES, song_end)
        nmeas = max(1.0, (end - t) / mt)
        blocks.append((t, end, _count_onsets(onsets, t, end) / nmeas))
        t = end
    if not blocks:
        return [Section(0, song_end, "default", "default")]
    lo, hi = _tier_thresholds([b[2] for b in blocks])
    secs: list[Section] = []
    for i, (a, b, d) in enumerate(blocks):
        tier = _tier(d, lo, hi)
        kind = _ENERGY_KIND[tier]
        if tier == "calm" and i == 0:
            kind = "intro"
        elif tier == "calm" and i == len(blocks) - 1:
            kind = "outro"
        secs.append(Section(a, b, f"auto_{kind}", kind))
    # Merge contiguous blocks of the same type (avoids the light flickering every 8 measures)
    merged: list[Section] = []
    for s in secs:
        if merged and merged[-1].kind == s.kind:
            merged[-1] = Section(merged[-1].start, s.end, merged[-1].name, s.kind)
        else:
            merged.append(s)
    return merged


def refine_sections(sections: list[Section], onsets: list[int],
                    tpb: int) -> list[Section]:
    """Remap sections with an unknown name (kind=='default') by their density
    relative to the song, instead of assuming calm energy."""
    onsets = sorted(onsets)
    if not onsets or all(s.kind != "default" for s in sections):
        return sections
    dens = [_count_onsets(onsets, s.start, s.end) / max(1.0, (s.end - s.start) / tpb)
            for s in sections]
    lo, hi = _tier_thresholds(dens)
    out: list[Section] = []
    for s, d in zip(sections, dens):
        if s.kind == "default":
            out.append(Section(s.start, s.end, s.name, _ENERGY_KIND[_tier(d, lo, hi)]))
        else:
            out.append(s)
    return out


_SECTION_RE = re.compile(r"\[\s*section\s+(.+?)\s*\]|\[\s*prc_(.+?)\s*\]")


def parse_sections(events: list[AbsEvent], song_end: int) -> list[Section]:
    """Extract practice sections from the EVENTS track (text/marker events)."""
    raw: list[tuple[int, str]] = []
    for ev in events:
        m = ev.msg
        txt = getattr(m, "text", None)
        if m.type in ("text", "marker") and txt:
            mt = _SECTION_RE.search(txt)
            if mt:
                raw.append((ev.abs_tick, mt.group(1) or mt.group(2)))
    raw.sort()
    sections: list[Section] = []
    for i, (tick, name) in enumerate(raw):
        end = raw[i + 1][0] if i + 1 < len(raw) else song_end
        sections.append(Section(tick, end, name, classify_section(name)))
    return sections


# ── Lights + post-proc by genre (themes) and energy tier ──────────────────────
#
# The STRUCTURE (section type) decides the energy tier; the GENRE (theme) decides
# the concrete light/post-proc flavor. Inspired by the Magma themes (VENUE_SPEC §2).

# Energy tier of each section type.
# Calibrated against the 20 official venue learn songs.
# verse/intro/bridge stay calm — the density refinement splits them into play/mellow
# based on onset density, matching the official ~30%/30% split.
SECTION_ENERGY = {
    "intro": "calm", "verse": "calm", "bridge": "calm", "outro": "calm",
    "default": "calm", "prechorus": "mid", "postchorus": "mid", "riff": "mid",
    "chorus": "high", "solo": "high", "build": "high", "drop": "high",
    "breakdown": "high",
}

# Themes by genre. Each tier = (lighting_preset, postproc). pace = cut-rate
# multiplier (metal/punk cut faster; ballads slower).
THEMES = {
    # name: { calm/mid/high: ([light palette], pp), pace }
    # Accent palettes rebalanced against official data (100 songs):
    # - dischord/searchlights reduced in mid/high (were over-represented)
    # - chorus/sweep/harmony added to boost underrepresented presets
    # - frenzy REMOVED from accents (was inflating auto preset count)
    # - stomp kept only in mid (1.4% official frequency)
    "rock":    {"calm": (["manual_warm", "loop_warm", "harmony", "loop_cool"], ["ProFilm_a", "ProFilm_b"]),
                 "mid": (["chorus", "frenzy", "sweep", "searchlights"], ["bright", "contrast_a"]),
                "high": (["strobe_slow", "dischord", "verse"], ["bright", "film_contrast"]), "pace": 1.0},
    "metal":   {"calm": (["silhouettes_spot", "manual_cool", "harmony"], ["film_b+w", "video_bw"]),
                "mid": (["silhouettes_spot", "sweep", "manual_cool"], ["contrast_a", "film_contrast"]),
                "high": (["strobe_slow", "dischord", "verse"], ["photo_negative", "film_contrast_red"]), "pace": 0.85},
    "prog":    {"calm": (["loop_cool", "harmony"], ["film_blue_filter", "desat_blue"]),
                "mid": (["manual_cool", "sweep"], ["film_contrast_blue", "contrast_a"]),
                "high": (["strobe_slow", "dischord", "verse"], ["film_contrast", "bright"]), "pace": 1.0},
    "pop":     {"calm": (["harmony", "loop_warm"], ["ProFilm_a", "bloom"]),
                "mid": (["manual_warm", "frenzy"], ["bright", "ProFilm_b"]),
                "high": (["chorus", "frenzy", "strobe_slow"], ["bloom", "bright"]), "pace": 1.0},
    "punk":    {"calm": (["loop_warm", "manual_warm"], ["video_a", "film_16mm"]),
                "mid": (["chorus", "frenzy", "sweep"], ["video_trails", "contrast_a"]),
                "high": (["frenzy", "strobe_slow", "verse"], ["film_16mm", "video_trails"]), "pace": 0.8},
    "synth":   {"calm": (["silhouettes_spot", "loop_cool", "harmony"], ["desat_blue", "film_blue_filter"]),
                "mid": (["manual_cool", "sweep"], ["ProFilm_psychedelic_blue_red", "desat_blue"]),
                "high": (["frenzy", "strobe_slow", "verse"], ["ProFilm_psychedelic_blue_red", "bloom"]), "pace": 1.0},
    "psych":   {"calm": (["loop_warm", "harmony"], ["posterize", "video_trails"]),
                "mid": (["frenzy", "manual_warm", "stomp"], ["video_trails", "ProFilm_mirror_a"]),
                "high": (["frenzy", "strobe_slow", "verse"], ["ProFilm_mirror_a", "posterize"]), "pace": 1.1},
    "slow":    {"calm": (["silhouettes_spot", "manual_warm", "harmony"], ["film_sepia_ink", "film_silvertone"]),
                "mid": (["harmony", "verse", "stomp"], ["film_silvertone", "ProFilm_a"]),
                "high": (["chorus", "frenzy", "strobe_slow"], ["bright", "ProFilm_a"]), "pace": 1.4},
    "vintage": {"calm": (["manual_warm", "loop_warm", "harmony"], ["film_sepia_ink", "film_16mm"]),
                "mid": (["sweep", "manual_warm", "stomp"], ["film_16mm", "film_contrast"]),
                "high": (["frenzy", "strobe_slow", "dischord"], ["film_contrast", "contrast_a"]), "pace": 1.1},
}
DEFAULT_THEME = "rock"

# Dramatic lights that win over any theme in specific sections.
SPECIAL_LIGHTING = {"build": "strobe_slow", "drop": "blackout_spot"}

# Genre map (keyword in song.ini) → theme.
GENRE_KEYWORDS = [
    (("death", "thrash", "metalcore", "djent", "heavy metal", "black metal",
      "doom", "metal", "hardcore"), "metal"),
    (("prog",), "prog"),
    (("punk", "garage"), "punk"),
    (("synth", "electronic", "edm", "dance", "techno", "house", "chiptune",
      "vocaloid", "j-pop", "jpop", "k-pop", "kpop"), "synth"),
    (("psych", "jam", "stoner"), "psych"),
    (("ballad", "acoustic", "soft"), "slow"),
    (("blues", "country", "folk", "jazz", "soul", "classic", "vintage",
      "oldies"), "vintage"),
    (("pop",), "pop"),
    (("rock", "indie", "alternative", "grunge"), "rock"),
]


def genre_to_theme(genre: str | None) -> str:
    """Map the song.ini genre to a theme. Default = rock."""
    if not genre:
        return DEFAULT_THEME
    g = genre.lower()
    for keys, theme in GENRE_KEYWORDS:
        if any(k in g for k in keys):
            return theme
    return DEFAULT_THEME


def _txt(tick: int, text: str) -> AbsEvent:
    """Create a text event on the VENUE track."""
    return AbsEvent(tick, mido.MetaMessage("text", text=text, time=0))


# Light sub-block (in measures) per energy tier: shorter = the light changes
# faster. Approximates the density of light changes in authored venues.
_LIGHT_BLOCK_MEASURES = {"calm": 4, "mid": 3, "high": 2}

# Light palettes BY SECTION TYPE, derived empirically from the 20 professional
# venues (real frequencies; see the study in CLAUDE.md). The section type is the
# PRIMARY DRIVER (a strong, consistent pattern across genres); the genre enters only
# as a tint (occasional accent). Solves the problem that the per-genre palettes
# covered only ~46% of the vocabulary used.
SECTION_LIGHT_POOL = {
    # Energy-dependent pools from official venue data (100 songs).
    # Calm pools are 4 presets: with hold=3 only positions 0-1 are reliably reached
    # in short sections, so under-represented presets (harmony, silhouettes, flare_slow)
    # are placed at position 0-1.
    "intro": {
        "calm": ["silhouettes_spot", "sweep", "harmony", "loop_cool"],
        "mid":  ["sweep", "verse", "manual_warm", "loop_warm", "loop_cool"],
        "high": ["sweep", "manual_warm", "loop_cool", "manual_cool", "blackout_fast"],
    },
    "verse": {
        "calm": ["silhouettes_spot", "loop_cool", "harmony", "manual_warm"],
        "mid":  ["loop_warm", "silhouettes_spot", "verse", "blackout_fast", "manual_cool"],
        "high": ["manual_cool", "silhouettes_spot", "strobe_fast", "blackout_fast", "manual_warm"],
    },
    "prechorus": {
        "calm": ["flare_slow", "harmony", "silhouettes_spot", "blackout_spot"],
        "mid":  ["manual_cool", "silhouettes_spot", "manual_warm", "frenzy", "strobe_slow"],
        "high": ["sweep", "searchlights", "flare_fast", "dischord", "blackout_fast"],
    },
    "chorus": {
        "calm": ["chorus", "harmony", "silhouettes_spot", "sweep"],
        "mid":  ["chorus", "loop_cool", "frenzy", "searchlights", "silhouettes_spot"],
        "high": ["dischord", "loop_cool", "searchlights", "flare_fast", "frenzy"],
    },
    "postchorus": {
        "calm": ["flare_fast", "silhouettes_spot", "searchlights", "harmony", "sweep"],
        "mid":  ["sweep", "manual_warm", "dischord", "blackout_fast", "flare_slow"],
        "high": ["harmony", "silhouettes_spot", "searchlights", "loop_warm", "dischord"],
    },
    "bridge": {
        "calm": ["silhouettes_spot", "loop_cool", "harmony", "manual_warm"],
        "mid":  ["silhouettes_spot", "silhouettes", "loop_cool", "harmony", "manual_warm"],
        "high": ["sweep", "silhouettes_spot", "loop_cool", "loop_warm", "flare_fast"],
    },
    "solo": {
        "calm": ["loop_warm", "harmony", "silhouettes_spot", "loop_cool"],
        "mid":  ["harmony", "silhouettes_spot", "manual_warm", "flare_fast", "sweep"],
        "high": ["flare_fast", "loop_warm", "searchlights", "loop_cool", "silhouettes_spot"],
    },
    "breakdown": {
        "calm": ["stomp", "silhouettes_spot", "flare_fast", "blackout_fast"],
        "mid":  ["harmony", "strobe_slow", "strobe_fast", "searchlights", "loop_warm"],
        "high": ["blackout_fast", "manual_cool", "frenzy", "strobe_slow", "manual_warm"],
    },
    "build": {
        "calm": ["blackout_spot", "harmony", "flare_slow", "loop_cool"],
        "mid":  ["sweep", "silhouettes_spot", "loop_warm", "blackout_spot", "frenzy"],
        "high": ["searchlights", "frenzy", "silhouettes_spot", "loop_cool", "strobe_fast"],
    },
    "drop": {
        "calm": ["frenzy", "strobe_fast", "flare_fast", "blackout_fast"],
        "mid":  ["frenzy", "strobe_fast", "flare_fast", "blackout_fast"],
        "high": ["frenzy", "strobe_fast", "flare_fast", "blackout_fast"],
    },
    "riff": {
        "calm": ["harmony", "loop_warm", "loop_cool", "sweep"],
        "mid":  ["loop_warm", "frenzy", "searchlights", "loop_cool", "manual_warm"],
        "high": ["loop_cool", "silhouettes_spot", "blackout_spot", "flare_fast", "loop_warm"],
    },
    "outro": {
        "calm": ["stomp", "silhouettes", "flare_fast", "blackout_spot"],
        "mid":  ["flare_fast", "manual_warm", "loop_cool", "loop_warm", "blackout_spot"],
        "high": ["sweep", "blackout_fast", "searchlights", "flare_fast", "manual_warm"],
    },
    "default": {
        "calm": ["manual_warm", "silhouettes_spot", "loop_warm", "blackout_fast"],
        "mid":  ["manual_warm", "flare_fast", "loop_warm", "searchlights"],
        "high": ["flare_fast", "blackout_fast", "manual_warm", "silhouettes_spot"],
    },
}

# Post-procs BY SECTION TYPE. Rebalanced against 30-song audio context study.
# Each pool ordered: official context-optimal filters first (higher cycling frequency).
SECTION_PP_POOL = {
    "intro":      ["film_contrast_red", "clean_trails", "film_sepia_ink", "shitty_tv",
                    "posterize", "film_silvertone", "bloom", "video_bw",
                    "photocopy"],
    "verse":      ["film_contrast_red", "clean_trails", "film_contrast", "ProFilm_b", "film_16mm",
                    "photocopy", "desat_posterize_trails", "video_trails", "shitty_tv", "film_contrast_blue",
                    "film_blue_filter", "ProFilm_a"],
    "prechorus":  ["shitty_tv", "bright", "clean_trails",
                    "film_contrast_red", "flicker_trails", "desat_posterize_trails", "photocopy",
                    "posterize", "ProFilm_a"],
    "chorus":     ["video_trails", "clean_trails", "film_contrast_red", "film_contrast_blue",
                   "film_contrast_green", "flicker_trails", "ProFilm_b", "bloom",
                   "ProFilm_psychedelic_blue_red", "bright"],
    "postchorus": ["film_contrast_red", "film_contrast_blue", "film_contrast_green",
                   "video_trails", "clean_trails", "shitty_tv", "film_contrast",
                   "bright", "video_a"],
    "bridge":     ["clean_trails", "video_trails", "film_contrast_red", "flicker_trails",
                   "film_contrast", "bright", "posterize", "film_blue_filter",
                   "desat_blue", "ProFilm_a"],
    "solo":       ["film_contrast_red", "shitty_tv", "video_trails", "flicker_trails",
                   "desat_posterize_trails", "photocopy", "horror_movie_special"],
    "breakdown":  ["posterize", "clean_trails", "space_woosh", "film_silvertone",
                   "shitty_tv", "horror_movie_special", "photo_negative", "film_contrast_red"],
    "build":      ["film_contrast", "flicker_trails", "clean_trails", "video_trails",
                    "film_contrast_blue", "desat_posterize_trails", "space_woosh", "film_contrast_red", "ProFilm_mirror_a"],
    "drop":       ["film_contrast_red", "flicker_trails", "space_woosh", "photo_negative",
                   "video_a", "horror_movie_special"],
    "riff":       ["film_contrast_red", "shitty_tv", "video_trails", "posterize",
                   "ProFilm_b", "film_16mm", "desat_posterize_trails",
                   "flicker_trails", "contrast_a"],
    "outro":      ["film_contrast_red", "video_security", "shitty_tv", "clean_trails", "film_contrast",
                   "video_a", "space_woosh", "film_silvertone", "film_sepia_ink",
                   "ProFilm_a", "video_bw"],
    "default":    ["film_contrast_red", "flicker_trails", "bright",
                   "film_silvertone", "film_contrast", "ProFilm_a", "ProFilm_psychedelic_blue_red"],
}

# Filter behavioral roles — derived from the 20 official venues study.
# BURST: typically in rapid clusters (<1 beat gaps), intense sections
# SHORT: transições 1-4 beats, prechorus/bridge
# HOLD: seguram ≥4 beats, intro/outro/verse calmo
# None = no strong bias (used in any context)
_PP_FILTER_ROLE = {
    # BURST (cluster-heavy, <1 beat after)
    "film_contrast_red": "burst", "horror_movie_special": "burst",
    "video_security": "burst", "clean_trails": "burst", "shitty_tv": "burst",
    "desat_posterize_trails": "burst", "photo_negative": "burst",
    "bright": "burst", "ProFilm_mirror_a": "burst",
    "ProFilm_psychedelic_blue_red": "burst", "film_blue_filter": "burst",
    # SHORT (1-4 beats after)
    "film_b+w": "short", "film_contrast_blue": "short", "ProFilm_b": "short",
    "photocopy": "short", "bloom": "short",
    # HOLD (≥4 beats after, the one held in a cluster)
    "film_16mm": "hold", "video_a": "hold", "ProFilm_a": "hold",
    "film_silvertone": "hold", "posterize": "hold", "desat_blue": "hold",
    "film_contrast_green": "hold", "film_sepia_ink": "hold", "contrast_a": "hold",
}


def _reorder_pp_pool(pool: list[str], tier: str) -> list[str]:
    """Reorder pool so the tier-appropriate filters come first:
    high energy → burst filters first; calm → hold filters first; mid → short first."""
    role_pref = {"high": "burst", "calm": "hold", "mid": "short"}
    pref = role_pref.get(tier)
    if not pref:
        return pool
    preferred = [f for f in pool if _PP_FILTER_ROLE.get(f) == pref]
    others = [f for f in pool if _PP_FILTER_ROLE.get(f) != pref]
    return preferred + others


# Temperature swap: pulls the temperature-bearing presets toward the section's real
# TIMBRE (audio spectral centroid). A dark/bassy section (low centroid) → 'warm';
# a bright/cymbal/distorted section (high centroid) → 'cool'. The mid third (or no
# audio) stays neutral and the pool is untouched. Song-relative — no absolute colour.
_WARM_OF = {"manual_cool": "manual_warm", "loop_cool": "loop_warm"}
_COOL_OF = {"manual_warm": "manual_cool", "loop_warm": "loop_cool"}


def _env_tier(env: list[tuple[int, str]] | None, tick: int) -> str:
    """Local energy tier at `tick` from a sorted (start_tick, tier) envelope.
    Returns the tier of the rightmost breakpoint <= tick (first one if before all)."""
    if not env:
        return "mid"
    import bisect
    i = bisect.bisect_right([b[0] for b in env], tick) - 1
    return env[max(0, i)][1]


def _warmth_pool(pool: list[str], warmth: str | None) -> list[str]:
    """Bias a light pool toward `warmth` ('warm'/'cool'); keeps length/structure,
    only swaps the warm/cool presets. None → unchanged."""
    if warmth == "warm":
        return [_WARM_OF.get(p, p) for p in pool]
    if warmth == "cool":
        return [_COOL_OF.get(p, p) for p in pool]
    return pool


# Pauses (≥2 measures with no notes) → blackout/silhouette (74% pattern in prof. venues)
_PAUSE_LIGHT = ["blackout_fast", "silhouettes", "blackout_spot", "blackout_slow"]


def _section_lights(theme: dict, s: "Section") -> list[str]:
    """Section's light palette (list to cycle)."""
    if s.kind in SPECIAL_LIGHTING:
        return [SPECIAL_LIGHTING[s.kind]]
    return theme[section_energy(s)][0]


def _section_pps(theme: dict, s: "Section") -> list[str]:
    """Section's post-proc palette (list to cycle)."""
    return theme[section_energy(s)][1]


# Post-proc sub-block (measures): coarser than the light — the screen filter
# shouldn't flicker as fast.
_PP_BLOCK_MEASURES = {"calm": 8, "mid": 6, "high": 4}


# Light model learned from the professional venues (20 charts, see CLAUDE.md):
# switch the PRESET at the drums' rhythm (not keyframes). 69% of the changes fall on
# a drum hit; median cadence 0.25–1 beat in intense parts, 4–8 in calm ones.

# Light-change cadence (beats between switches) per energy tier. Calibrated to the
# AVERAGE density of the professional venues (~1 switch / 1.5–6 beats); their median
# in bursts is lower, but they have long intervals that compensate.
_LIGHT_CADENCE = {"calm": 6.5, "mid": 3.5, "high": 1.9}

# Per-section-kind base cadence (beats between light changes).
# Learned from 20 official venue songs — each section kind has a characteristic
# density regardless of energy tier. The energy factor then adjusts within the kind.
_KIND_CADENCE: dict[str, float] = {
    "verse":     7.5,   # sparse — verse is restrained
    "prechorus": 4.5,   # building up
    "chorus":    2.8,   # dense, the "money" section
    "bridge":    3.5,   # transitional
    "intro":     5.0,   # establishing
    "outro":     4.5,
    "riff":      2.5,   # high-energy riff
    "breakdown": 3.0,
    "build":     2.2,   # peak of tension
    "drop":      1.8,   # maximum intensity
    "default":   4.5,
}

# Energy tier adjustment multiplier (gentler than the flat cadence was)
_ENERGY_FACTOR = {"calm": 1.6, "mid": 1.35, "high": 1.15}

# Base "pulse" presets per tier — the warm/cool/blackout alternation creates the
# strobe effect. The theme flavor (auto presets) enters as an accent.
_LIGHT_PULSE = {
    "calm": ["manual_warm", "manual_cool"],
    "mid": ["manual_warm", "manual_cool", "stomp"],
    "high": ["manual_warm", "manual_cool", "blackout_fast", "stomp"],
}
# Every how many switches a theme accent (auto preset) is inserted.
_LIGHT_ACCENT_EVERY = 5

# Hold (re-emissions of the same preset) per energy tier. Instead of advancing the
# pool every cycle, we re-emit the same preset `hold` times before advancing.
# This adds the 24% same-preset re-emissions that official venues use for "pulsing".
# calm sections hold longer (more re-emissions), high sections advance every cycle.
_LIGHT_HOLD = {"calm": 3, "mid": 2, "high": 1}

# Cluster-then-hold: within each cluster, rapid sub-beat gaps between presets.
# The pattern defines gaps (in beats) between consecutive events in a cluster.
# After the cluster, the last preset is held for _LIGHT_HOLD_BARS bars.
_LIGHT_CLUSTER_PAT = {
    "calm": [0.5],              # 2 events per step: burst(0.5b) gap
    "mid":  [0.5],              # 2 events per step: burst(0.5b) gap
    "high": [0.5, 0.5],         # 3 events per step: burst×2
}

# Audio-driven trigger density calibration
_AUDIO_ACCENT_TOP_PCT = 70     # Keep top 30% of flux peaks (70th percentile = strongest 30%)
_TRIGGER_MIN_GAP_BEATS = 0.5   # Base minimum gap between triggers in beats
_DENSITY_LOW_THRESH = 0.5      # Below this ratio → inject fallback triggers
_DENSITY_HIGH_THRESH = 1.5     # Above this ratio → prune triggers
_DOWNBEAT_INTERVAL = 1         # Every N bars for downbeat triggers

# Markov transition weights learned from 100 official venue learn songs.
# Maps previous_preset → {next_preset: weight}. Covers the 22 presets found
# in the official MIDIs. Weights are raw transition counts ×10 (fractional
# counts smoothed). Self-transitions are included — the hold mechanism already
# re-emits, but the Markov self-weight biases the NEXT cycle when hold expires.
# Presets not listed fall back to uniform selection from the available pool.
_LIGHT_MARKOV: dict[str, dict[str, float]] = {
    "loop_cool": {
        "silhouettes_spot": 278, "dischord": 50, "searchlights": 20,
        "loop_warm": 30, "manual_cool": 15, "strobe_slow": 10,
        "harmony": 15, "verse": 8, "manual_warm": 12, "chorus": 10,
    },
    "silhouettes_spot": {
        "loop_cool": 188, "dischord": 185, "harmony": 60,
        "manual_cool": 55, "manual_warm": 45, "searchlights": 25,
        "loop_warm": 30, "chorus": 15, "verse": 10, "flare_slow": 8,
    },
    "dischord": {
        "loop_cool": 147, "silhouettes_spot": 114, "frenzy": 68,
        "searchlights": 25, "chorus": 20, "harmony": 18,
        "manual_cool": 15, "strobe_fast": 12, "strobe_slow": 10,
        "loop_warm": 10, "manual_warm": 8, "flare_fast": 8,
    },
    "searchlights": {
        "chorus": 73, "strobe_fast": 45, "loop_cool": 20,
        "silhouettes_spot": 25, "dischord": 15, "harmony": 12,
        "frenzy": 12, "verse": 8, "manual_warm": 6, "loop_warm": 6,
    },
    "chorus": {
        "searchlights": 92, "loop_cool": 20, "dischord": 15,
        "silhouettes_spot": 20, "harmony": 15, "frenzy": 15,
        "strobe_fast": 12, "verse": 8, "manual_cool": 8, "loop_warm": 6,
    },
    "harmony": {
        "silhouettes_spot": 59, "loop_warm": 44, "loop_cool": 25,
        "searchlights": 15, "chorus": 12, "dischord": 10,
        "manual_warm": 10, "manual_cool": 8, "verse": 5,
    },
    "frenzy": {
        "dischord": 51, "silhouettes_spot": 25, "loop_cool": 20,
        "searchlights": 15, "chorus": 12, "harmony": 10,
        "strobe_fast": 10, "verse": 5, "manual_cool": 5,
    },
    "loop_warm": {
        "harmony": 32, "loop_cool": 25, "silhouettes_spot": 20,
        "manual_warm": 15, "chorus": 10, "searchlights": 10,
        "dischord": 8, "manual_cool": 8, "verse": 5,
    },
    "manual_warm": {
        "manual_cool": 57, "silhouettes_spot": 51, "loop_cool": 20,
        "harmony": 15, "searchlights": 10, "dischord": 8,
        "chorus": 5, "verse": 5,
    },
    "manual_cool": {
        "manual_warm": 45, "silhouettes_spot": 33, "loop_cool": 25,
        "harmony": 12, "searchlights": 8, "chorus": 5,
    },
    "verse": {
        "sweep": 48, "loop_cool": 10, "silhouettes_spot": 8,
        "dischord": 5, "chorus": 5,
    },
    "sweep": {
        "verse": 49, "loop_cool": 15, "silhouettes_spot": 10,
        "dischord": 5, "chorus": 5, "searchlights": 5,
    },
    "strobe_fast": {
        "searchlights": 44, "loop_cool": 10, "silhouettes_spot": 8,
        "dischord": 8, "chorus": 5, "frenzy": 5,
    },
    "strobe_slow": {
        "silhouettes_spot": 15, "loop_cool": 12, "dischord": 8,
        "searchlights": 8, "harmony": 8, "chorus": 5,
    },
    "flare_fast": {
        "flare_fast": 44, "loop_cool": 8, "silhouettes_spot": 6,
        "dischord": 5,
    },
    "flare_slow": {
        "flare_slow": 57, "loop_cool": 8, "silhouettes_spot": 6,
        "harmony": 5,
    },
    "blackout_fast": {
        "loop_cool": 10, "silhouettes_spot": 8, "dischord": 8,
        "searchlights": 6, "chorus": 5,
    },
    "blackout_spot": {
        "loop_cool": 10, "silhouettes_spot": 8, "dischord": 5,
        "searchlights": 5,
    },
    "blackout_slow": {
        "loop_cool": 8, "silhouettes_spot": 6, "dischord": 5,
    },
    "stomp": {
        "loop_cool": 8, "silhouettes_spot": 6, "dischord": 5,
        "frenzy": 5,
    },
    "silhouettes": {
        "loop_cool": 5, "silhouettes_spot": 5, "dischord": 4,
    },
    "bre": {
        "loop_cool": 3, "silhouettes_spot": 3,
    },
}


def _markov_choice(last: str | None, candidates: list[str],
                   markov_table: dict[str, dict[str, float]] | None = None,
                   default_fallback: str = "loop_cool",
                   rng: random.Random | None = None) -> str:
    """Select from `candidates` weighted by transition probability from `last`.

    Uses the specified `markov_table` (defaults to `_LIGHT_MARKOV`). Falls back
    to the 1st candidate if `last` is None or has no known transitions.
    `rng` (the design's seeded RNG) makes the draw reproducible per song.
    """
    table = markov_table or _LIGHT_MARKOV
    if last is None or not candidates:
        return candidates[0] if candidates else default_fallback
    weights = table.get(last)
    if weights is None:
        return candidates[0] if candidates else default_fallback
    wlist = [weights.get(c, 0.1) for c in candidates]
    total = sum(wlist)
    if total <= 0:
        return candidates[0]
    r = (rng or random).random() * total
    acc = 0.0
    for c, w in zip(candidates, wlist):
        acc += w
        if r <= acc:
            return c
    return candidates[-1]


def _in_span(t: int, spans: list[tuple[int, int]]) -> bool:
    import bisect
    if not spans:
        return False
    starts = [a for a, _ in spans]
    i = bisect.bisect_right(starts, t) - 1
    return 0 <= i < len(spans) and spans[i][0] <= t < spans[i][1]


def _gather_lighting_triggers(section, audio_onsets, drum_onsets, energy_env,
                               time_sig_map, tpb, energy) -> list[int]:
    """Gather trigger ticks from audio signals."""
    triggers = set()
    triggers.add(section.start)  # P1: section boundary

    # P2: Audio flux accents (keep top 30%)
    if audio_onsets:
        for tick in audio_onsets:
            if section.start <= tick < section.end:
                triggers.add(tick)

    # P3: Downbeat grid
    for t_sig, num, den in time_sig_map:
        if t_sig > section.end:
            break
        bar_ticks = tpb * 4  # assume 4/4 as base
        t = t_sig
        while t < section.end:
            if t >= section.start:
                triggers.add(t)
            t += bar_ticks

    # P4: Energy tier changes from audio envelope
    if energy_env:
        for tick, _ in energy_env:
            if section.start <= tick < section.end:
                triggers.add(tick)

    # P5: Drum accents on-beat
    if drum_onsets:
        for tick in drum_onsets:
            if section.start <= tick < section.end:
                nearest_beat = round(tick / tpb) * tpb
                if abs(tick - nearest_beat) <= tpb // 8:
                    triggers.add(tick)

    # Merge & deduplicate with min-gap
    sorted_triggers = sorted(triggers)
    min_gap_ticks = max(tpb // 4, int(tpb * _TRIGGER_MIN_GAP_BEATS))
    merged = []
    for tick in sorted_triggers:
        if not merged or tick - merged[-1] >= min_gap_ticks:
            merged.append(tick)

    # Density calibration
    kind_cad = _KIND_CADENCE.get(section.kind, _KIND_CADENCE["default"])
    energy_fac = _ENERGY_FACTOR.get(energy, 1.0)
    section_beats = (section.end - section.start) / tpb if tpb else 1
    cadence_events = max(1, section_beats / (kind_cad * energy_fac)) if kind_cad * energy_fac else 1
    target_triggers = max(1, int(cadence_events))

    if not merged:
        return merged

    density_ratio = len(merged) / target_triggers if target_triggers else 1

    if density_ratio > _DENSITY_HIGH_THRESH:
        # Too many: keep only P1 (section boundary) + evenly sample rest
        keep = [merged[0]]  # Always keep section start
        keep_count = min(target_triggers, len(merged))
        step = len(merged) / keep_count
        for i in range(1, keep_count):
            idx = min(int(i * step), len(merged) - 1)
            keep.append(merged[idx])
        merged = sorted(set(keep))

    return merged


def _cadence_triggers(section, tpb, kind_cad, energy_fac) -> list[int]:
    """Fallback: generate triggers at regular cadence intervals."""
    step = max(tpb // 8, int(tpb * kind_cad * energy_fac))
    triggers = []
    if step <= 0:
        step = tpb
    t = section.start
    while t < section.end:
        triggers.append(t)
        t += step
    return triggers


def build_lighting(sections: list[Section], theme: dict, tpb: int,
                   time_sig_map: list,
                   drum_onsets: list[int] | None = None,
                   pause_spans: list[tuple[int, int]] | None = None,
                   strobe_spans: list[tuple[int, int]] | None = None,
                   audio_onsets: list[int] | None = None,
                   energy_env: list[tuple[int, str]] | None = None,
                   drop_ticks: list[int] | None = None,
                   design: "VenueDesign | None" = None) -> list[AbsEvent]:
    """Professional-style lightshow, DRIVEN BY THE SECTION TYPE (the primary pattern
    learned from the 20 venues). Cycles the section pool (SECTION_LIGHT_POOL) at the
    drums' rhythm and sprinkles a genre accent (theme) every _LIGHT_ACCENT_EVERY
    switches (secondary tint). In pauses (≥2 measures), blackout/silhouette.
    In double-pedal/blast bursts (strobe_spans), pins CONTINUOUS strobe_fast
    ('Painkiller' effect). Cadence by energy (refined by audio)."""
    out: list[AbsEvent] = []
    light_events: list[tuple[int, str]] = []   # (tick, preset) to generate keyframes
    rng = design.rng if design is not None else None
    drums = sorted(drum_onsets) if drum_onsets else []
    pause_spans = pause_spans or []
    strobe_spans = strobe_spans or []
    audio_onsets = sorted(audio_onsets) if audio_onsets else []
    env = sorted(energy_env) if energy_env else None
    pi = 0
    last: str | None = None
    for si, s in enumerate(sections):
        energy = section_energy(s)
        pool_by_kind = SECTION_LIGHT_POOL.get(s.kind, SECTION_LIGHT_POOL["default"])
        if isinstance(pool_by_kind, dict):
            base = _warmth_pool(pool_by_kind.get(energy, pool_by_kind.get("calm", [])), s.warmth)
        else:
            base = _warmth_pool(pool_by_kind, s.warmth)
        accents = ([SPECIAL_LIGHTING[s.kind]] if s.kind in SPECIAL_LIGHTING
                   else _warmth_pool(_section_lights(theme, s), s.warmth))
        # ── Motivo por grupo (repetição estrutural, dev/repetition_stats.json:
        # oficiais reutilizam o look — Jaccard same-group 0.53 vs 0.18, 1.º preset
        # igual 57%). 1.ª ocorrência GRAVA a sequência; repetições REPETEM-NA.
        # Energia divergente da 1.ª ocorrência → replay só do 1.º preset.
        motif: list[str] | None = None
        recording = False
        if design is not None:
            group = design.group_at(si)
            if design.is_first_occurrence(si):
                recording = True
                group.light_motif = []
            elif group.light_motif:
                first_energy = section_energy(sections[group.section_idxs[0]])
                motif = (group.light_motif if energy == first_energy
                         else group.light_motif[:1])
        mk = 0                          # posição no motivo em replay
        placed: list[int] = []
        # ── Gather trigger ticks from audio signals ──
        triggers = _gather_lighting_triggers(s, audio_onsets, drum_onsets, env,
                                              time_sig_map, tpb, energy)
        # ── Fallback to cadence if no triggers or no audio ──
        use_triggers = (len(audio_onsets) > 0 and len(triggers) >= 2)
        if not use_triggers:
            triggers = _cadence_triggers(s, tpb, _KIND_CADENCE.get(s.kind, _KIND_CADENCE["default"]),
                                         _ENERGY_FACTOR.get(energy, 1.0))
        # ── Âncora partilhada: o 1.º evento da secção cai no downbeat que a
        # câmara e o pp também usam (oficiais: offset mediano 0.00 beats).
        if design is not None:
            anchor = design.anchors[si]
            if s.start <= anchor < s.end:
                triggers = [anchor] + [t for t in triggers if t > anchor]
        # ── Emit clusters at triggers ──
        i = 0
        for tick in triggers:
            if tick < s.start or tick >= s.end:
                continue
            # Check strobe/pause
            if _in_span(tick, strobe_spans):
                last = "strobe_fast"
                continue
            if _in_span(tick, pause_spans):
                preset = _PAUSE_LIGHT[pi % len(_PAUSE_LIGHT)]
                pi += 1
                out.append(_txt(tick, f"[lighting ({preset})]"))
                light_events.append((tick, preset))
                placed.append(tick)
                last = preset
                continue
            # ── Mini-cluster: emit 2-3 rapid changes per step ──
            local = _env_tier(env, tick) if env else energy
            cluster_gaps = _LIGHT_CLUSTER_PAT.get(local, [0.5])
            # Clímax (último chorus): +1 flip por cluster ≈ a densidade 1.57×
            # dos oficiais (arc_stats.json) — o pico é luz, não pyro.
            if design is not None and si == design.climax_idx:
                cluster_gaps = list(cluster_gaps) + [0.5]
            tt = tick
            for gi in range(len(cluster_gaps) + 1):
                if tt >= s.end:
                    break
                if motif is not None and mk < len(motif):
                    preset = motif[mk]           # replay do look do grupo
                elif i % _LIGHT_ACCENT_EVERY == _LIGHT_ACCENT_EVERY - 1:
                    preset = accents[(i // _LIGHT_ACCENT_EVERY) % len(accents)]
                else:
                    preset = _markov_choice(last, base, rng=rng)
                if recording:
                    group.light_motif.append(preset)
                mk += 1
                out.append(_txt(tt, f"[lighting ({preset})]"))
                light_events.append((tt, preset))
                placed.append(tt)
                last = preset
                i += 1
                if gi < len(cluster_gaps):
                    tt += max(1, int(tpb * cluster_gaps[gi]))
    # Keyframes [next]: the MANUAL presets (verse/chorus/manual_*/dischord/stomp) are
    # STATIC until a keyframe advances them. The official venues keyframe them ~1×
    # per beat (snap to hits). AUTO presets (frenzy/flare/loop/strobe…) animate
    # themselves → no keyframes. Without this, our manual light stayed frozen.
    out += _build_light_keyframes(light_events, sections, tpb, drums,
                                  pause_spans, strobe_spans)
    # 'Painkiller' strobe: pins strobe_fast at the start of the burst. Does NOT emit
    # 'strobe_off' (a token the official ones never use) — the light resumes on its own
    # because the main loop marks last="strobe_fast" during the burst and re-emits the
    # section preset right after the end of the span (persistent state, like the
    # official venues).
    for a, b in strobe_spans:
        out.append(_txt(a, "[lighting (strobe_fast)]"))
    # Audio drops (intensity collapse after a loud stretch) → punch a blackout_spot at
    # the moment everything cuts out, the classic build->drop look. blackout_spot is an
    # AUTO preset (self-animating) so it needs no keyframe; the next section/cadence
    # change resumes the show.
    for d in (drop_ticks or []):
        if not _in_span(d, pause_spans) and not _in_span(d, strobe_spans):
            out.append(_txt(d, "[lighting (blackout_spot)]"))
    return out


# Keyframe cadence (in BEATS between [next]) by section energy — mirrors the Magma
# themes' per-section `keyframe_rate`. The section's light already pulses at the
# energy-driven _LIGHT_CADENCE; these keyframes animate the MANUAL presets that
# persist, so we DENSIFY high-energy parts (½ beat) rather than starve calm ones —
# choruses pulse faster, calm verses stay gentle, none go silent. Floored at 1/4 beat.
_KEYFRAME_RATE = {"high": 0.55, "mid": 0.9, "calm": 1.8}


def _energy_for_tick(sections: list["Section"], tick: int) -> str:
    """Effective energy of the section containing `tick` (defaults to 'calm')."""
    for s in sections:
        if s.start <= tick < s.end:
            return section_energy(s)
    return "calm"


def _build_light_keyframes(light_events: list[tuple[int, str]],
                           sections: list["Section"], tpb: int,
                           drums: list[int],
                           pause_spans: list[tuple[int, int]],
                           strobe_spans: list[tuple[int, int]]) -> list[AbsEvent]:
    """Generate the `[next]` that make the MANUAL presets advance (snap to the nearest
    hit ±1/4 beat), from the preset's tick to the next light change. The cadence now
    follows the section energy (`_KEYFRAME_RATE`): 1 beat in high-energy sections,
    every 2 in mid, every 4 in calm — instead of a flat 1×/beat everywhere. Does not
    keyframe inside pauses/strobe nor auto presets. Pattern of the official venues."""
    if not light_events:
        return []
    song_end = sections[-1].end if sections else 0
    le = sorted(light_events, key=lambda x: x[0])
    out: list[AbsEvent] = []
    for idx, (tick, name) in enumerate(le):
        if name not in LIGHTING_MANUAL:
            continue
        nxt = le[idx + 1][0] if idx + 1 < len(le) else song_end
        step = max(tpb // 4, int(tpb * _KEYFRAME_RATE.get(
            _energy_for_tick(sections, tick), 1.0)))
        b = tick + step
        while b < nxt:
            if not _in_span(b, pause_spans) and not _in_span(b, strobe_spans):
                kk = _nearest(b, drums, tpb // 4, floor=tick + 1) if drums else None
                kk = kk if kk is not None else b
                if tick < kk < nxt:
                    out.append(_txt(kk, "[next]"))
            b += step
    return out


def _fast_runs(onsets: list[int], fast_gap: int, min_span: int) -> list[tuple[int, int]]:
    """Runs of onsets whose consecutive gap <= `fast_gap` and that last >= `min_span`."""
    d = sorted(onsets or [])
    if len(d) < 3:
        return []
    spans: list[tuple[int, int]] = []
    run_start = prev = d[0]
    for t in d[1:]:
        if t - prev <= fast_gap:
            prev = t
            continue
        if prev - run_start >= min_span:
            spans.append((run_start, prev))
        run_start = prev = t
    if prev - run_start >= min_span:
        spans.append((run_start, prev))
    return spans


def _merge_spans(spans: list[tuple[int, int]], bridge: int) -> list[tuple[int, int]]:
    """Sort and coalesce spans whose gap is < `bridge` (also merges overlaps)."""
    merged: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if merged and a - merged[-1][1] < bridge:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def find_strobe_spans(drum_onsets: list[int], tpb: int,
                      dbass_onsets: list[int] | None = None) -> list[tuple[int, int]]:
    """'Painkiller'-style strobe spans: fast and SUSTAINED bursts deserve
    CONTINUOUS [lighting (strobe_fast)] instead of the normal cycling. Two sources:
      1. SNARE+TOMS+kicks (fills / consecutive snares) — gap <= ~1/4 beat (16th).
      2. DOUBLE BASS (`dbass_onsets`, note 95) — looser gap (~1/2 beat, 8th),
         because sustained double-bass at 8ths is already 'blast' and calls for strobe
         even without being so fast. Does NOT include cymbals/hi-hat (filtered upstream).
    `min_span` = 1.75 beats (between the dense 1.5 and the old 2.0).
    Runs separated by < 1 beat merge. Everything derived from beat fractions."""
    min_span = int(tpb * 1.75)             # sustained >= 1.75 beats (intermediate)
    bridge = tpb                           # merges runs with a gap < 1 beat
    spans = _fast_runs(drum_onsets, int(tpb / 4 * 1.12), min_span)        # 16th
    if dbass_onsets:
        spans += _fast_runs(dbass_onsets, int(tpb / 2 * 1.1), min_span)   # 8th dbass
    return _merge_spans(spans, bridge)


def find_pause_spans(onsets: list[int], time_sig_map: list, tpb: int,
                     min_measures: int = 2) -> list[tuple[int, int]]:
    """Pause spans: gaps ≥ min_measures measures with no notes at all."""
    onsets = sorted(onsets)
    spans: list[tuple[int, int]] = []
    for a, b in zip(onsets, onsets[1:]):
        if b - a >= measure_ticks_at(a, time_sig_map, tpb) * min_measures:
            spans.append((a + tpb, b))     # starts 1 beat after the last note
    return spans


# ── Spotlights + Sing-along (notes on the VENUE track, learned from prof. venues) ─
#
# Note-map decoded from the 20 professional venues: notes 37–41 = spotlight per
# member (keys=41 confirmed by absence; ascending order), 85–87 = sing-along of the
# 3 harmonies. These are NOTES (not text) on the VENUE track.

SPOT_NOTE = {"drums": 37, "bass": 38, "guitar": 39, "vocal": 40, "keys": 41}
SINGALONG_NOTES = [87, 86, 85]   # harmony 1, 2, 3


def _note_span(start: int, end: int, note: int) -> list[AbsEvent]:
    """note_on/note_off pair for a sustained span on the VENUE track."""
    if end <= start:
        end = start + 1
    return [
        AbsEvent(start, mido.Message("note_on", note=note, velocity=100, time=0)),
        AbsEvent(end, mido.Message("note_off", note=note, velocity=0, time=0)),
    ]


# Featured member per section type (the spotlight follows whoever "leads"). Solo uses
# the soloist (detected by name). Learned from the venues: vocal in the sung parts,
# guitar in riffs/breakdowns, etc.
_FEATURED_INST = {
    "verse": "vocal", "prechorus": "vocal", "chorus": "vocal",
    "postchorus": "vocal", "bridge": "vocal", "intro": "guitar",
    "riff": "guitar", "breakdown": "guitar", "build": "guitar",
    "drop": "guitar", "outro": "vocal", "default": "vocal",
}


def _phrases(onsets: list[int], start: int, end: int, gap: int) -> list[tuple[int, int]]:
    """Split the onsets of [start,end) into PHRASES (runs separated by a gap ≥ `gap`).
    Each phrase becomes a spotlight span — replicates the per-phrase use of prof. venues."""
    import bisect
    lo = bisect.bisect_left(onsets, start)
    hi = bisect.bisect_left(onsets, end)
    seg = onsets[lo:hi]
    if not seg:
        return []
    spans: list[tuple[int, int]] = []
    run_a = prev = seg[0]
    for t in seg[1:]:
        if t - prev >= gap:
            spans.append((run_a, prev))
            run_a = t
        prev = t
    spans.append((run_a, prev))
    return spans


def build_spotlights(sections: list[Section], inst_onsets: dict[str, list[int]],
                     tpb: int) -> list[AbsEvent]:
    """Spotlight EVERY member that plays, each following ITS OWN phrasing (several
    beams at once, like the official venues → a ~balanced distribution over the 5 notes
    37-41 — drums/bass/guitar/vocal/keys). Each phrase (run of notes separated by
    ≥ `gap` of silence) lights that member's spotlight during the phrase.
    The gap is set to ~1 beat so we get a natural pulse (more phrase-starts = more
    note_on events, matching official density ~225 events/song)."""
    out: list[AbsEvent] = []
    gap = tpb * 2                       # ~2 beat gap → natural pulse, matches official density ~225/song
    # gap calibration: 3b→0.60×, 2b→0.79×, 1b→2.27×. Sweet spot near 2b.
    members = [m for m in SPOT_NOTE if inst_onsets.get(m)]
    for s in sections:
        for inst in members:
            ons = sorted(inst_onsets[inst])
            for a, b in _phrases(ons, s.start, s.end, gap):
                out += _note_span(a, min(b + tpb // 2, s.end), SPOT_NOTE[inst])
    return out


def build_singalong(sections: list[Section], vocal_onsets: list[int],
                    n_harm: int, tpb: int) -> list[AbsEvent]:
    """Sing-along (crowd/band sings) in the chorus sections with REAL vocals active.
    Uses as many lines as there are harmonies (1–3); with no HARM tracks it uses 1 line
    — the official venues use sing-along even without authored harmonies (~24/song)."""
    if not vocal_onsets:
        return []
    vo = sorted(vocal_onsets)
    notes = SINGALONG_NOTES[:max(1, min(3, n_harm))]
    out: list[AbsEvent] = []
    import bisect
    # Study of the 20 official ones: sing-along is NOT only in the chorus — it spreads
    # over any section with singable vocals (chorus 48%, verse 24%, solo/prechorus/
    # bridge/breakdown/outro the rest). We replicate it: eligible in all sections with
    # sustained vocals; intro/build (lead-in) are left out.
    _SING_KINDS = {"chorus", "postchorus", "verse"}
    for s in sections:
        if s.kind not in _SING_KINDS:
            continue
        lo = bisect.bisect_left(vo, s.start)
        hi = bisect.bisect_left(vo, s.end)
        if hi - lo < 2:                       # needs vocals in the section
            continue
        a, b = vo[lo], min(vo[hi - 1] + tpb, s.end)
        for n in notes:
            out += _note_span(a, b, n)
    return out


# ── Pyrotechnics / show effects ───────────────────────────────────────────────

# Climax sections (receive sparse pyro even at 'mid' energy).
_PYRO_SECTIONS = {"chorus", "drop", "build", "breakdown", "riff", "solo"}


def build_pyro(sections: list[Section], drum_onsets: list[int],
               tpb: int, accents: list[int] | None = None,
               design: "VenueDesign | None" = None) -> list[AbsEvent]:
    """[bonusfx]/[bonusfx_optional] driven by the real INTENSITY and the hits, not by
    the structure. The official venues vary hugely (0 in calm/wall-of-sound songs,
    20–40 in metal with stabs). Derived rule:
      • 'high' sections → dense pyro (~1 per measure), on each band ACCENT;
      • climax 'mid' sections → sparse pyro (marks the start);
      • 'calm' sections → nothing.
    Scales itself: songs with no intense sections get ~no pyro; metal with many high
    sections fills up. Each hit snaps to the nearest accent/drum hit. ~1 in 3 is
    `_optional` (proportion of the heavy official ones). Spaced ≥ ~3/4 measure.
    NOTE: the choice of 0 pyro by some official ones (The Who, BABYMETAL, Deafheaven) is
    stylistic/period, NOT derivable from the MIDI (drum density doesn't separate them) —
    so the excess is limited (per-section cap + dense only in climax), but those can't
    be zeroed without harming legitimate metal."""
    out: list[AbsEvent] = []
    drum_onsets = sorted(drum_onsets) if drum_onsets else []
    accents = sorted(accents) if accents else []
    last = -10 ** 9
    min_gap = tpb * 8                         # ≥ 2 measures between pyros
    per_section_cap = 1                       # at most 1 per section
    n = 0
    for s in sections:
        e = section_energy(s)
        if e == "high" and s.kind in _PYRO_SECTIONS:
            step = tpb * 12                   # ~1 every 3 measures
        else:
            continue                          # only high+climax → pyro
        placed_here = 0
        t = s.start
        while t < s.end and placed_here < per_section_cap:
            tick = _nearest(t, accents, tpb, floor=s.start)         # band accent
            if tick is None:
                tick = _nearest(t, drum_onsets, tpb // 2, floor=s.start)
            if tick is None:
                tick = t
            if tick < s.end and tick - last >= min_gap:
                out.append(_txt(tick, "[bonusfx_optional]" if n % 3 == 2 else "[bonusfx]"))
                last = tick
                n += 1
                placed_here += 1
            t += step
    # Pyro parity no clímax: se a secção de clímax ficou sem pyro mas a música
    # tem pyro noutras secções, injetar 1 evento para paridade (não escalada).
    if design is not None and design.climax_idx is not None:
        climax_has = any(
            ev.abs_tick >= sections[design.climax_idx].start
            and ev.abs_tick < sections[design.climax_idx].end
            for ev in out
        )
        if not climax_has and len(out) > 0:
            s = sections[design.climax_idx]
            anchor = design.anchors[design.climax_idx]
            if s.start <= anchor < s.end:
                out.append(_txt(anchor, "[bonusfx_optional]"))
    return out


# Post-proc placement model — CLUSTER-then-HOLD, matching the 20 official venues.
# The pp_study found the official gap distribution is strongly BIMODAL: median gap
# 0.7 beats but mean 5.0, i.e. 56% of changes are bursts (<1 beat apart) and 20% are
# long holds (≥4 beats), with only ~0.17 pp/beat overall. The originals do NOT space
# single changes evenly — they flip a small CLUSTER of filters rapidly at an anchor
# (section/energy change), then HOLD one filter for many beats until the next anchor.
#
#  _PP_CLUSTER  : how many rapid changes fire at each anchor (more when intense).
#  _PP_CLUSTER_SUBDIV : beats between the changes inside a cluster (sub-beat = burst,
#                       one beat = short gap — tuned per tier to match the 56/24/20 mix).
#  _PP_HOLD     : beats from one cluster anchor to the next (the long hold).
# Each anchor fires a CLUSTER whose internal gaps follow this beat PATTERN (the first
# change is on the downbeat anchor; each value is the gap in beats to the next change
# in the same cluster). Mixing 0.5-beat steps (bursts <1) with ~2-beat steps (short
# gaps 1-4) reproduces the official 56% burst / 24% short split; the long inter-anchor
# hold supplies the 20% holds. More/tighter changes when the local energy is high.
_PP_CLUSTER_PAT = {
    "calm": [0.5, 2.0],                 # 3 changes: 1 burst + 1 short & hold
    "mid":  [0.5, 0.5, 2.0],            # 4 changes: 2 burst + 1 short & hold
    "high": [0.5, 0.5, 2.0],            # 4 changes: 2 burst + 1 short & hold
}
# Hold to the next anchor, in BARS — longer gaps between clusters = more holds
# (gap ≥4 beats). Calibrated to match official density per tier:
#   calm: ~0.50 events/bar (was 0.38, under-generating)
#   high: ~0.67 events/bar (was 1.00, over-generating)
_PP_HOLD_BARS = {"calm": 6, "mid": 6, "high": 6}

# Per-section-kind hold DISTORTION.  Some section kinds are much sparser (solo) or
# denser (build) than the tier average in the official venues.  These coefficients
# adjust the hold length for those outliers only.  1.0 = same as the tier default.
# Calibrated against the 30-song audio context study ratios.
_PP_HOLD_KINDS = {
    "solo":       {"high": 1.70},
    "riff":       {"mid":  1.70},
    "bridge":     {"calm": 1.80},
    "postchorus": {"mid":  0.55},
    "build":      {"high": 0.75},
    "outro":      {"calm": 0.75},
}

# Inside an audio strobe/blast WALL the originals flip even faster and longer — we
# extend the cluster to span the wall (capped at one bar) on the half-beat.
_PP_BURST_SUBDIV = 0.5

# Markov transition weights for post-processing filters.
# Learned from 20 official venue songs (1,905 transitions).
# 37.2% self-transition rate overall. Weights are derived from raw counts.
# Key families: Film Contrast (green↔blue 76%), TV/Silvertone, Trails (43-47% self),
# Desat/Stylized, Effects.
_PP_MARKOV: dict[str, dict[str, float]] = {
    # ── Film Contrast Family ────────────────────────────
    "film_contrast_green": {
        "film_contrast_blue": 113, "film_contrast_green": 9,
        "film_contrast": 8, "film_contrast_red": 5, "bright": 3,
    },
    "film_contrast_blue": {
        "film_contrast_green": 112, "film_contrast_blue": 23,
        "film_contrast": 10, "film_contrast_red": 5, "video_trails": 4,
    },
    "film_contrast_red": {
        "film_contrast_red": 10, "film_contrast": 8, "film_contrast_blue": 5,
        "clean_trails": 5, "film_contrast_green": 4, "bright": 3,
    },
    "film_contrast": {
        "film_contrast": 50, "film_contrast_green": 11, "film_contrast_blue": 8,
        "film_contrast_red": 6, "clean_trails": 5, "bright": 4,
    },
    # ── TV / Silvertone Family ──────────────────────────
    "shitty_tv": {
        "shitty_tv": 60, "film_silvertone": 44, "ProFilm_b": 20,
        "ProFilm_a": 19, "film_contrast_red": 5, "bright": 4,
        "clean_trails": 4, "video_trails": 3,
    },
    "film_silvertone": {
        "film_silvertone": 36, "shitty_tv": 44, "ProFilm_a": 5,
        "ProFilm_b": 3, "clean_trails": 2,
    },
    "ProFilm_a": {
        "ProFilm_a": 46, "shitty_tv": 13, "ProFilm_b": 8,
        "flicker_trails": 6, "bright": 5, "clean_trails": 5,
        "desat_posterize_trails": 4, "video_trails": 4,
    },
    "ProFilm_b": {
        "ProFilm_b": 37, "video_trails": 29, "clean_trails": 26,
        "shitty_tv": 20, "bright": 6, "film_contrast_red": 5,
        "desat_posterize_trails": 4, "ProFilm_a": 4,
    },
    # ── Trails Family ───────────────────────────────────
    "video_trails": {
        "video_trails": 104, "clean_trails": 57, "bright": 12,
        "ProFilm_b": 8, "flicker_trails": 6, "desat_posterize_trails": 5,
        "film_contrast_blue": 4, "shitty_tv": 4,
    },
    "clean_trails": {
        "clean_trails": 97, "video_trails": 39, "ProFilm_b": 26,
        "film_contrast_red": 8, "bright": 6, "flicker_trails": 5,
        "desat_posterize_trails": 5, "shitty_tv": 4,
    },
    "flicker_trails": {
        "flicker_trails": 35, "clean_trails": 8, "video_trails": 6,
        "desat_posterize_trails": 5, "bright": 4, "ProFilm_a": 4,
    },
    "desat_posterize_trails": {
        "desat_posterize_trails": 39, "video_trails": 8, "clean_trails": 6,
        "posterize": 6, "desat_blue": 5, "shitty_tv": 4,
        "bright": 3, "flicker_trails": 3,
    },
    # ── Desat / Stylized Family ────────────────────────
    "desat_blue": {
        "desat_posterize_trails": 6, "desat_blue": 1,
        "posterize": 2, "clean_trails": 1,
    },
    "posterize": {
        "posterize": 11, "desat_posterize_trails": 8, "bright": 4,
        "video_trails": 3, "clean_trails": 3, "bloom": 2,
    },
    "bloom": {
        "bloom": 7, "film_contrast_red": 4, "clean_trails": 3,
        "bright": 2, "film_contrast": 2,
    },
    "space_woosh": {
        "space_woosh": 15, "flicker_trails": 3, "video_trails": 3,
        "clean_trails": 3, "bright": 2,
    },
    # ── Bright ──────────────────────────────────────────
    "bright": {
        "bright": 55, "clean_trails": 8, "video_trails": 6,
        "ProFilm_b": 6, "film_contrast_red": 5, "shitty_tv": 4,
        "posterize": 4, "desat_posterize_trails": 3,
    },
    # ── Effects Family ──────────────────────────────────
    "photocopy": {
        "photocopy": 20, "clean_trails": 5, "video_trails": 4,
        "bright": 3, "shitty_tv": 3,
    },
    "photo_negative": {
        "photo_negative": 14, "desat_posterize_trails": 3,
        "video_trails": 3, "clean_trails": 2, "bright": 2,
    },
    "video_a": {
        "video_a": 14, "clean_trails": 3, "video_trails": 3,
        "bright": 2, "shitty_tv": 2,
    },
    "video_bw": {
        "video_bw": 12, "video_trails": 4, "clean_trails": 3,
        "bright": 2, "shitty_tv": 2,
    },
    "horror_movie_special": {
        "horror_movie_special": 5, "video_trails": 2, "clean_trails": 2,
        "bright": 1, "shitty_tv": 1,
    },
    "film_sepia_ink": {
        "film_sepia_ink": 3, "shitty_tv": 2,
    },
    "video_security": {
        "video_security": 1, "clean_trails": 1,
    },
    "ProFilm_mirror_a": {
        "ProFilm_mirror_a": 1, "ProFilm_a": 1,
    },
    "contrast_a": {
        "contrast_a": 2, "film_sepia_ink": 2, "clean_trails": 1,
    },
    "film_blue_filter": {
        "clean_trails": 3, "film_blue_filter": 2, "film_contrast_blue": 1,
    },
}

# Markov transition weights for camera cuts (coop_* shots only).
# Learned from 20 official venue songs. Uses SHORT names (pool format, e.g. "V_Near"),
# matching the SECTION_CAMERA pool entries. Key patterns:
#   • Pair switching: GV_Near ↔ BV_Near, G_Hand ↔ B_Hand, BG_Near ↔ GK_Near
#   • Medium shots (Near) → mostly other medium shots
#   • Wide shots (All_Behind, All_Far) → tend to stay wide (28% self-transition)
#   • Drum closeups (D_Hand) → tend to repeat (14% self-transition)
#   • Vocals (V_Near/V_Closeup) → rarely repeat, mostly go to other members
_CAMERA_MARKOV: dict[str, dict[str, float]] = {
    "All_Behind": {"V_Near": 37, "D_Near": 33, "D_Hand": 32, "V_Closeup": 32, "All_Far": 30, "G_Hand": 25, "BV_Near": 25, "All_Near": 23, "D_Head": 23, "D_Behind": 23, "B_Near": 19, "BG_Near": 18, "BK_Near": 16, "B_Hand": 16, "GV_Near": 16},
    "All_Far": {"V_Near": 59, "D_Near": 58, "D_Hand": 52, "All_Behind": 48, "G_Hand": 48, "V_Closeup": 42, "All_Near": 38, "Front_Near": 32, "K_Hand": 30, "G_Near": 28, "D_Behind": 23, "BG_Near": 22, "B_Near": 16, "BK_Near": 16, "K_Near": 16},
    "All_Near": {"D_Hand": 62, "V_Near": 58, "V_Closeup": 48, "D_Near": 45, "All_Far": 45, "G_Hand": 42, "G_Near": 38, "B_Near": 37, "K_Hand": 36, "D_Behind": 30, "Front_Near": 28, "All_Behind": 26, "B_Hand": 25, "D_Head": 25, "BV_Near": 22},
    "BD_Near": {"All_Near": 33, "V_Closeup": 33, "V_Near": 27, "Front_Near": 21, "G_Near": 19, "All_Behind": 19, "BV_Near": 16, "Front_Behind": 16, "All_Far": 14, "KV_Near": 13, "B_Behind": 13, "B_Hand": 12, "GV_Near": 11, "K_Behind": 11, "V_Behind": 10},
    "BG_Behind": {"V_Closeup": 31, "V_Near": 31, "D_Near": 24, "D_Hand": 23, "D_Head": 17, "B_Near": 17, "All_Far": 14, "G_Hand": 13, "Front_Near": 12, "GV_Near": 11, "All_Near": 11, "BV_Near": 10, "G_Near": 10, "D_Behind": 8, "K_Hand": 8},
    "BG_Near": {"V_Closeup": 52, "V_Near": 50, "D_Hand": 50, "D_Near": 47, "G_Hand": 34, "B_Hand": 31, "All_Near": 30, "G_Near": 29, "D_Behind": 28, "D_Head": 28, "Front_Near": 27, "B_Near": 26, "DV_Near": 21, "K_Near": 21, "All_Behind": 19},
    "BK_Behind": {"V_Near": 28, "D_Near": 24, "V_Closeup": 22, "D_Hand": 17, "B_Near": 15, "All_Far": 13, "D_Head": 13, "Front_Near": 13, "D_Behind": 12, "GV_Near": 10, "All_Near": 10, "BV_Near": 9, "K_Near": 9, "K_Hand": 8, "G_Near": 8},
    "BK_Near": {"V_Closeup": 43, "D_Near": 43, "D_Hand": 42, "V_Near": 39, "B_Hand": 33, "D_Behind": 28, "B_Near": 25, "K_Hand": 24, "D_Head": 23, "Front_Near": 21, "All_Near": 20, "K_Near": 20, "G_Hand": 20, "DV_Near": 19, "G_Near": 18},
    "BV_Behind": {"D_Near": 16, "V_Near": 15, "V_Closeup": 15, "Front_Near": 14, "D_Head": 12, "All_Near": 12, "B_Near": 9, "G_Hand": 6, "D_Behind": 5, "G_Near": 5, "B_Hand": 5, "BG_Near": 4, "BK_Near": 4, "GK_Near": 4, "All_Far": 4},
    "BV_Near": {"V_Closeup": 57, "D_Near": 49, "V_Near": 49, "D_Head": 33, "BG_Near": 29, "GK_Near": 28, "D_Hand": 28, "BK_Near": 27, "B_Hand": 24, "D_Behind": 24, "B_Near": 24, "Front_Near": 23, "All_Behind": 22, "All_Near": 21, "Front_Behind": 20},
    "B_Behind": {"D_Near": 27, "Front_Near": 24, "V_Closeup": 24, "V_Near": 22, "D_Hand": 21, "B_Hand": 21, "B_Near": 19, "G_Hand": 18, "D_Behind": 13, "G_Near": 12, "K_Hand": 12, "BG_Near": 11, "K_Near": 10, "BK_Near": 10, "GK_Near": 10},
    "B_Hand": {"V_Closeup": 100, "D_Hand": 91, "V_Near": 60, "D_Near": 37, "K_Hand": 34, "B_Near": 34, "G_Hand": 33, "Front_Near": 32, "G_Near": 29, "D_Behind": 28, "BV_Near": 25, "BK_Near": 23, "All_Near": 23, "BG_Near": 22, "GK_Near": 21},
    "B_Head": {"D_Hand": 38, "V_Near": 23, "B_Near": 21, "D_Head": 21, "V_Closeup": 19, "D_Near": 12, "G_Near": 12, "K_Head": 10, "GV_Near": 10, "B_Hand": 10, "D_Behind": 10, "G_Head": 9, "All_Near": 9, "G_Hand": 9, "Front_Near": 9},
    "B_Near": {"V_Near": 69, "V_Closeup": 67, "D_Hand": 60, "D_Behind": 59, "D_Near": 57, "K_Near": 51, "B_Hand": 38, "G_Near": 37, "Front_Near": 36, "G_Hand": 35, "D_Head": 35, "BG_Near": 33, "BK_Behind": 28, "GK_Behind": 28, "BG_Behind": 27},
    "DG_Near": {"V_Closeup": 25, "Front_Near": 22, "V_Near": 18, "All_Far": 18, "All_Near": 17, "All_Behind": 17, "BV_Near": 16, "B_Near": 15, "B_Behind": 14, "Front_Behind": 13, "G_Near": 11, "K_Near": 10, "BG_Near": 10, "B_Hand": 9, "K_Behind": 9},
    "DV_Near": {"BG_Near": 22, "V_Closeup": 19, "B_Near": 18, "GK_Near": 18, "All_Near": 17, "BK_Near": 17, "V_Near": 17, "Front_Near": 13, "All_Far": 13, "G_Near": 10, "All_Behind": 9, "BV_Near": 8, "G_Hand": 8, "B_Hand": 8, "BG_Behind": 8},
    "D_Behind": {"V_Near": 57, "V_Closeup": 45, "BV_Near": 40, "Front_Near": 40, "G_Near": 39, "B_Near": 39, "All_Near": 36, "BG_Near": 28, "GK_Near": 27, "B_Hand": 27, "All_Far": 26, "BK_Near": 24, "KV_Near": 22, "G_Hand": 22, "All_Behind": 20},
    "D_Hand": {"V_Closeup": 107, "B_Hand": 104, "V_Near": 85, "G_Hand": 79, "Front_Near": 62, "B_Near": 62, "All_Far": 59, "All_Near": 55, "G_Near": 49, "BG_Near": 48, "BV_Near": 46, "BK_Near": 40, "K_Hand": 38, "D_Near": 37, "GK_Near": 32},
    "D_Head": {"V_Closeup": 52, "V_Near": 35, "All_Far": 33, "All_Near": 32, "Front_Near": 31, "B_Head": 31, "BV_Near": 30, "B_Near": 30, "D_Hand": 25, "GV_Near": 24, "BG_Near": 22, "G_Near": 20, "B_Hand": 18, "GK_Near": 17, "BK_Near": 17},
    "D_Near": {"V_Closeup": 88, "All_Near": 77, "V_Near": 67, "G_Hand": 64, "B_Near": 58, "D_Hand": 55, "B_Hand": 50, "BG_Near": 50, "G_Near": 48, "Front_Near": 45, "BV_Near": 44, "BK_Near": 44, "All_Behind": 38, "GK_Near": 38, "K_Near": 34},
    "Front_Behind": {"V_Near": 41, "D_Near": 33, "D_Hand": 30, "V_Closeup": 26, "G_Hand": 24, "B_Hand": 22, "B_Near": 21, "All_Near": 21, "G_Near": 21, "BV_Near": 20, "GV_Near": 16, "D_Behind": 16, "BG_Near": 16, "BK_Near": 14, "K_Hand": 14},
    "Front_Near": {"V_Near": 65, "B_Near": 51, "D_Hand": 50, "D_Near": 49, "V_Closeup": 45, "B_Hand": 41, "D_Behind": 36, "G_Hand": 34, "BV_Near": 34, "K_Hand": 32, "G_Near": 32, "D_Head": 32, "DG_Near": 28, "All_Behind": 26, "K_Near": 25},
    "GK_Behind": {"V_Near": 28, "D_Near": 24, "V_Closeup": 23, "B_Near": 16, "D_Hand": 16, "D_Head": 14, "Front_Near": 12, "All_Far": 11, "K_Near": 10, "D_Behind": 9, "GV_Near": 9, "All_Near": 9, "BV_Near": 9, "G_Near": 9, "K_Hand": 8},
    "GK_Near": {"D_Hand": 40, "D_Near": 38, "V_Near": 36, "V_Closeup": 34, "B_Hand": 30, "B_Near": 25, "D_Behind": 23, "Front_Near": 23, "D_Head": 23, "K_Hand": 21, "G_Near": 20, "G_Hand": 19, "All_Near": 18, "DV_Near": 18, "K_Near": 18},
    "GV_Behind": {"V_Closeup": 12, "D_Near": 9, "All_Far": 7, "V_Near": 7, "All_Near": 7, "D_Behind": 6, "D_Hand": 6, "Front_Near": 5, "BD_Near": 5, "All_Behind": 5, "B_Near": 4, "D_Head": 3, "V_Behind": 3, "K_Near": 3, "Front_Behind": 3},
    "GV_Near": {"V_Closeup": 44, "D_Near": 38, "V_Near": 32, "D_Head": 25, "B_Near": 22, "All_Near": 20, "All_Behind": 19, "D_Behind": 17, "BD_Near": 16, "B_Behind": 16, "Front_Near": 15, "B_Hand": 14, "BG_Near": 12, "K_Behind": 11, "K_Near": 11},
    "G_Behind": {"B_Near": 17, "Front_Near": 15, "V_Closeup": 13, "D_Near": 13, "D_Head": 11, "V_Near": 11, "B_Hand": 10, "K_Near": 10, "BG_Near": 10, "D_Hand": 9, "BK_Near": 9, "GK_Near": 9, "D_Behind": 8, "K_Hand": 8, "All_Behind": 8},
    "G_Hand": {"D_Hand": 125, "B_Hand": 82, "V_Closeup": 49, "V_Near": 47, "D_Near": 39, "All_Near": 36, "D_Behind": 35, "Front_Near": 34, "B_Near": 26, "BG_Near": 24, "G_Near": 24, "K_Hand": 24, "All_Far": 23, "BV_Near": 23, "Front_Behind": 22},
    "G_Head": {"D_Hand": 34, "V_Closeup": 23, "V_Near": 17, "B_Head": 15, "D_Near": 15, "Front_Near": 14, "K_Near": 14, "D_Head": 13, "K_Head": 12, "D_Behind": 11, "All_Near": 11, "BG_Near": 9, "B_Near": 9, "DG_Near": 7, "BK_Near": 6},
    "G_Near": {"V_Near": 65, "D_Near": 59, "B_Near": 50, "V_Closeup": 48, "D_Hand": 47, "D_Behind": 39, "K_Near": 38, "Front_Near": 34, "D_Head": 27, "B_Hand": 27, "BG_Near": 27, "BG_Behind": 24, "K_Hand": 21, "All_Near": 20, "B_Behind": 19},
    "KV_Behind": {"Front_Near": 10, "V_Closeup": 8, "V_Near": 8, "D_Head": 7, "B_Near": 5, "G_Hand": 5, "G_Near": 4, "D_Near": 4, "All_Near": 3, "B_Hand": 3, "D_Behind": 3},
    "KV_Near": {"V_Near": 28, "V_Closeup": 28, "D_Near": 23, "BG_Near": 19, "GK_Near": 17, "BK_Near": 17, "D_Behind": 14, "B_Hand": 13, "G_Hand": 12, "V_Behind": 12, "D_Hand": 12, "Front_Near": 11, "All_Far": 11, "DG_Near": 9, "G_Near": 9},
    "K_Behind": {"V_Closeup": 18, "V_Near": 18, "D_Near": 18, "D_Hand": 18, "B_Hand": 12, "Front_Near": 12, "All_Near": 12, "G_Hand": 11, "B_Near": 11, "K_Hand": 10, "D_Behind": 9, "G_Near": 8, "All_Far": 7, "BV_Near": 7, "DG_Near": 7},
    "K_Hand": {"D_Hand": 61, "G_Hand": 38, "V_Near": 28, "V_Closeup": 27, "D_Behind": 27, "G_Near": 26, "B_Near": 26, "Front_Near": 25, "D_Near": 21, "B_Hand": 21, "All_Behind": 19, "All_Far": 17, "All_Near": 16, "BD_Near": 16, "GV_Near": 14},
    "K_Head": {"D_Hand": 17, "V_Near": 14, "V_Closeup": 14, "G_Near": 12, "B_Near": 11, "B_Hand": 10, "G_Hand": 9, "G_Head": 9, "B_Head": 8, "D_Behind": 8, "D_Near": 7, "GV_Near": 7, "BV_Near": 7, "Front_Behind": 7, "Front_Near": 6},
    "K_Near": {"D_Near": 40, "V_Closeup": 36, "V_Near": 35, "D_Hand": 34, "B_Near": 32, "G_Near": 27, "B_Hand": 27, "D_Behind": 25, "G_Hand": 21, "BV_Near": 17, "BK_Near": 16, "All_Near": 16, "Front_Near": 15, "G_Head": 15, "BG_Near": 15},
    "V_Behind": {"D_Near": 26, "V_Closeup": 25, "B_Near": 24, "D_Hand": 19, "BG_Near": 18, "D_Head": 17, "G_Near": 16, "Front_Near": 14, "BK_Near": 13, "GK_Near": 13, "D_Behind": 12, "B_Hand": 12, "V_Near": 11, "K_Hand": 11, "All_Near": 11},
    "V_Closeup": {"D_Near": 95, "D_Hand": 90, "B_Near": 79, "G_Hand": 71, "Front_Near": 67, "B_Hand": 63, "All_Far": 56, "G_Near": 56, "BG_Near": 54, "All_Near": 52, "D_Head": 49, "D_Behind": 45, "K_Near": 42, "K_Hand": 40, "V_Near": 40},
    "V_Near": {"D_Near": 90, "B_Near": 77, "B_Hand": 67, "V_Closeup": 60, "BG_Near": 58, "Front_Near": 58, "G_Near": 54, "D_Hand": 51, "BV_Near": 49, "BD_Near": 47, "G_Hand": 46, "Front_Behind": 44, "BK_Near": 43, "All_Behind": 42, "GK_Near": 42},
}

# Camera cut levels from Onyx VenueGen (0=wide, 8=most specific).
# Transitions must go DOWN in level (or stay at 0), never up.
# This prevents jumping from a key closeup directly to a guitar closeup
# without going through a wider shot first.
_CAMERA_LEVELS: dict[str, int] = {
    # Nível 8 — Keys (mais específicos)
    "K_Near": 8, "K_Behind": 8, "K_Head": 8, "K_Hand": 8,
    # Nível 7 — Guitar + Keys
    "GK_Near": 7, "GK_Behind": 7,
    # Nível 6 — Bass + Keys
    "BK_Near": 6, "BK_Behind": 6,
    # Nível 5 — Bass + Guitar
    "BG_Near": 5, "BG_Behind": 5,
    # Nível 4 — Keys + Vocal
    "KV_Near": 4, "KV_Behind": 4,
    # Nível 3 — Guitar + Vocal
    "GV_Near": 3, "GV_Behind": 3,
    # Nível 2 — Bass+Vocal duos, Guitar shots
    "BV_Near": 2, "BV_Behind": 2,
    "DG_Near": 2,
    "G_Near": 2, "G_Behind": 2, "G_Head": 2, "G_Hand": 2,
    # Nível 1 — Bass shots
    "BD_Near": 1,
    "B_Near": 1, "B_Behind": 1, "B_Head": 1, "B_Hand": 1,
    # Nível 0 — Drums, Vocal, group shots (wide)
    "DV_Near": 0,
    "V_Near": 0, "V_Behind": 0, "V_Closeup": 0,
    "D_Near": 0, "D_Behind": 0, "D_Head": 0, "D_Hand": 0,
    "Front_Near": 0, "Front_Behind": 0,
    "All_Near": 0, "All_Far": 0, "All_Behind": 0,
    # Directed cuts are special: level 0 (always allowed)
    "D_Gtr": 0, "D_Gtr_CLS": 0, "D_Gtr_Cam_PR": 0, "D_Gtr_Cam_PT": 0,
    "D_Bass": 0, "D_Bass_CLS": 0, "D_Bass_Cam": 0,
    "D_Drums": 0, "D_Drums_Point": 0, "D_Drums_KD": 0, "D_Drums_LT": 0, "D_Drums_CLS": 0,
    "D_Keys": 0, "D_Keys_Cam": 0,
    "D_Vocals": 0, "D_Vox_CLS": 0, "D_Vox_Cam_PR": 0, "D_Vox_Cam_PT": 0,
    "D_All": 0, "D_All_Cam": 0, "D_All_LT": 0, "D_All_Yeah": 0,
    "D_Stagedive": 0, "D_Crowdsurf": 0,
    "D_Crowd": 0, "D_Crowd_Gtr": 0, "D_Crowd_Bass": 0,
    "D_BRE": 0, "D_BRE_Jump": 0,
    "D_Duo_GB": 0, "D_Duo_KB": 0, "D_Duo_KG": 0, "D_Duo_KV": 0,
    "D_Duo_Gtr": 0, "D_Duo_Bass": 0, "D_Duo_Drums": 0,
    "D_Gtr_NP": 0, "D_Bass_NP": 0, "D_Drums_NP": 0, "D_Keys_NP": 0, "D_Vox_NP": 0,
}

# Post-proc TONE bias by audio timbre (Section.warmth). The pp_study over the 20
# official venues showed B&W/desaturated filters sit in DARK, quieter audio
# (bright_p 39) while bright/contrast/filmic filters sit in BRIGHT, loud audio
# (bright_p 60-67). So a 'cool' (dark) section pulls desaturated presets to the
# front of its pool, a 'warm' (bright) section pulls bright/contrast ones. Same
# pp set, just reordered (keeps the calibrated frequency budget); neutral/no-audio
# section = pool untouched (no-op). Song-relative — no absolute brightness.
_PP_DARK = ("b+w", "bw", "sepia", "silvertone", "photocopy", "desat", "16mm",
            "security", "negative")
_PP_BRIGHT = ("clean", "contrast", "bloom", "profilm")


def _pp_tone_pool(pool: list[str], warmth: str | None) -> list[str]:
    if not warmth:
        return pool
    pref = _PP_DARK if warmth == "cool" else _PP_BRIGHT
    # stable sort: preferred-family filters first, original order preserved within groups
    return sorted(pool, key=lambda p: 0 if any(k in p.lower() for k in pref) else 1)


def _beat_len_at(tick: int, time_sig_map: list, tpb: int) -> int:
    """Length of one beat (the time-sig denominator unit) in ticks at `tick`."""
    num, den = 4, 4
    for t, n, d in time_sig_map:
        if t <= tick:
            num, den = n, d
        else:
            break
    return max(1, tpb * 4 // den)


def _first_beat_at_or_after(start: int, time_sig_map: list, tpb: int) -> int:
    """Snap `start` UP to the nearest downbeat-aligned beat boundary. The grid is
    measured from the active time-signature's start tick so changes land on the
    bar/beat the way the official venues do (their pp sit ~56% on a beat, ~32% on a
    downbeat — and the sparse songs are ~96% on a beat)."""
    sig_start, num, den = 0, 4, 4
    for t, n, d in time_sig_map:
        if t <= start:
            sig_start, num, den = t, n, d
        else:
            break
    beat = max(1, tpb * 4 // den)
    rel = start - sig_start
    k = (rel + beat - 1) // beat        # ceil to the next beat
    return sig_start + k * beat


def _first_downbeat_at_or_after(start: int, time_sig_map: list, tpb: int) -> tuple[int, int]:
    """Snap `start` UP to the nearest bar boundary (downbeat) of the active time-sig,
    measured from that sig's start tick. Returns (downbeat_tick, bar_ticks)."""
    sig_start, num, den = 0, 4, 4
    for t, n, d in time_sig_map:
        if t <= start:
            sig_start, num, den = t, n, d
        else:
            break
    bar = max(1, (tpb * 4 // den) * num)
    rel = start - sig_start
    k = (rel + bar - 1) // bar          # ceil to the next bar
    return sig_start + k * bar, bar


def build_postproc(sections: list[Section], theme: dict, tpb: int,
                   time_sig_map: list,
                   drum_onsets: list[int] | None = None,
                   energy_env: list | None = None,
                   strobe_spans: list | None = None,
                   design: "VenueDesign | None" = None) -> list[AbsEvent]:
    """Professional-style post-proc, placed the way the 20 official venues do it
    (pp_study, dev/_venue_pp_study.py): CLUSTER-then-HOLD.

      * At each anchor (snapped to the beat grid) a small CLUSTER of filters flips
        rapidly on a sub-beat subdivision — the bursts the originals do (56% of their
        gaps are <1 beat). The cluster is bigger / tighter when the local energy is
        high, smaller / on-beat (short gaps) when calm.
      * After the cluster the last filter is HELD for many beats (the 20% of official
        gaps that are ≥4 beats) until the next anchor — keeping ~0.17 pp/beat overall.
      * Inside an audio strobe/blast WALL the cluster is extended to span the wall
        (capped at one bar) for the fast flicker the originals use during blasts.
    """
    out: list[AbsEvent] = []
    env = sorted(energy_env) if energy_env else None
    walls = sorted(strobe_spans) if strobe_spans else []
    rng = design.rng if design is not None else None
    last: str | None = None

    for si, s in enumerate(sections):
        # Pool BY SECTION TYPE (primary); fallback to the genre palette.
        palette = SECTION_PP_POOL.get(s.kind) or _section_pps(theme, s)
        # Audio-timbre tone bias (dark->desaturated, bright->bright/contrast).
        palette = _pp_tone_pool(palette, s.warmth)
        if not palette:
            continue
        # Reorder by energy tier: burst filters first in high, hold first in calm.
        sect_energy = section_energy(s)
        palette = _reorder_pp_pool(palette, sect_energy)
        # Look dominante do grupo (pp_hold): o último filtro de cada cluster é
        # forçado a este filtro, dando coerência visual às secções repetidas.
        pp_hold = None
        if design is not None:
            group = design.group_at(si)
            if group.pp_hold is None:
                group.pp_hold = (rng.choice(palette[:3]) if rng is not None
                                 and len(palette) >= 3 else palette[0])
                if group.pp_hold not in design.dominant_pp:
                    design.dominant_pp.append(group.pp_hold)
            pp_hold = group.pp_hold
        # Timbre gate: the pp_study shows loudness does NOT predict pp density
        # (corr≈0.04). The one cue that separates the two faces of a loud wall is
        # TIMBRE: a BRIGHT/aggressive wall (metalcore, e.g. BMTH — 836 pp) gets dense
        # flicker, while a DARK/atmospheric wall (post-rock, e.g. Deafheaven — 56 pp)
        # the artist leaves almost untouched. So a 'cool' (dark) section is demoted one
        # energy notch for pp purposes (high->mid->calm: smaller clusters, longer holds)
        # and gets NO strobe-wall flicker; a 'warm' (bright) section keeps its tier.
        demote = (s.warmth == "cool")
        t, bar = _first_downbeat_at_or_after(s.start, time_sig_map, tpb)
        if design is not None:
            t = max(t, design.anchors[si]) if design.anchors[si] < s.end else t
        while t < s.end:
            beat = _beat_len_at(t, time_sig_map, tpb)
            # Local tier: audio envelope if present, else the section's mean tier.
            tier = _env_tier(env, t) if env else sect_energy
            if demote:
                tier = {"high": "mid", "mid": "calm", "calm": "calm"}[tier]

            # Cluster gap pattern for this tier; inside an audio strobe wall add one
            # extra half-beat flip — but ONLY on bright/aggressive walls (a dark
            # atmospheric wall is held, not flickered).
            pattern = list(_PP_CLUSTER_PAT[tier])
            if not demote:
                for ws, we in walls:
                    if ws <= t < we:
                        pattern = [_PP_BURST_SUBDIV] + pattern
                        break

            # ── Cluster: first change on the downbeat, the rest stepped by `pattern` ──
            tt = t
            for gi in range(len(pattern) + 1):
                if tt >= s.end:
                    break
                # Markov-weighted selection from the available filter pool,
                # biased by the previous filter's transition probabilities.
                # The LAST emission of each cluster is forced to the group's
                # dominant look (pp_hold) for visual coherence.
                if pp_hold is not None and gi == len(pattern):
                    pp = pp_hold
                else:
                    pp = _markov_choice(last, palette, _PP_MARKOV, "clean_trails",
                                        rng=rng)
                out.append(_txt(tt, f"[{pp}.pp]"))
                last = pp
                if gi < len(pattern):
                    tt += max(1, int(beat * pattern[gi]))

            # ── Hold to the next downbeat anchor (whole bars), filter held across it ──
            hb = _PP_HOLD_BARS[tier]
            if design is not None and si == design.climax_idx:
                hb = max(1, round(hb / CLIMAX_PP_FACTOR))
            kind_coeff = _PP_HOLD_KINDS.get(s.kind, {}).get(tier, 1.0)
            t += max(int(hb * bar * kind_coeff), ((tt - t) // bar + 1) * bar)
    out.sort(key=lambda e: e.abs_tick)
    return out


# ── Camera ────────────────────────────────────────────────────────────────────

# Cut rate (seconds per cut) per section type. Intense sections cut faster.
# Converted to ticks via the tempo_map (never hard-coded BPM).
# Tightened values after the study of the 20 official ones: the chorus (1.3) already
# hit 97% of the official density, but the sparse sections (verse/solo/riff/intro/outro)
# were at 36-61% — the official ones cut the camera faster even there. Moderate
# tightening of the laggers, keeping the order "intense sections cut faster".
SECTION_PACE_S = {
    "intro": 1.8, "verse": 1.5, "prechorus": 1.4, "chorus": 1.3,
    "postchorus": 1.5, "bridge": 1.5, "solo": 1.0, "build": 1.5,
    "drop": 1.3, "breakdown": 1.2, "riff": 1.1, "outro": 1.7, "default": 1.5,
}

# Cut pool per section type (mix of standard + directed). The generator cycles
# avoiding repeating the previous cut. Directed items add dramatic variety.
#
# REAL cadence of the official venues (measured on the learn songs): ~87% framing coop_*,
# only ~13% directed. So each pool is DOMINATED by framing (All_/X_Near/Behind/
# closeups/duos) with 1 directed as a sparse accent — the generator cycles avoiding
# repeating the previous cut, so the directed appears ~1 in every 6-7 cuts. The directed
# ones remain guarded by _guard_directed (_NP if not playing; crowd only with vocals).
# FRAMING (coop) pools per section — distribution calibrated to the official ones (closeup
# ~37% > single_near > duo_near > behind > group > far). The directed cut is NOT here;
# it comes from the event layer (see cut_events.detect_events + build_camera PASS 2).
# Per-instrument distribution RE-BALANCED to the 20 official ones: there the camera is
# split evenly (bass 70 ≳ drums 65 ≈ guitar 62 > vocals 52 > keys 36 per song). Before
# we hammered the vocalist (V_Near+V_Closeup in almost the whole section) and forgot the
# bassist (only in duos) and the keys → vocalist 86, bass 21, keys 6.
# Now each section gives SINGLE presence to bass/drums/keys and ≤1 vocal cut.
# Framing pools per section type. ENRICHED with the under-used head-closeup and behind
# variants after the coop variety study (the 20 official venues use a MEDIAN of 33
# DISTINCT coop framings/song with the top one only ~11%; we sat at 12 distinct / 24%).
# Bigger pools + the anti-recency window (recent_coop) widen the variety per song and
# pull the top-cut share down toward the official ~11%.
SECTION_CAMERA = {
    "intro":      ["All_Far", "B_Near", "All_Behind", "D_Near", "K_Hand", "G_Hand", "V_Closeup", "Front_Near", "G_Behind", "D_Head", "B_Behind"],
    "verse":      ["V_Near", "B_Near", "B_Hand", "D_Hand", "K_Near", "G_Near", "DV_Near", "V_Closeup", "G_Head", "D_Head", "B_Behind", "K_Hand"],
    "prechorus":  ["B_Near", "K_Hand", "V_Near", "All_Near", "D_Hand", "K_Near", "BG_Near", "G_Near", "V_Closeup", "B_Head", "D_Behind"],
    "chorus":     ["V_Near", "B_Near", "All_Near", "D_Hand", "K_Near", "G_Near", "All_Behind", "KV_Near", "V_Closeup", "G_Hand", "B_Hand", "D_Head", "Front_Near"],
    "postchorus": ["B_Near", "All_Behind", "K_Near", "V_Closeup", "D_Near", "Front_Behind", "All_Near", "G_Near", "B_Hand", "D_Hand", "V_Near"],
    "bridge":     ["B_Near", "B_Hand", "K_Near", "K_Hand", "D_Near", "BK_Near", "KV_Near", "B_Head", "K_Head", "G_Near", "D_Behind"],
    "build":      ["D_Near", "D_Hand", "B_Near", "K_Near", "BD_Near", "All_Near", "D_Head", "B_Hand", "G_Near", "All_Behind"],
    "drop":       ["All_Near", "D_Hand", "B_Near", "K_Near", "V_Near", "All_Behind", "D_Near", "G_Hand", "D_Head", "B_Behind"],
    "breakdown":  ["D_Hand", "D_Near", "B_Near", "D_Head", "BK_Near", "DG_Near", "All_Behind", "G_Hand", "B_Head", "K_Near", "All_Near"],
    "riff":       ["G_Hand", "G_Near", "B_Near", "D_Hand", "K_Near", "GK_Near", "BG_Near", "G_Head", "G_Behind", "D_Head", "B_Hand"],
    "outro":      ["All_Far", "All_Behind", "B_Near", "D_Near", "K_Near", "V_Closeup", "All_Near", "Front_Behind", "G_Behind", "B_Head", "D_Behind"],
    "default":    ["V_Near", "B_Near", "D_Hand", "K_Near", "G_Near", "All_Near", "V_Closeup", "B_Hand", "D_Head", "G_Head"],
}

# Framings that involve each instrument — only valid if that instrument exists
# (charted / with a signal); otherwise they would film an absent character. Removed from
# the pool in build_camera (_absent_framings). Includes the DUO framings: BK_Near
# (bass+keys) drops if EITHER of the two is missing. All_*/Front_* (group shots) always stay.
_KEYS_FRAMINGS = {"K_Near", "K_Behind", "K_Hand", "K_Head", "BK_Near", "BK_Behind",
                  "GK_Near", "GK_Behind", "KV_Near", "KV_Behind"}
_BASS_FRAMINGS = {"B_Near", "B_Behind", "B_Hand", "B_Head", "BD_Near", "BV_Behind",
                  "BV_Near", "BG_Behind", "BG_Near", "BK_Behind", "BK_Near"}
_GUITAR_FRAMINGS = {"G_Near", "G_Behind", "G_Hand", "G_Head", "DG_Near", "GV_Behind",
                    "GV_Near", "BG_Behind", "BG_Near", "GK_Behind", "GK_Near"}
_DRUMS_FRAMINGS = {"D_Near", "D_Behind", "D_Hand", "D_Head", "DV_Near", "BD_Near",
                   "DG_Near"}
# Vocals enter the SAME filter (the camera should not point at a nonexistent vocalist),
# but with the most permissive rule: any vocal SIGNAL (chart, lyrics OR audio 'lead'
# → present in inst_onsets["vocal"]) keeps them. They are the exception that can gain
# presence from the audio alone; the other instruments require a real chart.
_VOCAL_FRAMINGS = {"V_Near", "V_Behind", "V_Closeup", "DV_Near", "BV_Behind", "BV_Near",
                   "GV_Behind", "GV_Near", "KV_Behind", "KV_Near"}


# GROUP shots (full-band) — always valid: they film the whole stage, the absent ones
# simply don't appear. Safe fallback when a section pool becomes empty after removing
# the framings of absent instruments.
_GROUP_FRAMINGS = ["All_Near", "All_Far", "All_Behind"]

# Mapa: framing cut → instrumento(s) que filma.
# Usado pelo _playing_framing para saber se o instrumento está a tocar.
# Single-instrument: um set com 1 elemento.
# Multi-instrument: set com vários (basta UM estar a tocar — lógica OR).
# Group shots: set vazio (sempre válidos, filmam o palco todo).
_COOP_INSTR: dict[str, set[str]] = {
    # Vocals
    "V_Near": {"vocal"}, "V_Behind": {"vocal"}, "V_Closeup": {"vocal"},
    # Bass
    "B_Near": {"bass"}, "B_Behind": {"bass"}, "B_Hand": {"bass"}, "B_Head": {"bass"},
    # Drums
    "D_Near": {"drums"}, "D_Behind": {"drums"}, "D_Hand": {"drums"}, "D_Head": {"drums"},
    # Keys
    "K_Near": {"keys"}, "K_Behind": {"keys"}, "K_Hand": {"keys"}, "K_Head": {"keys"},
    # Guitar
    "G_Near": {"guitar"}, "G_Behind": {"guitar"}, "G_Hand": {"guitar"}, "G_Head": {"guitar"},
    # Multi-instrument (duos)
    "DV_Near": {"drums", "vocal"}, "BV_Near": {"bass", "vocal"},
    "GV_Near": {"guitar", "vocal"}, "KV_Near": {"keys", "vocal"},
    "BK_Near": {"bass", "keys"}, "BK_Behind": {"bass", "keys"},
    "BG_Near": {"bass", "guitar"}, "BG_Behind": {"bass", "guitar"},
    "DG_Near": {"drums", "guitar"}, "BD_Near": {"bass", "drums"},
    "GK_Near": {"guitar", "keys"}, "GK_Behind": {"guitar", "keys"},
    "BV_Behind": {"bass", "vocal"}, "BV_Near": {"bass", "vocal"},
    "GV_Behind": {"guitar", "vocal"}, "KV_Behind": {"keys", "vocal"},
    "BD_Near": {"bass", "drums"},
    # Group shots (always valid)
    "All_Near": set(), "All_Far": set(), "All_Behind": set(),
    "Front_Near": set(), "Front_Behind": set(),
}


def _playing_framing(pool: list[str], tick: int,
                     inst_onsets: dict[str, list[int]] | None,
                     tpb: int) -> list[str]:
    """Filter framing pool: keep only cuts where featured instrument(s) are playing.

    Single-instrument cut → that instrument must have an onset within ±2 beats.
    Multi-instrument cut  → AT LEAST ONE instrument must be playing (OR logic).
    Group/All shots       → always valid (film the whole stage).

    Falls back to group shots if the pool would be empty.
    """
    if not inst_onsets:
        return pool
    win = tpb * 2  # 2-beat window (same as _guard_directed for featured instruments)
    result = []
    for cut in pool:
        insts = _COOP_INSTR.get(cut, set())
        if not insts:               # group shot — sempre válido
            result.append(cut)
        elif any(_playing_near(inst_onsets.get(i), tick, win) for i in insts):
            result.append(cut)
    # Fallback: group shots (nunca vazio — _GROUP_FRAMINGS tem 3)
    return result or [c for c in _GROUP_FRAMINGS] or ["All_Near"]


def _safe_framing(framing: list[str], bad: set[str]) -> list[str]:
    """Remove from `framing` the cuts of absent instruments. If nothing is left (the
    pool only had cuts of absent ones — e.g. bridge with only bass/keys), does NOT
    restore the original pool (that would reintroduce the absent ones): falls back to
    full-band group shots."""
    f = [c for c in framing if c not in bad]
    return f or [c for c in _GROUP_FRAMINGS if c not in bad] or ["All_Near"]


def _absent_framings(inst_onsets: dict[str, list[int]] | None) -> set[str]:
    """Set of framings to EXCLUDE because they film an absent instrument.
    An instrument counts as present if it has onsets in `inst_onsets` (real charts for
    the band; vocal also counts with an audio/lyrics signal). With no inst_onsets, it
    filters nothing (legacy behavior)."""
    if not inst_onsets:
        return set()
    bad: set[str] = set()
    if not inst_onsets.get("keys"):   bad |= _KEYS_FRAMINGS
    if not inst_onsets.get("bass"):   bad |= _BASS_FRAMINGS
    if not inst_onsets.get("guitar"): bad |= _GUITAR_FRAMINGS
    if not inst_onsets.get("drums"):  bad |= _DRUMS_FRAMINGS
    if not inst_onsets.get("vocal"):  bad |= _VOCAL_FRAMINGS
    return bad


# Framing → instrument(s) it features (used to bias the bed toward whoever LEADS a
# section). Each maps to the same per-instrument framing sets used by the absent-filter.
_INST_FRAMINGS = {"keys": _KEYS_FRAMINGS, "bass": _BASS_FRAMINGS,
                  "guitar": _GUITAR_FRAMINGS, "drums": _DRUMS_FRAMINGS,
                  "vocal": _VOCAL_FRAMINGS}


def _featured_instrument(inst_onsets: dict[str, list[int]] | None,
                         start: int, end: int,
                         totals: dict[str, int]) -> str | None:
    """Who CARRIES this section, song-relative. We rank by onsets-in-window / total
    onsets of that instrument across the whole song — the instrument that 'steps up'
    here relative to its own baseline. Using a raw count would always pick drums (they
    play continuously); the relative measure surfaces the guitar in a riff, the vocal
    in a chorus, etc. Returns None when nothing clearly leads (keeps the bed neutral)."""
    if not inst_onsets:
        return None
    import bisect
    best, best_score = None, 0.0
    for inst in ("guitar", "bass", "drums", "keys", "vocal"):
        ons = inst_onsets.get(inst)
        tot = totals.get(inst, 0)
        if not ons or tot < 4:
            continue
        cnt = bisect.bisect_left(ons, end) - bisect.bisect_left(ons, start)
        if cnt < 2:
            continue
        score = cnt / tot
        if score > best_score:
            best, best_score = inst, score
    return best


def _bias_pool(pool: list[str], inst: str | None) -> list[str]:
    """Reorder a framing pool so the cuts that feature `inst` come FIRST (the bed leans
    toward the featured instrument, especially the section's opening cut). Anti-recency
    + no-immediate-repeat still rotate through the rest, so variety and the calibrated
    instrument quota are largely preserved — this is a gentle lean, not a lock."""
    if not inst:
        return pool
    fr = _INST_FRAMINGS.get(inst, set())
    lead = [c for c in pool if c in fr]
    return lead + [c for c in pool if c not in fr] if lead else pool


# NOTE: the old section/energy directed POOLS (`_SECTION_DIRECTED`), the instrumental
# substitution map (`_NO_VOCAL_SUB`) and the full-band injector cycle (`_ALLBAND_*`)
# were removed with the event-engine rewrite — directed cuts now come from detected
# musical events (see cut_events.py / docs/CUTS_ALGORITHM_STUDY.md), not pool rotation.

# Solo pools per instrument — featuring the soloist via closeups/coop framing with
# 1 directed CLS as an accent (solos justify more focus than the rest of the song).
SOLO_CAMERA = {
    "guitar": ["G_Near", "G_Hand", "G_Head", "DG_Near", "G_Behind", "D_Gtr_CLS"],
    "bass":   ["B_Near", "B_Hand", "B_Head", "BD_Near", "B_Behind", "D_Bass_CLS"],
    "drums":  ["D_Near", "D_Hand", "D_Head", "All_Near", "D_Behind", "D_Drums"],
    "keys":   ["K_Near", "K_Hand", "K_Head", "KV_Near", "K_Behind", "D_Keys"],
    "vocal":  ["V_Near", "V_Closeup", "GV_Near", "DV_Near", "V_Behind", "D_Vox_CLS"],
}

def _solo_instrument(name: str) -> str:
    n = name.lower()
    for key in ("bass", "drum", "keys", "vocal", "guitar"):
        if key in n:
            return {"drum": "drums", "vocal": "vocal"}.get(key, key)
    return "guitar"


def _cut_event(tick: int, cut: str) -> AbsEvent | None:
    """Map a cut name (standard or directed) to its text event."""
    if cut in CAMERA_CUTS:
        return _txt(tick, f"[{CAMERA_CUTS[cut]}]")
    if cut in DIRECTED_CUTS:
        return _txt(tick, f"[{DIRECTED_CUTS[cut]}]")
    return None


def _nearest(t: int, xs: list[int], window: int, floor: int) -> int | None:
    """Element of `xs` nearest to t within ±window and ≥ floor (or None)."""
    if not xs:
        return None
    import bisect
    i = bisect.bisect_left(xs, t)
    best, bestd = None, window + 1
    for ci in (i - 1, i):
        if 0 <= ci < len(xs):
            d = abs(xs[ci] - t)
            if d <= window and d < bestd and xs[ci] >= floor:
                best, bestd = xs[ci], d
    return best


def _snap_to_music(t: int, accents: list[int], tpb: int, floor: int) -> int:
    """Snap the cut to MUSIC time: first to a structural accent (emphasis) within
    ±1 beat; otherwise to the nearest BEAT (±1/2 beat) — in MIDI ticks a beat is always
    `tpb`. Never moves back before `floor` (monotonicity)."""
    a = _nearest(t, accents, window=tpb, floor=floor)
    if a is not None:
        return a
    b = round(t / tpb) * tpb         # nearest beat
    if abs(b - t) <= tpb // 2 and b >= floor:
        return b
    return t


def _section_onset_gap(onsets: list[int], start: int, end: int) -> int | None:
    """Median gap between onsets within the section (in ticks), or None if < 2 notes.
    Measures how fast the MUSIC moves in the section."""
    import bisect
    lo = bisect.bisect_left(onsets, start)
    hi = bisect.bisect_left(onsets, end)
    seg = onsets[lo:hi]
    if len(seg) < 2:
        return None
    gaps = sorted(seg[i + 1] - seg[i] for i in range(len(seg) - 1))
    return gaps[len(gaps) // 2]


# Directed cut → instrument it features (single-character performance cuts).
# Guard: only fires if that instrument is playing near the tick; otherwise uses
# the _NP variant (same character, idle action) — avoids "air-guitar" with no music.
_DIRECTED_INSTR = {
    "D_Gtr": "guitar", "D_Gtr_CLS": "guitar", "D_Gtr_Cam_PR": "guitar",
    "D_Gtr_Cam_PT": "guitar",
    "D_Bass": "bass", "D_Bass_CLS": "bass", "D_Bass_Cam": "bass",
    "D_Drums": "drums", "D_Drums_Point": "drums", "D_Drums_KD": "drums",
    "D_Drums_LT": "drums", "D_Drums_CLS": "drums",
    "D_Keys": "keys", "D_Keys_Cam": "keys",
    "D_Vocals": "vocal", "D_Vox_CLS": "vocal", "D_Vox_Cam_PR": "vocal",
    "D_Vox_Cam_PT": "vocal",
}
_DIRECTED_NP = {
    "D_Gtr": "D_Gtr_NP", "D_Bass": "D_Bass_NP", "D_Drums": "D_Drums_NP",
    "D_Keys": "D_Keys_NP", "D_Vocals": "D_Vox_NP",
}
# GESTURE/showmanship cuts: the animation shows the character NOT playing (drummer
# turning the sticks to the camera, guitarist/bassist working the crowd, _np variants
# idle). They only make sense in a downtime of that instrument — if it HAS notes nearby,
# the gesture contradicts the charted animation. Map: cut → instrument that must be idle.
_DIRECTED_NOTPLAYING = {
    "D_Crowd_Gtr": "guitar", "D_Crowd_Bass": "bass", "D_Drums_Point": "drums",
    "D_Gtr_NP": "guitar", "D_Bass_NP": "bass", "D_Drums_NP": "drums",
    "D_Keys_NP": "keys", "D_Vox_NP": "vocal",
}
# Cuts that involve the crowd/sing-along — only make sense with vocals/lyrics.
_DIRECTED_SING = {"D_Crowd", "D_Crowd_Gtr", "D_Crowd_Bass", "D_All_Yeah",
                  "D_Stagedive", "D_Crowdsurf"}
# DUOS (book p.349): two members interacting. Only make sense if BOTH play nearby
# (e.g. duo_kb with no keys doesn't exist). Suffixes: _g=gtr, _b=bass, _k=keys, _v=vox.
_DIRECTED_DUO = {
    "D_Duo_GB": ("guitar", "bass"), "D_Duo_KB": ("keys", "bass"),
    "D_Duo_KG": ("keys", "guitar"), "D_Duo_KV": ("keys", "vocal"),
    "D_Duo_Gtr": ("guitar", "vocal"), "D_Duo_Bass": ("bass", "vocal"),
    "D_Duo_Drums": ("drums", "vocal"),
}


def _playing_near(onsets: list[int] | None, tick: int, win: int) -> bool:
    """True if the instrument has any onset within ±win ticks of `tick`."""
    if not onsets:
        return False
    import bisect
    i = bisect.bisect_left(onsets, tick)
    for ci in (i - 1, i):
        if 0 <= ci < len(onsets) and abs(onsets[ci] - tick) <= win:
            return True
    return False


def _guard_directed(cut: str, tick: int, tpb: int,
                    inst_onsets: dict[str, list[int]] | None) -> str | None:
    """Adjust a directed cut to the musical context: a featured instrument that isn't
    playing → _NP variant; crowd/sing-along without vocals → None (falls to framing).
    Non-directed cuts and generic dramatic ones pass through intact."""
    if inst_onsets is None:
        return cut
    inst = _DIRECTED_INSTR.get(cut)
    if inst is not None and not _playing_near(inst_onsets.get(inst), tick, tpb * 2):
        # ABSENT instrument (no chart at all) → there's no character to film, not even
        # idle: falls to framing (None). Only in a momentary PAUSE (it has notes
        # somewhere, but not here) does the _NP variant make sense — the character
        # exists, just stopped. (Vocals are the exception allowed elsewhere.)
        if not inst_onsets.get(inst):
            return None
        # _NP shows the character idle; cuts with no _NP variant (closeups/cam) fall
        # to framing (None) instead of filming someone who isn't playing.
        return _DIRECTED_NP.get(cut)
    gesture = _DIRECTED_NOTPLAYING.get(cut)
    if gesture is not None and _playing_near(inst_onsets.get(gesture), tick, tpb):
        # Gesture (crowd_g/crowd_b/drums_pnt/_np) with the instrument PLAYING nearby →
        # contradicts the charted animation; falls to framing (None).
        return None
    duo = _DIRECTED_DUO.get(cut)
    if duo is not None:
        # A duo requires BOTH members playing nearby; otherwise it falls to framing (None).
        if not all(_playing_near(inst_onsets.get(d), tick, tpb * 2) for d in duo):
            return None
    if cut in _DIRECTED_SING:
        # Crowd/sing-along requires REAL vocals (chart/lyrics) — the audio proxy
        # ("vocal" without "_vocal_real") has no words for the crowd to sing.
        real = inst_onsets.get("_vocal_real")
        if real is None or not _playing_near(real, tick, tpb * 4):
            return None                          # no real vocals → no crowd/sing
    return cut


def _level_filter(candidates: list[str], min_level: int) -> list[str]:
    """Filter cuts by Onyx level rule: level < min_level OR level == 0 if min == 0.
    
    Falls back to all candidates if filter would empty the pool.
    """
    if not candidates:
        return candidates
    if min_level == 0:
        filtered = [c for c in candidates if _CAMERA_LEVELS.get(c, 999) == 0]
    else:
        filtered = [c for c in candidates if _CAMERA_LEVELS.get(c, 999) < min_level]
    return filtered if filtered else candidates


def build_camera(sections: list[Section], tempo_map: list, time_sig_map: list,
                 tpb: int, bre_spans: list[tuple[int, int]] | None = None,
                 pace_scale: float = 1.0,
                 accents: list[int] | None = None,
                 onsets: list[int] | None = None,
                 inst_onsets: dict[str, list[int]] | None = None,
                 audio_onsets: list[int] | None = None,
                 energy_env: list[tuple[int, str]] | None = None,
                 band_activity: dict[str, list[int]] | None = None,
                 design: "VenueDesign | None" = None) -> list[AbsEvent]:
    """Places camera cuts at the rate of SECTION_PACE_S (× the theme's pace_scale),
    cycling the section pool without repeating the previous cut. Injects D_BRE on BREs.
    If `accents` is given, snaps each cut to the nearest musical accent (±1 beat) —
    cuts on music time instead of clock time.
    If `onsets` is given, does NOT cut faster than the music moves: in sparse
    sections (few notes) the pace is stretched to the real note spacing (avoids
    over-cutting in slow ballads/post-metal)."""
    from collections import deque
    from .cut_events import detect_events
    out: list[AbsEvent] = []
    rng = design.rng if design is not None else None
    accents = sorted(accents) if accents else []
    # Merge MIDI accents + audio flux accents for richer snap targets
    if audio_onsets:
        audio_acc = sorted(audio_onsets)
        # Merge sorted lists (O(n+m))
        merged = []
        i = j = 0
        while i < len(accents) and j < len(audio_acc):
            if accents[i] < audio_acc[j]:
                merged.append(accents[i])
                i += 1
            elif audio_acc[j] < accents[i]:
                merged.append(audio_acc[j])
                j += 1
            else:
                merged.append(accents[i])
                i += 1
                j += 1
        merged.extend(accents[i:])
        merged.extend(audio_acc[j:])
        accents = merged
    onsets = sorted(onsets) if onsets else []
    min_gap = tpb // 2               # never two cuts less than 1/8 apart
    max_floor = tpb * 6              # but never slower than 1 cut / 6 beats
    bad_framings = _absent_framings(inst_onsets)
    # Song-wide onset totals per instrument (for the featured-instrument bias, Phase 4).
    inst_totals = ({k: len(v) for k, v in inst_onsets.items() if v}
                   if inst_onsets else {})
    # Full-band cuts are spaced ≥ this (book: "use sparingly"; ~4/song like the officials).
    fullband_gap = tpb * 32

    def _framing_only(pool: list[str]) -> list[str]:
        """Keep only coop_* framings (the directed are now injected by the event layer)."""
        f = [c for c in pool if c in CAMERA_CUTS]
        return f or ["All_Near"]

    # ── PASS 1: framing bed (coop_* only) → list of (tick, cut) slots ─────────────
    # The bed is the paced, snapped, anti-recency framing camera. Directed cuts are
    # NOT chosen here anymore; they come from PASS 2 (musical events) and replace the
    # nearby filler. This separates "who/where to point the camera" (bed) from "react
    # to a musical moment" (events) — see docs/CUTS_ALGORITHM_STUDY.md.
    slots: list[tuple[int, str]] = []
    last_cut: str | None = None
    last_tick = -10 ** 9
    recent_coop: deque = deque(maxlen=6)
    min_framing_level: int = 999  # Onyx level tracker: starts "infinite"

    # Lead-in: cut from tick 0 to the 1st section (officials never freeze on the intro).
    if sections and sections[0].start > tpb:
        s0 = sections[0]
        framing0 = _framing_only(SECTION_CAMERA.get(s0.kind, SECTION_CAMERA["default"]))
        if bad_framings:
            framing0 = _safe_framing(framing0, bad_framings)
        pace_ms0 = SECTION_PACE_S.get(s0.kind, 3.0) * 1000.0 * pace_scale
        t = idx = 0
        while t < s0.start:
            placed = _snap_to_music(t, accents, tpb, floor=last_tick + min_gap)
            if placed <= last_tick:
                placed = max(t, last_tick + min_gap)
            if placed >= s0.start:
                break
            level_pool = _level_filter(framing0, min_framing_level)
            level_pool = _playing_framing(level_pool, placed, inst_onsets, tpb)
            cut = _markov_choice(last_cut, level_pool, _CAMERA_MARKOV, level_pool[0], rng=rng)
            if cut == last_cut:
                idx += 1
                cut = _markov_choice(last_cut, level_pool, _CAMERA_MARKOV, level_pool[idx % len(level_pool)], rng=rng)
            slots.append((placed, cut))
            last_cut, last_tick = cut, placed
            cut_level = _CAMERA_LEVELS.get(cut, 999)
            if cut_level < min_framing_level:
                min_framing_level = cut_level
            idx += 1
            t += max(ms_to_ticks(pace_ms0, t, tempo_map, tpb), min_gap)

    for si, s in enumerate(sections):
        energy = _camera_energy(s)
        if s.kind == "solo":
            pool = _framing_only(SOLO_CAMERA[_solo_instrument(s.name)])
        else:
            pool = _framing_only(SECTION_CAMERA.get(s.kind, SECTION_CAMERA["default"]))
        if bad_framings:                             # don't film absent instruments
            pool = _safe_framing(pool, bad_framings)
        if s.kind != "solo":                         # solos already focus via SOLO_CAMERA
            feat = (design.group_at(si).featured_inst if design is not None
                    else _featured_instrument(inst_onsets, s.start, s.end, inst_totals))
            pool = _bias_pool(pool, feat)
        pace_ms = SECTION_PACE_S.get(s.kind, 3.0) * 1000.0 * pace_scale
        # Audio nudge: a quieter-than-structural section cuts slower.
        _E = {"calm": 0, "mid": 1, "high": 2}
        delta = _E[energy] - _E[SECTION_ENERGY.get(s.kind, "calm")]
        pace_ms *= 1.18 ** (-delta)
        # Density floor: don't cut faster than the note spacing (slow ballads/post-metal).
        density_floor = 0
        if onsets:
            g = _section_onset_gap(onsets, s.start, s.end)
            if g is not None:
                density_floor = min(g, max_floor)
        idx = 0
        t = s.start
        while t < s.end:
            step = max(ms_to_ticks(pace_ms, t, tempo_map, tpb), density_floor)
            placed = _snap_to_music(t, accents, tpb, floor=last_tick + min_gap)
            if placed <= last_tick:
                placed = max(t, last_tick + min_gap)
            # Onyx level filter: restrict pool to cuts that respect level progression
            level_pool = _level_filter(pool, min_framing_level)
            if not level_pool:
                level_pool = pool
            # Playing-state filter: remove cuts que filmam instrumentos em pausa
            level_pool = _playing_framing(level_pool, placed, inst_onsets, tpb)
            cut = _markov_choice(last_cut, level_pool, _CAMERA_MARKOV, level_pool[0], rng=rng)
            if cut == last_cut:
                idx += 1
                cut = _markov_choice(last_cut, level_pool, _CAMERA_MARKOV, level_pool[idx % len(level_pool)], rng=rng)
            # Anti-recency
            for _ in range(len(level_pool)):
                if cut not in recent_coop and cut != last_cut:
                    break
                idx += 1
                cut = _markov_choice(last_cut, level_pool, _CAMERA_MARKOV, level_pool[idx % len(level_pool)], rng=rng)
            if placed < s.end:
                slots.append((placed, cut))
                recent_coop.append(cut)
                cut_level = _CAMERA_LEVELS.get(cut, 999)
                if cut_level < min_framing_level:
                    min_framing_level = cut_level
                last_cut, last_tick = cut, placed
            idx += 1
            t += step

    # Snap slot to shared anchor (sync with lighting/pp)
    if design is not None:
        for si, s in enumerate(sections):
            anchor = design.anchors[si]
            if not (s.start <= anchor < s.end):
                continue
            # Find nearest slot to anchor
            nearest_idx = None
            nearest_dist = min_gap + 1
            for idx, (tk, cut) in enumerate(slots):
                if s.start <= tk < s.end:
                    d = abs(tk - anchor)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_idx = idx
            if nearest_idx is not None and nearest_dist >= min_gap // 2:
                # Move nearest slot to anchor tick
                slots[nearest_idx] = (anchor, slots[nearest_idx][1])

    # ── PASS 2: directed cuts from MUSICAL EVENTS (precise hit ticks) ─────────────
    # Each event owns candidate cuts (most→least specific); the guard adapts them to
    # context (_NP if idle, None if it makes no sense). Full-band cuts are throttled to
    # `fullband_gap`; one stage dive per song; anti-recency on the rest.
    events = detect_events(sections, inst_onsets, accents, bre_spans, time_sig_map, tpb,
                           audio_onsets=audio_onsets, energy_env=energy_env,
                           band_activity=band_activity, design=design)
    events.sort(key=lambda e: (e.tick, -e.priority))
    accepted: list[tuple[int, str]] = []
    recent_dir: deque = deque(maxlen=4)
    last_fullband = -10 ** 9
    did_stagedive = False
    _rand = rng if rng is not None else random
    for e in events:
        chosen = None
        # Level filter candidates
        candidates = _level_filter(e.cuts, 0)  # min_level=0: only level-0 directed cuts
        if not candidates:
            candidates = e.cuts
        for cand in candidates:
            g = _guard_directed(cand, e.tick, tpb, inst_onsets)
            if g is None:
                continue
            # 42% chance of self-repeat (matching official data)
            if accepted and g == accepted[-1][1] and _rand.random() < 0.42:
                chosen = g
                break
            if g in recent_dir and chosen is None:
                chosen = g           # fallback
                continue
            if g not in recent_dir:
                chosen = g
                break
        if chosen is None:
            continue
        is_fullband = chosen in ("D_All", "D_All_Cam", "D_All_LT", "D_All_Yeah")
        if is_fullband and e.tick - last_fullband < fullband_gap:
            continue                                 # space out full-band (sparingly)
        if chosen in ("D_Stagedive", "D_Crowdsurf"):
            if did_stagedive:
                continue                             # at most one per song
            did_stagedive = True
        accepted.append((e.tick, chosen))
        recent_dir.append(chosen)
        if is_fullband:
            last_fullband = e.tick

    # ── Add companion shots at the same tick as primary directed cuts ──────
    # Official venues frequently fire MULTIPLE directed cuts at the same tick
    # (e.g. directed_drums_lt + directed_guitar_cls). Add companions using
    # co-occurrence pairs learned from 100 official songs.
    from .cut_events import add_companion_shots
    companions = add_companion_shots(accepted)

    # ── MERGE: directed wins near a framing slot; drop the filler within min_gap ──
    import bisect
    dir_ticks = sorted(t for t, _ in accepted)

    def _near_directed(tk: int) -> bool:
        i = bisect.bisect_left(dir_ticks, tk)
        for ci in (i - 1, i):
            if 0 <= ci < len(dir_ticks) and abs(dir_ticks[ci] - tk) < min_gap:
                return True
        return False

    merged = [(tk, c) for tk, c in slots if not _near_directed(tk)]
    # Ghost guard: slots do PASS 1 que filmam instrumento em pausa (sem onset
    # nos ±2 beats) são substituídos por group shots, evitando o "parado a tocar".
    cleaned: list[tuple[int, str]] = []
    for tk, cut in merged:
        if cut in CAMERA_CUTS and inst_onsets:
            insts = _COOP_INSTR.get(cut, set())
            if insts and not any(_playing_near(inst_onsets.get(i), tk, tpb * 2)
                                for i in insts):
                # Instrumento em pausa — substituir por group shot
                cleaned.append((tk, "All_Near"))
                continue
        cleaned.append((tk, cut))
    merged = cleaned
    merged += list(accepted)
    merged += companions
    merged.sort(key=lambda x: x[0])

    prev_tick = -10 ** 9
    prev_cut: str | None = None
    for tk, cut in merged:
        if cut == prev_cut:
            continue
        if tk - prev_tick < min_gap:
            if tk != prev_tick:          # allow same-tick different-category companions
                continue
        ev = _cut_event(tk, cut)
        if ev is not None:
            out.append(ev)
            prev_tick, prev_cut = tk, cut
    return out


# ── Assembly ──────────────────────────────────────────────────────────────────

def find_bre_spans(events: list[AbsEvent]) -> list[tuple[int, int]]:
    """Big Rock Ending spans (global marker 120) on an instrument track."""
    spans: list[tuple[int, int]] = []
    open_t: int | None = None
    for ev in sorted(events, key=lambda e: e.abs_tick):
        m = ev.msg
        if m.type == "note_on" and getattr(m, "velocity", 0) > 0 and m.note == 120:
            open_t = ev.abs_tick
        elif (m.type == "note_off" or (m.type == "note_on" and m.velocity == 0)) \
                and getattr(m, "note", None) == 120 and open_t is not None:
            spans.append((open_t, ev.abs_tick))
            open_t = None
    return spans


def load_genre(folder: str) -> str | None:
    """Read the `genre` field of the song.ini in the song's folder (if it exists)."""
    import os
    path = os.path.join(folder, "song.ini")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                if "=" in line:
                    k, _, v = line.partition("=")
                    if k.strip().lower() == "genre":
                        return v.strip()
    except OSError:
        return None
    return None


def resolve_sections(events_track: list[AbsEvent], song_end: int,
                     onsets: list[int], time_sig_map: list, tpb: int) -> list[Section]:
    """Resolve the song's sections: reads markers; if there are none, synthesizes them
    by density; if there are some but with unknown names, refines them by density.
    Does NOT change names — only the internal classification (kind)."""
    sections = parse_sections(events_track, song_end)
    if not sections:
        return synthesize_sections(song_end, onsets, time_sig_map, tpb)
    return refine_sections(sections, onsets, tpb)


# ── Character animations (VENUE_SPEC §11) ─────────────────────────────────────
#
# Per-PART-track mood markers, derived from the sections + each instrument's
# presence. Not on the VENUE track — on each instrument track.

_INTENSE_THEMES = {"metal", "punk"}

# How many measures of rest WITHIN a section before the instrument visibly idles
# (puts the instrument down). Per-instrument thresholds derived from the 20 official
# venues: drums/bass idle sooner (shorter rests visible), guitar holds longer
# (flicker-prone on sparse riffs), vocal is the most rest-heavy (idle 21%).
_IDLE_DOWNTIME = {"drums": 2, "bass": 2, "guitar": 6, "keys": 3, "vocal": 1}


def phrase_end_ticks(track) -> list[int]:
    """Ticks of the note_offs of the vocal phrase markers (pitch 105/106) — end of each
    vocal phrase. Used so the vocalist only lowers the mic at the end of the phrase."""
    t = 0
    open_at: dict[int, int] = {}
    ends: list[int] = []
    for m in track:
        t += m.time
        note = getattr(m, "note", None)
        if note not in (105, 106):
            continue
        if m.type == "note_on" and m.velocity > 0:
            open_at[note] = t
        elif m.type == "note_off" or (m.type == "note_on" and m.velocity == 0):
            if note in open_at:
                ends.append(t)
                del open_at[note]
    return sorted(set(ends))


def _part_instrument(track_name: str) -> str | None:
    """Map a PART track's name to the instrument it animates (or None)."""
    n = track_name.upper()
    if "DRUM" in n:
        return "drums"
    if "BASS" in n:
        return "bass"
    if "KEY" in n:
        return "keys"
    if "VOCAL" in n:
        return "vocal"
    if "GUITAR" in n or "RHYTHM" in n or "GEMS" in n:
        return "guitar"
    return None


# Main PART track names per instrument — variant tracks (REAL_*, *_ANIM_*,
# *_E/_M/_H/_X) don't carry mood markers in the official venues.
_MAIN_PART = {
    "PART GUITAR", "PART BASS", "PART DRUMS", "PART KEYS", "PART VOCALS",
}


def _is_main_part_track(track_name: str) -> bool:
    """True if this is the primary playable PART track (not a variant)."""
    return track_name.strip() in _MAIN_PART


def _section_at(sections: list[Section], tick: int) -> Section | None:
    for s in sections:
        if s.start <= tick < s.end:
            return s
    return sections[-1] if sections else None


def _energy_tier_at(s: Section, tick: int) -> str:
    """Energy tier of the sub-span COVERING `tick` (not the section peak). A calm pocket
    inside an otherwise-loud section reads calm; falls back to the section energy when
    there are no audio spans (MIDI-only path)."""
    if s.energy_spans:
        for a, b, tier in s.energy_spans:
            if a <= tick < b:
                return tier
        return (s.energy_spans[0][2] if tick < s.energy_spans[0][0]
                else s.energy_spans[-1][2])
    return section_energy(s)


def _idle_marker_at(sections: list[Section], tick: int,
                    instrument: str = "guitar") -> str:
    """The right 'not playing' marker for the EXACT tick of a rest. Uses the
    SECTION's overall energy (not the sub-span at the exact tick) so idle_intense
    fires whenever the section is loud, even if the idle tick lands in a calm
    pocket within the section. idle_intense is only for keys/vocal sitting out a
    loud section — the official venues use it 10%/22% for keys/vocal but <2% for
    guitar/bass/drums, so we restrict it to those instruments. idle_realtime is
    reserved for SONG BOUNDARIES only (emitted separately)."""
    s = _section_at(sections, tick)
    if s is None:
        return "[idle]"
    if section_energy(s) == "high" and instrument in ("keys", "vocal"):
        return "[idle_intense]"
    return "[idle]"



_MOOD_LADDER = ["mellow", "play", "intense"]
_ENERGY_LEVEL = {"calm": 0, "mid": 1, "high": 2}
_LEVEL_ENERGY = {0: "calm", 1: "mid", 2: "high"}


def _camera_energy(s: Section) -> str:
    """Energy tier the CAMERA uses to pick directed cuts (jumps/kicks/crowd vs
    closeups). Uses the STRONGEST post-gate sub-span tier instead of the section
    MEAN, so it inherits the heaviness gate: a loud-but-not-heavy chorus whose
    'high' spans were demoted to 'mid' (sung chorus, not a breakdown) no longer
    triggers the energetic directed pool / dircut_at_start. Falls back to the
    section mean when there are no audio spans (MIDI-only path unchanged)."""
    if s.energy_spans:
        top = max(_ENERGY_LEVEL.get(t, 0) for _, _, t in s.energy_spans)
        return _LEVEL_ENERGY[top]
    return section_energy(s)
# Official venues: non-unison comes from (1) idle instruments, (2) sub-section energy
# variation via energy_spans, (3) different instruments in different sections — NOT from
# a modulo-based stagger.  The old stagger artificially inflated mellow because calm
# sections produce 100% mellow and mid sections produce 50% mellow.  With no stagger,
# mood is purely energy-driven: calm→mellow, mid→play, high→intense.


def _anim_state(s: Section, instrument: str,
                sec_idx: int = 0, sec_onset_density: float = 0,
                song_mean_density: float = 1) -> str:
    """Decide an instrument's mood marker in a section. Mood follows the section
    energy (calm→mellow / mid→play / high→intense), with onset density refinement:
    a calm section where the instrument plays densely bumps up to play (active, not
    resting), and a high section where the instrument plays sparsely bumps down to
    play (present but not driving)."""
    if s.kind == "solo" and _solo_instrument(s.name) == instrument:
        return "[play_solo]"
    return _mood_for_level(_ENERGY_LEVEL[section_energy(s)], instrument, sec_idx,
                           sec_onset_density, song_mean_density)


def _mood_for_level(level: int, instrument: str, sec_idx: int,
                     sec_density: float = 0, song_mean_density: float = 1) -> str:
    """Map an energy level (0/1/2) to a mood marker, with onset-density refinement.
    Density ratio = section density / song mean.  Dense calm → bump up to play;
    sparse high → bump down to play.  Non-unison comes from idle instruments and
    density variation, not a modulo stagger."""
    if sec_density > 0 and song_mean_density > 0:
        ratio = sec_density / max(song_mean_density, 0.01)
        if level == 0 and ratio > 1.2:
            level = 1
        elif level == 1 and ratio < 0.6:
            level = 0
        elif level == 2 and ratio < 0.6:
            level = 1
    return f"[{_MOOD_LADDER[max(0, min(level, 2))]}]"


def build_animations(part_onsets: list[int], sections: list[Section],
                     tpb: int, time_sig_map: list,
                     instrument: str) -> list[AbsEvent]:
    """Generate mood markers for ONE instrument. Offset of ±1/8 on transitions;
    [idle_realtime] at song boundaries; [idle] on long downtime after the last note."""
    onsets = sorted(part_onsets)
    if not onsets:
        return [_txt(0, "[idle_realtime]")]
    if not sections:
        return [_txt(0, "[idle_realtime]")]
    floor = onsets[0]
    # quarter note (1 beat) offset for anchor/idle placement
    quarter = max(1, tpb // 2)
    last_onset = onsets[-1]

    # Compute onset density (onsets per beat) for mood refinement.
    # Song-wide mean density gives a baseline; each section's density vs the mean
    # drives the calm→play / high→play bump.  MIDI-derived, song-relative.
    total_beats = max(1.0, (last_onset - floor) / tpb)
    song_mean_density = len(onsets) / total_beats  # onsets/beat

    def _sec_density(sec_start: int, sec_end: int) -> float:
        n = _count_onsets(onsets, sec_start, sec_end)
        beats = max(1.0, (sec_end - sec_start) / tpb)
        return n / beats

    timeline: list[tuple[int, str]] = []
    import bisect

    def _first_onset_in(a: int, b: int) -> int | None:
        i = bisect.bisect_left(onsets, a)
        return onsets[i] if i < len(onsets) and onsets[i] < b else None

    def _anchor(on: int, first: bool, cur_level: int) -> int:
        if first:
            return max(s.start, floor)
        cand = on - quarter
        if cand < floor:
            return floor
        sec = _section_at(sections, cand)
        if sec is not None and _ENERGY_LEVEL[_energy_tier_at(sec, cand)] != cur_level:
            return on
        return cand

    # Official pattern: [idle] is placed ~1 beat after the last note, only when
    # the gap to the NEXT note is long. We check: for each pair of consecutive
    # onsets, if the gap >= _IDLE_DOWNTIME_MEASURES measures, emit [idle] right
    # after the first onset (the "last note before the rest").
    idle_ranges: list[tuple[int, int]] = []
    idle_ticks: set[int] = set()
    dt_measures = _IDLE_DOWNTIME.get(instrument, 4)
    for a, b in zip(onsets, onsets[1:]):
        mt = measure_ticks_at(a, time_sig_map, tpb)
        if b - a >= dt_measures * mt:
            idle_ranges.append((a, b))
            idle_ticks.add(a + quarter)

    def _in_idle(tick: int) -> bool:
        return any(a < tick < b for a, b in idle_ranges)

    for i, s in enumerate(sections):
        playing = _count_onsets(onsets, s.start, s.end) > 0
        solo = s.kind == "solo" and _solo_instrument(s.name) == instrument
        if playing and s.energy_spans:
            for j, (a, b, tier) in enumerate(s.energy_spans):
                on = _first_onset_in(a, b)
                if on is None:
                    continue
                if solo:
                    mood = "[play_solo]"
                else:
                    cur_level = _ENERGY_LEVEL[tier]
                    mood = _mood_for_level(cur_level, instrument, i,
                                           _sec_density(a, b), song_mean_density)
                tick = _anchor(on, i == 0 and j == 0,
                               _ENERGY_LEVEL[tier])
                if _in_idle(tick):
                    continue
                timeline.append((tick, mood))
        elif playing:
            on = _first_onset_in(s.start, s.end)
            cur_level = _ENERGY_LEVEL[section_energy(s)]
            if i == 0:
                tick = max(s.start, floor)
            elif on:
                tick = _anchor(on, False, cur_level)
            else:
                tick = max(s.start - quarter, floor)
            if not _in_idle(tick):
                timeline.append((tick, _anim_state(s, instrument, i,
                                  _sec_density(s.start, s.end), song_mean_density)))

    # Emit idle markers at the computed ticks
    for it in sorted(idle_ticks):
        timeline.append((it, _idle_marker_at(sections, it, instrument)))

    # Boundary idle markers: [idle_realtime] at song start and end.
    # End boundary: only when the gap from last note to end is large (≥ 2 measures),
    # otherwise the section's own [idle] covers it.  This prevents over-emitting
    # idle_realtime (2 beats was too short — almost every song has >2 beats after
    # the last vocal note).
    if sections:
        first_sec = sections[0]
        if first_sec.kind in ("intro", "default") or floor > first_sec.start + quarter:
            timeline.insert(0, (0, "[idle_realtime]"))
        last_sec = sections[-1]
        song_end = last_sec.end
        end_mt = measure_ticks_at(last_onset, time_sig_map, tpb)
        end_thresh = max(2 * quarter, 4 * end_mt) if end_mt else 2 * quarter
        if last_onset + end_thresh < song_end:
            timeline.append((last_onset + end_thresh, "[idle_realtime]"))

    timeline.sort(key=lambda x: x[0])
    out: list[AbsEvent] = []
    last: str | None = None
    for tick, state in timeline:
        if state != last:
            out.append(_txt(tick, state))
            last = state
    return out


def _bass_strummap(onsets: list[int], tpb: int) -> str:
    """Bass StrumMap heuristic from the chart's rhythm (no audio):
    many fast notes (intervals < 1/8 in the majority) → pick; otherwise fingers.
    Slap is not reliably detectable from MIDI → we never assume it."""
    if len(onsets) < 8:
        return "StrumMap_Default"
    gaps = [b - a for a, b in zip(onsets, onsets[1:]) if b > a]
    if not gaps:
        return "StrumMap_Default"
    eighth = tpb // 2
    fast = sum(1 for g in gaps if g < eighth) / len(gaps)
    return "StrumMap_Pick" if fast >= 0.5 else "StrumMap_Default"


def instrument_extras(instrument: str, onsets: list[int],
                      sections: list[Section], tpb: int,
                      phrase_ends: list[int] | None = None) -> list[AbsEvent]:
    """Instrument-specific markers (VENUE_SPEC §11):
      - Bass: StrumMap (right hand, 1/song).
      - Guitar/Bass: base HandMap + HandMap_Solo on that instrument's own solos.
      - Vocals: lower the mic ([idle]) at the end of the last phrase (not 1 beat after
        the last syllable — the phrase may extend beyond that).
    Chord HandMaps are derived automatically from the chart — we don't generate 1/note."""
    onsets = sorted(onsets)
    if not onsets:
        return []
    first = onsets[0]
    out: list[AbsEvent] = []
    if instrument == "bass":
        out.append(_txt(first, f"[map {_bass_strummap(onsets, tpb)}]"))
    if instrument in ("guitar", "bass"):
        out.append(_txt(first, "[map HandMap_Default]"))
    for si, s in enumerate(sections):
            if (s.kind == "solo" and _solo_instrument(s.name) == instrument
                    and _count_onsets(onsets, s.start, s.end) > 0):
                out.append(_txt(s.start, "[map HandMap_Solo]"))
                out.append(_txt(s.end, "[map HandMap_Default]"))
    if instrument == "vocal":
        # Put the mic down at the end of the last phrase. Without phrase markers,
        # falls back to the old behavior (1 beat after the last sung syllable).
        drop = onsets[-1] + tpb
        if phrase_ends:
            tail = [p for p in phrase_ends if p >= onsets[-1]]
            if tail:
                drop = max(drop, min(tail))
        out.append(_txt(drop, "[idle]"))
    # Dedup of consecutive equal map markers (e.g. Default followed by Default).
    out.sort(key=lambda e: e.abs_tick)
    deduped: list[AbsEvent] = []
    last_map: str | None = None
    for ev in out:
        txt = ev.msg.text
        if txt.startswith("[map "):
            if txt == last_map:
                continue
            last_map = txt
        deduped.append(ev)
    return deduped


def generate_animations(part_onsets_by_track: dict[str, list[int]],
                        sections: list[Section], theme_name: str, tpb: int,
                        time_sig_map: list,
                        vocal_phrase_ends: list[int] | None = None
                        ) -> dict[str, list[AbsEvent]]:
    """Animation markers (mood + instrument extras) for each PART track."""
    res: dict[str, list[AbsEvent]] = {}
    for tname, onsets in part_onsets_by_track.items():
        inst = _part_instrument(tname)
        if inst is None:
            continue
        # Only animate the MAIN PART track per instrument. Variant tracks
        # (PART REAL_*, PART *_ANIM_*) don't carry mood markers in the
        # official venues — generating for all variants inflates counts ~5x.
        if not _is_main_part_track(tname):
            continue
        markers = build_animations(onsets, sections, tpb,
                                   time_sig_map, inst)
        pe = vocal_phrase_ends if inst == "vocal" else None
        markers += instrument_extras(inst, onsets, sections, tpb, pe)
        res[tname] = markers
    return res


# Crowd intensity follows the SAME song-relative energy map as the rest of the venue
# (calm→mellow / mid→normal / high→intense), so the audience reads coherent with the
# performers, lights and pyro. The dead-quiet bookends (intro/outro) and the multi-bar
# pauses freeze the crowd (crowd_realtime) — the band stopped, so does the room.
_CROWD_MOOD = {"calm": "crowd_mellow", "mid": "crowd_normal", "high": "crowd_intense"}


def build_crowd(sections: list[Section], tpb: int,
                pause_spans: list[tuple[int, int]] | None = None) -> list[AbsEvent]:
    """Crowd state events on the EVENTS track, driven by the section/sub-span energy.

    Mood: calm→[crowd_mellow], mid→[crowd_normal], high→[crowd_intense]. Intro/outro
    and ≥2-measure pauses → [crowd_realtime] (frozen, no animation). Clap: [crowd_clap]
    once the crowd is energetic (mid/high), [crowd_noclap] when it drops back to calm or
    freezes. Tracks the LOCAL energy (energy_spans) so the audience swells and eases
    inside a long section instead of holding one tier. Dedups consecutive states."""
    pause_spans = pause_spans or []
    # Breakpoints: each section start, each energy sub-span boundary, each pause edge.
    pts: list[int] = []
    for s in sections:
        pts.append(s.start)
        for a, _b, _t in (s.energy_spans or []):
            pts.append(a)
    for a, b in pause_spans:
        pts.append(a)
        pts.append(b)
    pts = sorted(p for p in set(pts) if p >= 0)
    out: list[AbsEvent] = []
    last_mood: str | None = None
    last_clap: str | None = None
    for p in pts:
        s = _section_at(sections, p)
        if s is None:
            continue
        if _in_span(p, pause_spans) or s.kind in ("intro", "outro"):
            mood, clap = "crowd_realtime", "crowd_noclap"
        else:
            tier = _energy_tier_at(s, p)
            mood = _CROWD_MOOD[tier]
            clap = "crowd_clap" if tier in ("mid", "high") else "crowd_noclap"
        if mood != last_mood:
            out.append(_txt(p, f"[{mood}]"))
            last_mood = mood
        if clap != last_clap:
            out.append(_txt(p, f"[{clap}]"))
            last_clap = clap
    return out


def generate_venue(events_track: list[AbsEvent], bre_spans: list[tuple[int, int]],
                   song_end: int, tempo_map: list, time_sig_map: list,
                   tpb: int, theme: str = DEFAULT_THEME,
                   accents: list[int] | None = None,
                   onsets: list[int] | None = None,
                   sections: list[Section] | None = None,
                   drum_onsets: list[int] | None = None,
                   inst_onsets: dict[str, list[int]] | None = None,
                   n_harm: int = 0,
                   fill_onsets: list[int] | None = None,
                   dbass_onsets: list[int] | None = None,
                   audio_onsets: list[int] | None = None,
                    energy_env: list[tuple[int, str]] | None = None,
                    audio_strobe_spans: list[tuple[int, int]] | None = None,
                    drop_ticks: list[int] | None = None,
                    band_activity: dict[str, list[int]] | None = None) -> list[AbsEvent]:
    """Generate all the text events of an explicit VENUE, sorted by tick.
    `theme` is the THEMES key (derived from the genre via genre_to_theme).
    `accents` (ticks of the Expert accents) syncs the cuts with the music.
    `onsets` (all of the instrument's note-ons) feeds the density-driven
    fallbacks when sections are missing or the names don't classify.
    `drum_onsets` (drum hits) syncs the light keyframes + pyro with the
    music, for a show-style lightshow."""
    th = THEMES.get(theme, THEMES[DEFAULT_THEME])
    onsets = onsets or []
    if sections is None:
        sections = resolve_sections(events_track, song_end, onsets, time_sig_map, tpb)
    # Director pass: decisões song-level (grupos de repetição, âncoras de
    # fronteira, arco, RNG seeded do conteúdo) partilhadas por todos os sistemas.
    design = plan_venue(sections, onsets, inst_onsets, time_sig_map, tpb, song_end)
    out: list[AbsEvent] = []
    pause_spans = find_pause_spans(onsets, time_sig_map, tpb)
    # Strobe fires on fills/consecutive snares (snare+toms), not on fast cymbals.
    strobe_spans = find_strobe_spans(
        fill_onsets if fill_onsets is not None else (drum_onsets or []), tpb,
        dbass_onsets=dbass_onsets)
    # Merge audio-detected blast/tremolo walls (sustained spectral flux) with the
    # MIDI-derived strobe spans — catches audio-only walls the drums don't flag.
    if audio_strobe_spans:
        strobe_spans = _merge_spans(strobe_spans + list(audio_strobe_spans), tpb // 2)
    out += build_lighting(sections, th, tpb, time_sig_map, drum_onsets,
                          pause_spans, strobe_spans, audio_onsets=audio_onsets,
                          energy_env=energy_env, drop_ticks=drop_ticks,
                          design=design)
    out += build_postproc(sections, th, tpb, time_sig_map, drum_onsets,
                          energy_env=energy_env, strobe_spans=strobe_spans,
                          design=design)
    # Audio flux accents join the band accents as pyro candidates (real hits, incl.
    # audio-only ones); build_pyro still gates density/placement by energy + cap.
    pyro_accents = (sorted(set(accents or []) | set(audio_onsets))
                    if audio_onsets else accents)
    out += build_pyro(sections, drum_onsets or [], tpb, accents=pyro_accents,
                      design=design)
    # NOTE: crowd state events ([crowd_*]) are NOT VENUE-track events — RB3/YARG read
    # them from the EVENTS track. They are emitted via build_crowd() and injected into
    # EVENTS by the processor, not appended here.
    out += build_camera(sections, tempo_map, time_sig_map, tpb, bre_spans,
                        pace_scale=th["pace"], accents=accents, onsets=onsets,
                        inst_onsets=inst_onsets, audio_onsets=audio_onsets,
                        energy_env=energy_env, band_activity=band_activity,
                        design=design)
    if inst_onsets:
        out += build_spotlights(sections, inst_onsets, tpb)
        # Sing-along only with REAL vocals (chart/lyrics), never with the audio proxy.
        out += build_singalong(sections, inst_onsets.get("_vocal_real", []),
                               n_harm, tpb)
    out.sort(key=lambda e: e.abs_tick)
    return out


# ── BEAT track (VENUE_SPEC §11, p.380) ────────────────────────────────────────

_END_RE = re.compile(r"\[\s*end\s*\]")


def find_end_tick(events_track: list[AbsEvent], song_end: int) -> int:
    """Tick of the [end] event in EVENTS (otherwise the end of the song)."""
    for ev in events_track:
        txt = getattr(ev.msg, "text", None)
        if ev.msg.type in ("text", "marker") and txt and _END_RE.search(txt):
            return ev.abs_tick
    return song_end


def _ts_at(tick: int, time_sig_map: list) -> tuple[int, int]:
    num, den = 4, 4
    for t, n, d in time_sig_map:
        if t <= tick:
            num, den = n, d
        else:
            break
    return num, den


def _beat_grid(end_tick: int, time_sig_map: list, tpb: int) -> list[tuple[int, int, int]]:
    """Canonical pulse grid: list of (tick, note, dur) — downbeat (12) on the 1st
    beat of each measure, upbeat (13) on the rest. Runs up to 1 beat before [end].
    Derived from the time_sig_map (single source for build_beat_track and extend_beat_track)."""
    out: list[tuple[int, int, int]] = []
    t = 0
    stop = end_tick - tpb            # ends one beat before [end]
    while t < stop:
        num, den = _ts_at(t, time_sig_map)
        beat_len = max(1, int(tpb * 4 / den))
        dur = max(1, beat_len // 4)
        for b in range(num):
            bt = t + b * beat_len
            if bt >= stop:
                break
            out.append((bt, 12 if b == 0 else 13, dur))
        t += num * beat_len
    return out


def build_beat_track(end_tick: int, time_sig_map: list, tpb: int) -> mido.MidiTrack:
    """BEAT track: downbeat (note 12) on the 1st beat of each measure, upbeat (note 13)
    on the rest. Ends 1 beat before [end]. Derived from the time_sig_map."""
    track = mido.MidiTrack()
    track.name = "BEAT"
    prev = 0
    for bt, note, dur in _beat_grid(end_tick, time_sig_map, tpb):
        track.append(mido.Message("note_on", note=note, velocity=100, time=bt - prev))
        track.append(mido.Message("note_off", note=note, velocity=0, time=dur))
        prev = bt + dur
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def extend_beat_track(track: mido.MidiTrack, end_tick: int,
                      time_sig_map: list, tpb: int) -> mido.MidiTrack:
    """Ensure an EXISTING BEAT track reaches ~1 beat before [end]. A BEAT that ends
    early leaves the characters FROZEN in-game (the BandDirector stops receiving the
    pulse). If the track ends more than 1 measure before the end, it is extended with
    the missing beats of the canonical grid. Otherwise the same track is returned
    untouched. Does not alter the original beats (only appends at the end)."""
    # last pulse note_on (12/13) in the existing track
    t = 0
    last = -1
    for msg in track:
        t += msg.time
        if (msg.type == "note_on" and msg.velocity > 0
                and getattr(msg, "note", -1) in (12, 13)):
            last = t
    stop = end_tick - tpb
    num, den = _ts_at(max(0, last), time_sig_map)
    measure = max(1, int(tpb * 4 / den)) * max(1, num)
    # Empty track/no pulse, or it already reaches near the end (< 1 measure of slack).
    if last < 0 or last >= stop - measure:
        return track
    extra = [(bt, n, dur) for bt, n, dur in _beat_grid(end_tick, time_sig_map, tpb)
             if bt > last]
    if not extra:
        return track
    # Rebuild: copy the body (without end_of_track) and append the continuation.
    new = mido.MidiTrack()
    new.name = track.name
    cur = 0
    for msg in track:
        if msg.type == "end_of_track":
            continue
        cur += msg.time
        new.append(msg.copy())
    for bt, n, dur in extra:
        new.append(mido.Message("note_on", note=n, velocity=100, time=bt - cur))
        new.append(mido.Message("note_off", note=n, velocity=0, time=dur))
        cur = bt + dur
    new.append(mido.MetaMessage("end_of_track", time=0))
    return new


def build_venue_track(venue_events: list[AbsEvent]) -> mido.MidiTrack:
    """Build the VENUE MidiTrack (name + text events in delta-time)."""
    track = mido.MidiTrack()
    track.name = "VENUE"
    prev = 0
    for ev in sorted(venue_events, key=lambda e: e.abs_tick):
        track.append(ev.msg.copy(time=ev.abs_tick - prev))
        prev = ev.abs_tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track

"""
processor.py — MIDI file processing
"""
from __future__ import annotations
import mido
import os, shutil

from .constants import TRACK_TYPES, DIFF_OFFSET, DRUM_KICK_EXPERT, DRUM_KICK_2X, EXPERT_BASE
from .midi_utils import (
    build_tempo_map, build_time_sig_map, to_abs, to_track, pair_notes, notes_to_events, AbsEvent,
    tick_to_ms, ms_to_abs_tick, rescale_midi_tpb,
)
from .guitar import reduce_guitar
from .chart import is_chart, chart_to_midi
from .convert import normalize_source_midi
from .quality import groove_check, GROOVE_FLOOR, expert_accents
from .drums import reduce_drums_all
from .venue import (
    generate_venue, find_bre_spans, build_venue_track, load_genre, genre_to_theme,
    resolve_sections, generate_animations, _part_instrument,
    build_beat_track, extend_beat_track, find_end_tick, phrase_end_ticks,
    build_crowd, find_pause_spans,
)


_AUDIO_ENERGY = {"calm": 0, "mid": 1, "high": 2}
_ENERGY_NAME = {0: "calm", 1: "mid", 2: "high"}


def _rank01(values: list[float]) -> list[float]:
    """Song-relative rank in [0, 1] (min→0, max→1). Pure-python (no numpy) so the
    MIDI-only path has no extra dependency. Flat list → all 0.5."""
    n = len(values)
    if n <= 1:
        return [0.5] * n
    if max(values) - min(values) < 1e-12:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    for r, i in enumerate(order):
        out[i] = r / (n - 1)
    return out


def _section_midi_cues(mid, sections, part_onsets,
                       tpb: int = 480) -> tuple[list[float], list[float], list[float]]:
    """Three per-section MIDI energy cues (raw, un-normalized):
      • band fullness — how many instruments play in the section (full chorus vs
        sparse verse). A useful cue, BUT it conflates 'no vocals' with 'low energy':
        a heavy instrumental outro/breakdown (guitar+bass+drums going hard, no singer)
        scores only 3/4 and gets wrongly read as calm. So it is no longer used alone.
      • note density — gameplay onsets per beat (all instruments). Structure-free
        heaviness that survives a missing vocal track; the primary MIDI-only cue.
      • mean velocity — the chart's real dynamics (accents vs ghost notes).
    Velocities are read straight from the source MIDI gameplay notes (note < 103)."""
    import bisect
    # Onsets grouped per instrument (sorted) → fullness.
    by_inst: dict[str, list[int]] = {}
    for nm, ons in part_onsets.items():
        inst = _part_instrument(nm)
        if inst:
            by_inst.setdefault(inst, []).extend(ons)
    for inst in by_inst:
        by_inst[inst].sort()
    n_inst = max(1, len(by_inst))
    # (tick, velocity) of every gameplay note, for the mean-velocity cue.
    vel_ticks: list[int] = []
    vel_vals: list[int] = []
    for tr in mid.tracks:
        if _part_instrument(tr.name.strip().upper()) is None:
            continue
        t = 0
        for m in tr:
            t += m.time
            if m.type == "note_on" and m.velocity > 0 and getattr(m, "note", 999) < 103:
                vel_ticks.append(t)
                vel_vals.append(m.velocity)
    order = sorted(range(len(vel_ticks)), key=lambda i: vel_ticks[i])
    vel_ticks = [vel_ticks[i] for i in order]
    vel_vals = [vel_vals[i] for i in order]

    fullness: list[float] = []
    density: list[float] = []
    velocity: list[float] = []
    for s in sections:
        active = sum(1 for ons in by_inst.values()
                     if bisect.bisect_left(ons, s.end) - bisect.bisect_left(ons, s.start) > 0)
        fullness.append(active / n_inst)
        lo = bisect.bisect_left(vel_ticks, s.start)
        hi = bisect.bisect_left(vel_ticks, s.end)
        seg = vel_vals[lo:hi]
        velocity.append(sum(seg) / len(seg) if seg else 0.0)
        beats = max(1.0, (s.end - s.start) / float(tpb))
        density.append((hi - lo) / beats)        # gameplay onsets per beat
    return fullness, density, velocity


def _apply_audio_energy(folder: str, sections, tempo_map, tpb: int,
                        mid, part_onsets, audio_path: str | None = None,
                        theme: str | None = None) -> bool:
    """Set each section's energy using a HYBRID audio+MIDI approach.

    Audio (feel_envelope) decides calm AND high — it has higher agreement
    than the MIDI composite because it measures actual sound.  MIDI composite
    decides mid (audio's overlap zone where calm/high scores are ambiguous).

    Per-theme thresholds: each genre has a different audio baseline (metal is
    louder than slow), so the calm/high thresholds shift by theme.  Calibrated
    against 1450 official section markers across 100 venue learn songs
    (dev/theme_offset_search.py).  Themes without enough data or with inverted
    thresholds (audio can't separate tiers) fall back to the shared baseline.

    Pipeline:
      1. Audio feel_envelope < calm_threshold → calm; > high_threshold → high
      2. Audio ambiguous: MIDI composite decides
      3. energy_spans (sub-spans) add within-section dynamics from audio

    Without audio, falls back to MIDI-only composite thresholds."""
    if not sections:
        return False
    from .venue import SECTION_ENERGY
    full, dens, vel = _section_midi_cues(mid, sections, part_onsets, tpb)
    rfull, rdens, rvel = _rank01(full), _rank01(dens), _rank01(vel)
    struct = [_AUDIO_ENERGY[SECTION_ENERGY.get(s.kind, "calm")] / 2.0 for s in sections]
    # CONTENT-AWARE structural prior (labels lie): a TRANSITION section labelled
    # low-energy (intro/outro/bridge/breakdown/riff → calm prior) but playing at/above
    # the song's median note density is a busy instrumental passage, not a quiet one.
    # Floor its structural prior to 'mid' so the calm LABEL can't drag a heavy
    # instrumental outro/breakdown to 'calm'. Restricted to these transition kinds —
    # verse/chorus keep their own structural energy meaning untouched.
    _DECEPTIVE_CALM = {"intro", "outro", "bridge", "breakdown", "riff", "default"}
    _med = sorted(dens)[len(dens) // 2] if dens else 0.0
    struct = [max(st, 0.5) if (s.kind in _DECEPTIVE_CALM and d >= _med) else st
              for st, d, s in zip(struct, dens, sections)]

    au = None
    from . import audio as _audio
    if _audio.available():
        paths = [audio_path] if audio_path else _audio.find_song_audio(folder)
        if paths:
            au = _audio.section_energy_scores(paths, sections, tempo_map, tpb)
            # Per-section sub-spans so character moods follow the music WITHIN a
            # section (calm intro that builds) instead of one mood per section.
            all_onsets = sorted(set(t for ons in part_onsets.values() for t in ons))
            spans = _audio.section_energy_subspans(paths, sections, tempo_map, tpb,
                                                   onsets=all_onsets)
            if spans is not None:
                for s, sp in zip(sections, spans):
                    s.energy_spans = sp
    # ── MIDI composite: density-led mid/high decider ──────────────────────
    # Optimized against the 100 official venue learn songs' section energy
    # markers. Density + structure lead; fullness supports, velocity is
    # zeroed (charts are often near-flat in velocity).
    midi_comp = [0.70 * rdens[i] + 0.00 * rfull[i] + 0.20 * struct[i]
                 + 0.10 * rvel[i] for i in range(len(sections))]

    # ── Hybrid tier assignment ────────────────────────────────────────────
    # BLENDED approach: combined = alpha * audio + (1-alpha) * midi, then
    # thresholds on the combined score.  Alpha=0.6 means audio leads (60%)
    # but MIDI fills in the mid zone that audio can't separate.
    #
    # Calibrated against 1450 official section markers across 100 songs
    # (dev/blend_compare.py): alpha=0.6 + calm<0.40 + high>0.55 gives the
    # best balance of agreement (41.2%) and distribution (23/35/42 vs
    # official 31/35/35).
    #
    # Per-theme thresholds: slow songs have a quieter baseline, so the
    # calm threshold drops and the high threshold rises.
    _BLEND_ALPHA = 0.60
    _BLEND_CALM = 0.475
    _BLEND_HIGH = 0.570
    _THEME_THRESHOLDS = {
        "slow":    (0.45, 0.70),   # slow songs are quieter
    }
    blend_calm, blend_high = _THEME_THRESHOLDS.get(theme, (_BLEND_CALM, _BLEND_HIGH))

    for i, s in enumerate(sections):
        if au is not None:
            combined = _BLEND_ALPHA * au[i] + (1 - _BLEND_ALPHA) * midi_comp[i]
            if combined < blend_calm:
                s.energy = "calm"
            elif combined > blend_high:
                s.energy = "high"
            else:
                s.energy = "mid"
        else:
            # No audio — MIDI composite only
            if midi_comp[i] < 0.20:
                s.energy = "calm"
            elif midi_comp[i] < 0.60:
                s.energy = "mid"
            else:
                s.energy = "high"

    # Sub-spans from audio are NOT clamped — they provide within-section
    # dynamics that may differ from the section-level tier.

    return au is not None


_INST_TRACK = {"guitar": "PART GUITAR", "bass": "PART BASS", "drums": "PART DRUMS",
               "keys": "PART KEYS", "vocal": "PART VOCALS"}


_SNARE = 97
_KICKS = {95, 96}                                # kick + 2x-kick (double bass)
_TOM_OF_MARKER = {110: 98, 111: 99, 112: 100}   # marker → pad that becomes a TOM


def _double_bass_onsets(mid: mido.MidiFile) -> list[int]:
    """DOUBLE BASS onsets (note 95, 2x-kick) — to reinforce the strobe in zones of
    sustained blast/double-bass, with a looser gap than the normal fills."""
    ticks: list[int] = []
    for tr in mid.tracks:
        if "DRUM" not in tr.name.strip().upper():
            continue
        t = 0
        for m in tr:
            t += m.time
            if m.type == "note_on" and m.velocity > 0 and m.note == 95:
                ticks.append(t)
    return sorted(set(ticks))


def _drum_fill_onsets(mid: mido.MidiFile) -> list[int]:
    """Onsets of SNARE + TOMS + KICK/double-bass, EXCLUDING only cymbals and hi-hat.
    The strobe (Painkiller style) fires on drum fills, consecutive snares AND zones of
    fast kicks/double-bass — but NOT on fast cymbal/hi-hat work.
    In pro-drums a yellow/blue/green pad (98/99/100) is only a TOM if the
    110/111/112 marker is active over it; without a marker it is a CYMBAL/hi-hat → it
    doesn't count. Charts with NO marker at all (non-pro): there is no cymbal concept
    → include the pads."""
    ticks: list[int] = []
    for tr in mid.tracks:
        if "DRUM" not in tr.name.strip().upper():
            continue
        # Spans of the tom-markers (note_on..note_off) per marker.
        spans: dict[int, list[tuple[int, int]]] = {110: [], 111: [], 112: []}
        open_mark: dict[int, int] = {}
        pads: list[tuple[int, int]] = []   # (tick, note) of kick/snare/pads
        has_marker = False
        t = 0
        for m in tr:
            t += m.time
            if m.type == "note_on" and m.velocity > 0:
                if m.note in _TOM_OF_MARKER:
                    open_mark[m.note] = t
                    has_marker = True
                elif m.note in _KICKS or m.note == _SNARE or m.note in (98, 99, 100):
                    pads.append((t, m.note))
            elif (m.type == "note_off" or (m.type == "note_on" and m.velocity == 0)) \
                    and m.note in _TOM_OF_MARKER and m.note in open_mark:
                spans[m.note].append((open_mark.pop(m.note), t))
        marker_of_pad = {v: k for k, v in _TOM_OF_MARKER.items()}
        for tick, note in pads:
            if note in _KICKS or note == _SNARE:
                ticks.append(tick)                       # kick/double-bass + snare
            elif not has_marker:
                ticks.append(tick)                       # non-pro: ambiguous pad counts
            else:
                mk = marker_of_pad[note]
                if any(a <= tick < b for a, b in spans[mk]):
                    ticks.append(tick)                   # pad with an active marker = tom
    return sorted(set(ticks))


def _gather_lyrics(mid: mido.MidiFile) -> list[int]:
    """Ticks of the lyric events (syllables/words). Lets us drive the vocalist even
    without a charted PART VOCALS (lyrics only)."""
    ticks: list[int] = []
    for tr in mid.tracks:
        nm = tr.name.strip().upper()
        if "VOCAL" not in nm and not nm.startswith("HARM"):
            continue
        t = 0
        for m in tr:
            t += m.time
            txt = getattr(m, "text", "")
            if m.type in ("lyrics", "lyric") or (
                    m.type == "text" and txt and not txt.startswith("[")):
                ticks.append(t)
    return sorted(set(ticks))


def _build_anim_track(name: str, markers: list[AbsEvent]) -> mido.MidiTrack:
    """Animation track (mood markers only, no gameplay notes) for an ABSENT
    instrument — brings the character to life from the audio/lyrics."""
    tr = mido.MidiTrack()
    tr.name = name
    prev = 0
    for ev in sorted(markers, key=lambda e: e.abs_tick):
        tr.append(ev.msg.copy(time=ev.abs_tick - prev))
        prev = ev.abs_tick
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


_END_EVT_RE = __import__("re").compile(r"\[\s*end\s*\]")


def _has_end_event(events_track: list[AbsEvent]) -> bool:
    """True if the EVENTS track already has an [end] marker (text or marker)."""
    for ev in events_track:
        txt = getattr(ev.msg, "text", None)
        if ev.msg.type in ("text", "marker") and txt and _END_EVT_RE.search(txt):
            return True
    return False


def _diff_already_charted(events: list[AbsEvent], diff: str) -> bool:
    """True if the track already has hand-authored gems in this difficulty's note
    band (gems + open + force/markers of that difficulty). Used to SKIP regenerating
    a difficulty that the author already charted (option A), avoiding duplicate notes.
    Each difficulty occupies a separate 8-note band (Expert 95-102, Hard 83-90,
    Medium 71-78, Easy 59-66) so the bands never overlap.

    Only a real GEM counts (open at base-1 … Orange at base+4). The force HOPO/strum
    modifiers (base+5/base+6) are NOT gems: a .chart→.mid conversion can leave stray
    force markers in the lower-difficulty bands with no notes under them, which used
    to make us think the difficulty was authored and SKIP generating it."""
    base = EXPERT_BASE + DIFF_OFFSET[diff]
    lo, hi = base - 1, base + 4      # open (base-1) … Orange fret (base+4)
    for ev in events:
        if (ev.msg.type == "note_on" and ev.msg.velocity > 0
                and lo <= getattr(ev.msg, "note", -1) <= hi):
            return True
    return False


def _named_track(name: str) -> mido.MidiTrack:
    """Empty MidiTrack with a name (just end_of_track)."""
    tr = mido.MidiTrack()
    tr.name = name
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def _inject_meta(track: mido.MidiTrack, extra: list[AbsEvent]) -> mido.MidiTrack:
    """Merge text events (extra, in abs_tick) into an existing track, preserving its
    notes and name. Rewrites delta-times. Used for animations."""
    merged = [e for e in to_abs(track) if e.msg.type != "end_of_track"]
    merged += extra
    merged.sort(key=lambda e: e.abs_tick)
    out = mido.MidiTrack()
    prev = 0
    for ev in merged:
        out.append(ev.msg.copy(time=ev.abs_tick - prev))
        prev = ev.abs_tick
    out.append(mido.MetaMessage("end_of_track", time=0))
    return out


# ── Drums — Expert+ (pre-processing; the reduction lives in drums.py) ──────────

def _find_kick_note(track: mido.MidiTrack) -> int:
    counts: dict[int, int] = {}
    for msg in track:
        if msg.type == "note_on" and msg.velocity > 0:
            if msg.note in (DRUM_KICK_EXPERT, 33, 36):
                counts[msg.note] = counts.get(msg.note, 0) + 1
    if not counts:
        return DRUM_KICK_EXPERT
    return DRUM_KICK_EXPERT if DRUM_KICK_EXPERT in counts else max(counts, key=counts.get)


def _find_kick2x_ticks(kick_ons: list[tuple[int, float]], threshold_ms: float) -> set[int]:
    if not kick_ons:
        return set()
    bursts = []; cur = [kick_ons[0]]
    for i in range(1, len(kick_ons)):
        if kick_ons[i][1] - kick_ons[i-1][1] < threshold_ms:
            cur.append(kick_ons[i])
        else:
            bursts.append(cur); cur = [kick_ons[i]]
    bursts.append(cur)
    conv: set[int] = set()
    for burst in bursts:
        if len(burst) >= 3:
            for idx, (tick, _) in enumerate(burst):
                if idx % 2 == 1:
                    conv.add(tick)
    return conv


def _apply_expert_plus(
    events: list[AbsEvent],
    kick_note: int,
    tempo_map: list,
    tpb: int,
    threshold_ms: float,
) -> tuple[list[AbsEvent], dict]:
    from .midi_utils import tick_to_ms
    kick_ons = [(e.abs_tick, tick_to_ms(e.abs_tick, tempo_map, tpb))
                for e in events
                if e.msg.type == "note_on" and e.msg.velocity > 0
                and e.msg.note == kick_note]
    conv_ticks = _find_kick2x_ticks(kick_ons, threshold_ms)
    on2conv = {t: (t in conv_ticks) for t, _ in kick_ons}
    open_ons: dict[int, list[int]] = {}
    new_ev = []
    for ev in events:
        msg = ev.msg
        new_msg = msg.copy()
        if msg.type == "note_on" and msg.velocity > 0 and msg.note == kick_note:
            open_ons.setdefault(kick_note, []).append(ev.abs_tick)
            if ev.abs_tick in conv_ticks:
                new_msg = msg.copy(note=DRUM_KICK_2X)
        elif (msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)) \
                and msg.note == kick_note:
            stack = open_ons.get(kick_note, [])
            if stack:
                on_t = stack.pop(0)
                if on2conv.get(on_t):
                    new_msg = msg.copy(note=DRUM_KICK_2X)
        new_ev.append(AbsEvent(ev.abs_tick, new_msg))
    doubles_kept = sum(1 for i in range(len(kick_ons)-1)
        if kick_ons[i+1][1]-kick_ons[i][1] < threshold_ms
        and kick_ons[i][0] not in conv_ticks
        and kick_ons[i+1][0] not in conv_ticks)
    return new_ev, {
        "total_kicks": len(kick_ons),
        "converted": len(conv_ticks),
        "doubles_kept": doubles_kept,
    }


# NOTE: the drum reduction was rewritten guided by the Customs Book (pp.182–186) and
# now lives in `downcharter/drums.py` (gem model + cascade Hard←Expert,
# Medium←Hard, Easy←Medium). The old per-lane grid parameters
# (DRUM_PAD_GAP_BEATS, DRUM_KICK_GAP_BEATS, _thin_lane_grid*, fill-collapse,
# snare-grid, etc.) were removed. The history is archived in CLAUDE.md.
# ── Main processing ───────────────────────────────────────────────────────────

def process_midi(
    src_path: str,
    dst_path: str,
    diffs_to_gen: list[str],
    do_expert_plus: bool = True,
    threshold_ms: float = 125.0,
    do_venue: bool = False,
    audio_path: str | None = None,
    do_lipsync: bool = False,
    do_talkies: bool = False,
    do_drum_anim: bool = True,
) -> dict:
    """
    Process a MIDI file:
      - Generate reduced difficulties for every recognized instrument
      - Optional: convert double bass to Expert+ in PART DRUMS
      - Optional: generate a VENUE track (camera + lights + post-proc)
    Returns statistics.
    """
    try:
        mid = mido.MidiFile(src_path)
    except (ValueError, IOError):
        # Some MIDIs (e.g. miracle) have sysex/data bytes outside 0..127.
        # clip=True clamps them to the valid MIDI range instead of blowing up.
        mid = mido.MidiFile(src_path, clip=True)
    # RB3 requires 480 TPB; .chart imports keep the chart's native resolution
    # (often 192). Normalise up-front so all downstream tick math AND the output
    # are 480.
    rescale_midi_tpb(mid, 480)
    # Onyx-style source-format auto-detect: rename legacy tracks to RB names and
    # remap FoF star-power (note 103) to RB overdrive (116). No-op on RB-format
    # charts (incl. our own .chart imports). Runs before difficulty generation so
    # the whole pipeline sees standard track names and overdrive on 116.
    norm = normalize_source_midi(mid)
    tpb = mid.ticks_per_beat
    tempo_map = build_tempo_map(mid)
    time_sig_map = build_time_sig_map(mid)
    new_mid = mido.MidiFile(ticks_per_beat=tpb, type=mid.type)

    stats = {
        "total_kicks": 0, "converted_2x": 0, "doubles_kept": 0,
        "tracks_processed": 0, "groove_warnings": [], "diffs_skipped": [],
        "venue_events": 0, "venue_skipped": False,
        "venue_theme": None, "anim_events": 0, "beat_added": False,
        "beat_extended": False,
        "audio_used": False, "audio_drums": False, "audio_lyrics": False,
        "audio_anim_instr": [],
        "tracks_renamed": norm["tracks_renamed"], "sp_remapped": norm["sp_remapped"],
    }

    # Signals for generating the venue
    events_track: list[AbsEvent] = []
    bre_spans: list[tuple[int, int]] = []
    accent_src: list[AbsEvent] = []   # guitar events for accents
    all_onsets: set[int] = set()      # onsets of ALL instruments (density)
    part_onsets: dict[str, list[int]] = {}  # onsets per PART track (animations)
    vocal_tracks: list[str] = []      # names of the vocal PART tracks (charted or lyrics-only)
    has_beat = any(t.name.strip().upper() == "BEAT" for t in mid.tracks)
    has_venue = any(t.name.strip().upper() == "VENUE" for t in mid.tracks)
    song_end = 0

    for track in mid.tracks:
        track_name = track.name.strip().upper()
        track_type = TRACK_TYPES.get(track_name)

        events = to_abs(track)
        if events:
            song_end = max(song_end, events[-1].abs_tick)

        # Capturing signals for the venue
        if do_venue:
            if track_name == "EVENTS":
                events_track = events
            # A real BRE is authored on the melodic tracks (notes 120–124). On DRUMS
            # those notes are *drum fills* (activation lanes), not BREs → exclude.
            if track_type == "guitar":
                bre_spans += find_bre_spans(events)
            # Accents: prefer PART GUITAR; otherwise the 1st guitar track
            if track_type == "guitar" and (
                    not accent_src or track_name == "PART GUITAR"):
                accent_src = events
            # Gameplay onsets (note < 103 excludes markers) of each instrument
            if _part_instrument(track_name) is not None:
                if _part_instrument(track_name) == "vocal":
                    vocal_tracks.append(track.name)
                ons = [e.abs_tick for e in events
                       if e.msg.type == "note_on" and e.msg.velocity > 0
                       and getattr(e.msg, "note", 999) < 103]
                if ons:
                    part_onsets[track.name] = ons
                    all_onsets.update(ons)
            if track_name == "VENUE":
                # Option A: an existing VENUE is the author's work — keep it intact
                # (copied below via the track_type-None path) and skip generating ours.
                pass

        new_track = mido.MidiTrack()
        new_track.name = track.name
        new_mid.tracks.append(new_track)

        if track_type is None:
            # Unrecognized track — copy intact (PS sysex preserved; see
            # _restore_ps_sysex at the end to restore the 0xFF byte clamped by clip).
            for msg in track:
                if msg.type == "sysex":
                    stats["sysex_kept"] = stats.get("sysex_kept", 0) + 1
                new_track.append(msg.copy())
            continue

        stats["tracks_processed"] += 1

        if track_type == "drums":
            kick_note = _find_kick_note(track)
            if do_expert_plus:
                events, ks = _apply_expert_plus(
                    events, kick_note, tempo_map, tpb, threshold_ms)
                stats["total_kicks"]   += ks["total_kicks"]
                stats["converted_2x"] += ks["converted"]
                stats["doubles_kept"] += ks["doubles_kept"]

            all_events = list(events)
            # Option A: don't regenerate a difficulty the author already charted.
            gen_diffs = []
            for diff in diffs_to_gen:
                if _diff_already_charted(events, diff):
                    stats["diffs_skipped"].append(f"{track.name} {diff}")
                else:
                    gen_diffs.append(diff)
            reduced_by_diff = reduce_drums_all(
                events, gen_diffs, tempo_map, tpb, time_sig_map)
            for diff in gen_diffs:
                for ev in reduced_by_diff.get(diff, []):
                    if ev.msg.type in ("note_on", "note_off"):
                        all_events.append(ev)

        elif track_type == "guitar":
            # Apply hand position map to Expert guitar/bass (generates 101/102
            # force-HOPO/strum markers throughout the song, not just at start).
            from .guitar_handmap import apply_handmap
            events = apply_handmap(events, tpb)
            all_events = list(events)
            for diff in diffs_to_gen:
                # Option A: don't regenerate a difficulty the author already charted.
                if _diff_already_charted(events, diff):
                    stats["diffs_skipped"].append(f"{track.name} {diff}")
                    continue
                reduced = reduce_guitar(events, diff, tempo_map, tpb, time_sig_map)
                # Quality guard: flags reductions that lost the groove.
                if diff in GROOVE_FLOOR:
                    ok, score = groove_check(events, reduced, diff, tpb)
                    if not ok:
                        stats["groove_warnings"].append(
                            f"{track.name} {diff}: groove {score:.0%} "
                            f"(< {GROOVE_FLOOR[diff]:.0%})")
                for ev in reduced:
                    if ev.msg.type in ("note_on", "note_off"):
                        all_events.append(ev)

        # Sort and write to the single track. The sysex (e.g. Phase Shift "F0 50 53…",
        # open/tap notes) are PRESERVED — mido reads them with clip (0xFF→0x7F) and
        # _restore_ps_sysex restores the 0xFF in the final file so they end up like the
        # original (which YARG reads fine).
        all_events.sort(key=lambda e: e.abs_tick)
        prev = 0
        for ev in all_events:
            if ev.msg.type == "sysex":
                stats["sysex_kept"] = stats.get("sysex_kept", 0) + 1
            new_track.append(ev.msg.copy(time=ev.abs_tick - prev))
            prev = ev.abs_tick

    if do_venue:
        genre = load_genre(os.path.dirname(os.path.abspath(src_path)))
        theme = genre_to_theme(genre)
        accents = expert_accents(accent_src, tpb) if accent_src else []
        onsets = sorted(all_onsets)
        # Resolved sections (multi-instrument) — shared by venue+animations.
        sections = resolve_sections(events_track, song_end, onsets, time_sig_map, tpb)
        # AUDIO refinement (optional): mixes the real loudness with the structural
        # energy. With no audio/libs, sections.energy stays None (MIDI-only intact).
        stats["audio_used"] = _apply_audio_energy(
            os.path.dirname(os.path.abspath(src_path)), sections, tempo_map, tpb,
            mid, part_onsets, audio_path, theme)
        # Drum hits for the lightshow (synced keyframes + pyro).
        drum_onsets = sorted(
            t for nm, ons in part_onsets.items()
            if _part_instrument(nm) == "drums" for t in ons)
        # Snare + toms (fills/rolls) for the strobe — no cymbals/hi-hat.
        fill_onsets = _drum_fill_onsets(mid)
        dbass_onsets = _double_bass_onsets(mid)         # double-bass to reinforce strobe
        # Onsets per instrument (spotlights) + number of harmonies (sing-along).
        inst_onsets: dict[str, list[int]] = {}
        for nm, ons in part_onsets.items():
            inst = _part_instrument(nm)
            if inst:
                inst_onsets.setdefault(inst, []).extend(ons)
        for inst in inst_onsets:
            inst_onsets[inst].sort()
        n_harm = sum(1 for t in mid.tracks
                     if t.name.strip().upper() in ("HARM1", "HARM2", "HARM3"))

        # ── AUDIO Layers 1+2: enrich charts with few instruments ──────────────
        # Layer 1: pseudo-drums from the audio when there's no PART DRUMS (lightshow/pyro).
        # Layer 2: animate ABSENT instruments via per-band energy + lyrics.
        pseudo_anim: dict[str, list[int]] = {}
        present = set(inst_onsets)
        # REAL vocals (charted or lyrics) enable crowd/sing-along; the audio proxy
        # only enables directed performance/spotlight (no words to sing).
        vocal_real = "vocal" in present

        # Vocals from LYRICS (audio-independent): if there are no charted vocals but
        # there are lyrics in the MIDI, the vocals are driven by them (sing-along/directed/anim).
        lyrics = _gather_lyrics(mid)
        if "vocal" not in present and lyrics:
            inst_onsets.setdefault("vocal", lyrics)
            vocal_real = True
            stats["audio_lyrics"] = True
            # PART VOCALS exists (lyrics) but has no gems → it didn't enter part_onsets,
            # so the vocalist stayed ALWAYS idle (mood markers never injected).
            # Drive the vocalist's animation from the LYRICS on the existing track itself.
            vt = next((n for n in vocal_tracks
                       if n.strip().upper() == "PART VOCALS"), None) \
                or (vocal_tracks[0] if vocal_tracks else None)
            if vt is not None:
                part_onsets[vt] = lyrics       # → generate_animations animates the track
            else:
                pseudo_anim["vocal"] = lyrics   # no track → create a dedicated animation

        from . import audio as _audio
        a_paths = ([audio_path] if audio_path
                   else _audio.find_song_audio(os.path.dirname(os.path.abspath(src_path))))
        audio_accents = None
        energy_env = None
        audio_strobe = None
        drop_ticks = None
        band_activity: dict[str, list[int]] = {}
        if _audio.available() and a_paths:
            if not drum_onsets:                       # Layer 1
                pd = _audio.percussive_onset_ticks(a_paths, tempo_map, tpb)
                if pd:
                    drum_onsets = pd
                    stats["audio_drums"] = True
            # Strong audio transients (chorus hits, crashes, stabs) → snap light
            # changes / pyro to real musical hits, incl. audio-only ones.
            audio_accents = _audio.flux_accents(a_paths, tempo_map, tpb)
            if audio_accents:
                stats["audio_accents"] = len(audio_accents)
            # Color temperature per section from the audio timbre (spectral centroid):
            # dark→warm, bright→cool. Sets s.warmth so the lightshow tints each section.
            if sections:
                warmth = _audio.section_brightness_tiers(
                    a_paths, sections, tempo_map, tpb)
                if warmth:
                    for s, w in zip(sections, warmth):
                        s.warmth = w
                    stats["audio_warm"] = sum(1 for w in warmth if w == "warm")
                    stats["audio_cool"] = sum(1 for w in warmth if w == "cool")
            # Within-section energy envelope (sub-section composite score) → the light
            # cadence speeds up in the loud half of a section, eases in the quiet half.
            if sections:
                energy_env = _audio.energy_envelope(
                    a_paths, sections, tempo_map, tpb)
                if energy_env:
                    stats["audio_env_spans"] = len(energy_env)
            # Sustained spectral-flux walls (blast/tremolo) → continuous strobe, even
            # for audio-only walls the MIDI drums miss.
            audio_strobe = _audio.flux_strobe_spans(a_paths, tempo_map, tpb)
            if audio_strobe:
                stats["audio_strobe_spans"] = len(audio_strobe)
            # Blackout anchors = start of calm low-energy regions (ground-truth: the
            # 20 official venues put blackout in calm states, not at a sharp fall).
            if sections:
                drop_ticks = _audio.calm_blackout_ticks(
                    a_paths, sections, tempo_map, tpb)
                if drop_ticks:
                    stats["audio_blackout_calm"] = len(drop_ticks)
            # Audio band activity for camera identity refinement
            band_activity = {}
            for band in ("bass", "drums", "lead"):
                ba = _audio.band_activity_ticks(a_paths, tempo_map, tpb, band)
                if ba:
                    band_activity[band] = ba
            if band_activity:
                stats["audio_band_activity"] = list(band_activity.keys())
            # Character ANIMATION from the audio: ONLY vocals. Animating an ABSENT
            # bass/guitar/drums/keys creates a PART track with no charted gems —
            # RB3 doesn't render the character (camera/animation pointing at nothing)
            # and YARG doesn't even support characters. We only animate instruments
            # with a REAL chart. Vocals are the exception: the "animation" is the
            # mouth lipsync, which makes sense from the singing detected in the audio
            # (or from the lyrics) even without charted gems — the vocalist sings.
            # (Keys were already out for the same reason; now bass/guitar/drums follow
            # the rule.)
            if "vocal" not in present and not lyrics:  # no lyrics → 'lead' band
                ba = _audio.band_activity_ticks(a_paths, tempo_map, tpb, "lead")
                if ba:
                    pseudo_anim["vocal"] = ba
                    # Even without a chart/lyrics, we identified when there's singing →
                    # enables directed_vocals + spotlight (but NOT crowd/sing: no words).
                    inst_onsets.setdefault("vocal", ba)
        stats["audio_anim_instr"] = sorted(pseudo_anim)
        # Flag for the venue: only real vocals (chart/lyrics) authorize crowd/sing-along.
        if vocal_real:
            inst_onsets["_vocal_real"] = inst_onsets.get("vocal", [])

        if has_venue:
            # Option A: keep the author's existing VENUE (already copied intact) and
            # don't generate ours. Animations/BEAT/[end] below still run as usual.
            stats["venue_skipped"] = True
        else:
            venue_events = generate_venue(
                events_track, bre_spans, song_end, tempo_map, time_sig_map, tpb,
                theme, accents, onsets, sections=sections, drum_onsets=drum_onsets,
                inst_onsets=inst_onsets, n_harm=n_harm, fill_onsets=fill_onsets,
                dbass_onsets=dbass_onsets, audio_onsets=audio_accents,
                energy_env=energy_env, audio_strobe_spans=audio_strobe,
                drop_ticks=drop_ticks, band_activity=band_activity)
            new_mid.tracks.append(build_venue_track(venue_events))
            stats["venue_events"] = len(venue_events)
            stats["venue_theme"] = theme
            # Crowd state events ([crowd_*]) belong on the EVENTS track (RB3/YARG read
            # them there, NOT from VENUE). Tie them to the same energy map and inject.
            if sections:
                pause_spans = find_pause_spans(onsets, time_sig_map, tpb)
                crowd_events = build_crowd(sections, tpb, pause_spans)
                if crowd_events:
                    ei = next((i for i, t in enumerate(new_mid.tracks)
                               if t.name.strip().upper() == "EVENTS"), None)
                    if ei is not None:
                        new_mid.tracks[ei] = _inject_meta(new_mid.tracks[ei], crowd_events)
                    else:
                        new_mid.tracks.append(
                            _inject_meta(_named_track("EVENTS"), crowd_events))
                    stats["crowd_events"] = len(crowd_events)

        # Per-PART-track character animations (mood markers).
        # Vocal phrase ends (105/106) → the vocalist only lowers the mic at the end
        # of the phrase, not 1 beat after the last syllable.
        v_track = next((t for t in new_mid.tracks
                        if t.name.strip().upper() == "PART VOCALS"), None)
        v_pe = phrase_end_ticks(v_track) if v_track is not None else None
        anim = generate_animations(part_onsets, sections, theme, tpb, time_sig_map,
                                   vocal_phrase_ends=v_pe)
        anim_total = 0
        for i, tr in enumerate(new_mid.tracks):
            markers = anim.get(tr.name)
            if markers:
                new_mid.tracks[i] = _inject_meta(tr, markers)
                anim_total += len(markers)
        # Layer 2: animation tracks for ABSENT instruments (audio/lyrics).
        if pseudo_anim:
            from .venue import build_animations, instrument_extras
            existing = {t.name.strip().upper() for t in new_mid.tracks}
            for inst, ons in pseudo_anim.items():
                tname = _INST_TRACK[inst]
                if tname.upper() in existing:
                    continue
                markers = build_animations(ons, sections, tpb, time_sig_map, inst)
                markers += instrument_extras(inst, ons, sections, tpb)
                if markers:
                    new_mid.tracks.append(_build_anim_track(tname, markers))
                    anim_total += len(markers)
                    existing.add(tname.upper())
        stats["anim_events"] = anim_total

        # The instruments often stop BEFORE the audio really ends (outro, fade-out,
        # ring-out). The [end] event and the BEAT pulse must reach the END OF THE AUDIO,
        # otherwise the characters freeze and the camera drifts for the remaining seconds.
        # Compute the audio end in ticks and use it as a floor for song_end.
        song_end_audio = song_end
        try:
            from . import audio as _audio
            a_paths = ([audio_path] if audio_path
                       else _audio.find_song_audio(os.path.dirname(os.path.abspath(src_path))))
            dur_s = _audio.audio_duration_seconds(a_paths) if a_paths else None
            if dur_s:
                a_tick = ms_to_abs_tick(dur_s * 1000.0, tempo_map, tpb)
                if a_tick > song_end_audio:
                    song_end_audio = a_tick
                    stats["end_extended_to_audio"] = True
        except Exception:
            pass

        # [end] event: RB3 needs it to know where the song ends. Without it,
        # the BEAT track runs out and the BandDirector leaves the characters in a
        # slow drifting camera until the audio ends. The 20 official venues ALWAYS
        # have [end] on the last tick. If it's missing, we inject it into EVENTS at the
        # audio end.
        if song_end > 0 and not _has_end_event(events_track):
            ei = next((i for i, t in enumerate(new_mid.tracks)
                       if t.name.strip().upper() == "EVENTS"), None)
            end_ev = [AbsEvent(song_end_audio, mido.MetaMessage("text", text="[end]", time=0))]
            if ei is not None:
                new_mid.tracks[ei] = _inject_meta(new_mid.tracks[ei], end_ev)
            else:
                new_mid.tracks.append(_inject_meta(_named_track("EVENTS"), end_ev))
            stats["end_added"] = True

        # BEAT track (pulse): if the song doesn't have one, we create it. If it does,
        # we make sure it reaches the end — a BEAT that ends early (the case of the midi
        # in the "beattrack" folder) leaves the characters FROZEN in-game, because the
        # BandDirector stops receiving the pulse. In that case we extend it.
        if song_end > 0:
            end_tick = max(find_end_tick(events_track, song_end), song_end_audio)
            if not has_beat:
                new_mid.tracks.append(build_beat_track(end_tick, time_sig_map, tpb))
                stats["beat_added"] = True
            else:
                bi = next((i for i, t in enumerate(new_mid.tracks)
                           if t.name.strip().upper() == "BEAT"), None)
                if bi is not None:
                    ext = extend_beat_track(new_mid.tracks[bi], end_tick,
                                            time_sig_map, tpb)
                    if ext is not new_mid.tracks[bi]:
                        new_mid.tracks[bi] = ext
                        stats["beat_extended"] = True

    # Drummer limb animations (PART DRUMS notes 24-51): RB3 needs them authored or
    # the drummer stays idle. Synthesise them HERE, into the notes.mid track, from the
    # Expert gems — NOT at conversion time. No-op if the drums track is already animated.
    if do_drum_anim:
        try:
            from . import convert as _convert
            new_mid, drum_anim_stats = _convert.generate_drum_animations(new_mid)
            if drum_anim_stats.get("added"):
                stats["drum_anim_events"] = drum_anim_stats["added"]
        except Exception:
            pass

    # Generate talkies: for songs with lyrics, chart PART VOCALS as talky/unpitched
    # vocals (extended to the next syllable + gap). Onyx generates the lipsync itself
    # from the length of these tubes in the .ini→RB3/PS3 build.
    if (do_talkies or do_lipsync) and song_end > 0:
        _apply_lipsync(new_mid, dst_path, tempo_map, tpb, song_end, stats,
                       do_talkies=do_talkies, do_lipsync=do_lipsync)

    new_mid.save(dst_path)
    if stats.get("sysex_kept"):
        _restore_ps_sysex(dst_path)

    return stats


_VOCAL_TALKY_PITCH = 50          # note within the vocal range (36-84); irrelevant for talky


def _abs_phrase_ends(abs_evts: list[AbsEvent]) -> list[int]:
    """Ticks of the phrase-marker note_offs (105/106) in a list of AbsEvent."""
    open_at: dict[int, int] = {}
    ends: list[int] = []
    for e in sorted(abs_evts, key=lambda x: x.abs_tick):
        m = e.msg
        note = getattr(m, "note", None)
        if note not in (105, 106):
            continue
        if m.type == "note_on" and m.velocity > 0:
            open_at[note] = e.abs_tick
        elif m.type == "note_off" or (m.type == "note_on" and m.velocity == 0):
            if note in open_at:
                ends.append(e.abs_tick)
                del open_at[note]
    return sorted(set(ends))


def _gen_phrase_notes(notes: list[tuple[int, int]], tpb: int) -> list[tuple[int, int]]:
    """Group gems into phrases (gap > 2 beats) → spans for phrase markers (105)."""
    if not notes:
        return []
    ns = sorted(notes)
    gap = tpb * 2
    phrases: list[tuple[int, int]] = []
    start, end = ns[0]
    for s, e in ns[1:]:
        if s - end > gap:
            phrases.append((start, end))
            start = s
        end = max(end, e)
    phrases.append((start, end))
    return phrases


def _chart_vocals_from_lyrics(new_mid, tpb: int, stats,
                              tempo_map=None, folder: str | None = None,
                              write_gems: bool = True) -> None:
    """Create unpitched gems (talky, lyric '#') in PART VOCALS from the lyrics.

    Each note extends to near the next syllable / phrase end, leaving a GAP (notes
    never glued together). RB now has charted vocals; Onyx's autoLipsync reads these
    tubes (whose LENGTH defines the mouth opening) and generates a lipsync that holds
    the vowel — instead of the pointwise lipsync it fabricated from uncharted lyrics.
    It doesn't pitch them (we don't know the melody) → talky, full note.

    Audio confirmation: with an isolated vocal stem in `folder`, the note is not
    stretched blindly to the next syllable — it is cut where the singer actually
    stops (RMS envelope drops), avoiding fake long sustains across silent gaps.
    Falls back to pure geometry when no vocal stem is available."""
    idx = next((i for i, t in enumerate(new_mid.tracks)
                if t.name.strip().upper() == "PART VOCALS"), None)
    if idx is None:
        return
    track = new_mid.tracks[idx]
    abs_evts = [e for e in to_abs(track) if e.msg.type != "end_of_track"]

    # already charted (pitched gems in the vocal range)? then we don't touch it.
    # But when write_gems=False (lipsync-only), we still need spans from lyrics.
    if write_gems and any(e.msg.type == "note_on" and getattr(e.msg, "velocity", 0) > 0
           and 36 <= e.msg.note <= 84 for e in abs_evts):
        return

    markers = {"+", "#", "^", "*", "%"}
    syl = [e for e in abs_evts
           if (e.msg.type in ("lyrics", "lyric")
               or (e.msg.type == "text" and getattr(e.msg, "text", "")
                   and not e.msg.text.startswith("[")))
           and e.msg.text.strip() not in markers]
    if not syl:
        return
    syl.sort(key=lambda e: e.abs_tick)

    # Audio confirmation: load the vocal stem's voice-activity envelope (if any).
    va = None
    if tempo_map is not None and folder:
        try:
            from . import audio as _audio
            if _audio.available():
                vocal = _audio.find_vocal_audio(folder)
                if vocal:
                    va = _audio.voice_activity(vocal)
                elif vocal is None:
                    # No separated stems — try extracting vocal channels from .mogg
                    for f in os.listdir(folder):
                        if f.lower().endswith(".mogg"):
                            va = _audio.voice_activity_from_mogg(
                                os.path.join(folder, f))
                            break
        except Exception:
            va = None

    pends = _abs_phrase_ends(abs_evts)
    n = len(syl)
    trimmed = 0
    gems: list[tuple[int, int]] = []
    # (start_s, end_s, text, gain) per syllable → drives the LIPSYNC1 viseme track.
    lip_spans: list[tuple[float, float, str, float]] = []
    for i, e in enumerate(syl):
        t = e.abs_tick
        nxt = syl[i + 1].abs_tick if i + 1 < n else None
        pe = next((p for p in pends if p > t + 1), None)
        if nxt is not None and (pe is None or nxt <= pe):
            target = nxt                       # next syllable in the same phrase
        elif pe is not None:
            target = pe                        # last of the phrase → hold until the end
        else:
            target = t + tpb                   # last with no phrase → ~1 beat
        # Confirm the sustain against the vocal audio: if the voice goes silent
        # before `target`, end the note there instead of faking a long sustain.
        if va is not None:
            start_s = tick_to_ms(t, tempo_map, tpb) / 1000.0
            ceil_s = tick_to_ms(target, tempo_map, tpb) / 1000.0
            off_s = _audio.voice_offset_s(va, start_s, ceil_s)
            if off_s is not None:
                a_target = _audio._ms_to_tick(off_s * 1000.0, tempo_map, tpb)
                a_target = max(t + tpb // 4, a_target)   # keep a minimum note
                if a_target < target - tpb // 8:         # only if meaningfully shorter
                    target = a_target
                    trimmed += 1
        span = max(1, target - t)
        g = max(1, min(tpb // 6, span // 3))   # gap before the boundary (never glued)
        end = t + max(1, span - g)
        gems.append((t, end))
        # mark the lyric as talky ('#' at the end, after '-'/'='), if it isn't already.
        cur = e.msg.text
        if write_gems and not cur.rstrip().endswith(("#", "^")):
            e.msg = e.msg.copy(text=cur + "#")
        # Record the span (seconds) + loudness gain for the viseme track.
        if tempo_map is not None:
            s_s = tick_to_ms(t, tempo_map, tpb) / 1000.0
            e_s = tick_to_ms(end, tempo_map, tpb) / 1000.0
            gain = _audio.syllable_gain(va, s_s, e_s) if va is not None else 1.0
            lip_spans.append((s_s, e_s, cur, gain))

    # When only the LIPSYNC1 track is requested (talkies off), don't touch PART
    # VOCALS — just hand back the spans that drive the viseme track.
    if not write_gems:
        return lip_spans

    # phrase markers (105) if the track doesn't have them — RB needs them for the vocals.
    have_phrases = any(e.msg.type == "note_on" and getattr(e.msg, "velocity", 0) > 0
                       and e.msg.note in (105, 106) for e in abs_evts)
    phrase_spans = [] if have_phrases else _gen_phrase_notes(gems, tpb)

    for s_t, e_t in gems:
        abs_evts.append(AbsEvent(s_t, mido.Message(
            "note_on", note=_VOCAL_TALKY_PITCH, velocity=96, time=0)))
        abs_evts.append(AbsEvent(e_t, mido.Message(
            "note_off", note=_VOCAL_TALKY_PITCH, velocity=0, time=0)))
    for s_t, e_t in phrase_spans:
        abs_evts.append(AbsEvent(s_t, mido.Message("note_on", note=105, velocity=96, time=0)))
        abs_evts.append(AbsEvent(e_t, mido.Message("note_off", note=105, velocity=0, time=0)))

    new_tr = to_track(abs_evts)
    new_tr.append(mido.MetaMessage("end_of_track", time=0))
    new_mid.tracks[idx] = new_tr
    stats["vocals_charted"] = len(gems)
    if trimmed:
        stats["vocals_trimmed"] = trimmed
    if phrase_spans:
        stats["vocal_phrases_gen"] = len(phrase_spans)
    return lip_spans


def _apply_lipsync(new_mid, dst_path, tempo_map, tpb, song_end, stats,
                   do_talkies: bool = True, do_lipsync: bool = True) -> None:
    """Two independent lipsync outputs from the lyrics, each toggled separately:

    1. Talky vocals in PART VOCALS (``do_talkies``) — the path RB/Onyx's `.ini` import
       actually uses (charted tubes → autoLipsync). Works in-game today.
    2. A LIPSYNC1 viseme track (``do_lipsync``) — text-commands
       `[<viseme> <weight>[ hold|ease]]`, the exact format Onyx parses (confirmed in
       `Onyx/MIDI/Track/Lipsync.hs`: track name `LIPSYNC1`, graph token = curve out of
       the keyframe). Sparse keyframes (held vowels, diphthong glides), not dense 30fps
       deltas. Consumed by Onyx's milo workflow and future-proof for YARG (its
       `LoadLipsyncFromMilo` has a TODO to parse lipsync from MIDI). Timing/weight are
       audio-guided from the same spans as (1), beating both engines' built-in
       geometric generators.

    The spans are computed from the lyrics regardless; ``write_gems`` only controls
    whether PART VOCALS is rewritten, so the LIPSYNC1 track can be generated without
    charting talkies (and vice-versa)."""
    if not do_talkies and not do_lipsync:
        return
    folder = os.path.dirname(os.path.abspath(dst_path))
    spans = _chart_vocals_from_lyrics(new_mid, tpb, stats, tempo_map, folder,
                                      write_gems=do_talkies) or []
    if do_lipsync and spans and tempo_map is not None and song_end > 0:
        try:
            from . import lipsync as _lip
            # Extract phrase ends (seconds) from PART VOCALS for facial animation.
            phrase_ends_s: list[float] = []
            pv = next((t for t in new_mid.tracks
                       if t.name.strip().upper() == "PART VOCALS"), None)
            if pv is not None:
                pe_ticks = _abs_phrase_ends(list(to_abs(pv)))
                phrase_ends_s = [tick_to_ms(t, tempo_map, tpb) / 1000.0
                                 for t in sorted(set(pe_ticks))]
            song_len_s = tick_to_ms(song_end, tempo_map, tpb) / 1000.0
            keyframes = _lip.lipsync_keyframes_from_spans(
                spans,
                phrase_ends=phrase_ends_s,
                song_len_s=song_len_s,
                facial_seed=42,  # deterministic for reproducible builds
            )
            if keyframes:
                tr = _build_lipsync_track(keyframes, tempo_map, tpb)
                li = next((i for i, t in enumerate(new_mid.tracks)
                           if t.name.strip().upper() == "LIPSYNC1"), None)
                if li is not None:
                    new_mid.tracks[li] = tr
                else:
                    new_mid.tracks.append(tr)
                stats["lipsync_events"] = len(keyframes)
        except Exception:
            pass
        # NOTE: the .milo file is NOT built here. Processing only authors the
        # audio-guided LIPSYNC1 track (above); the actual <id>.milo_ps3 is created
        # later, at conversion time (ps3build.build_ps3_song), from this very track
        # / its charted talky vocals — so there's exactly one milo, made when the
        # PS3 pack is assembled.


_LIPSYNC_GRAPH_TOKEN = {"linear": "", "hold": " hold", "ease": " ease"}


def _build_lipsync_track(keyframes, tempo_map, tpb):
    """A `LIPSYNC1` MIDI track of `[<viseme> <weight>[ hold|ease]]` text-commands from
    sparse keyframes (time_s, viseme, weight, graph). Times → ticks via the tempo map.
    The graph token is the Onyx curve OUT of the keyframe (default linear, omitted).
    Opens with `[lang en]` so the renderer picks the right rules.

    Keyframes at syllable boundaries can collapse to the same tick after ms→tick
    conversion, producing two events for the same viseme at the same tick (e.g.
    ``[Cage_hi 140]`` and ``[Cage_hi 0]`` both at T).  We deduplicate: for each
    (tick, viseme) pair, only the LAST event survives — it represents the state
    going forward from that tick."""
    from . import audio as _audio
    abs_evts = [
        AbsEvent(0, mido.MetaMessage("track_name", name="LIPSYNC1", time=0)),
        AbsEvent(0, mido.MetaMessage("text", text="[lang en]", time=0)),
    ]
    for time_s, viseme, weight, graph in keyframes:
        tick = _audio._ms_to_tick(time_s * 1000.0, tempo_map, tpb)
        tok = _LIPSYNC_GRAPH_TOKEN.get(graph, "")
        abs_evts.append(AbsEvent(tick, mido.MetaMessage(
            "text", text=f"[{viseme} {weight}{tok}]", time=0)))

    # Deduplicate: for each (tick, viseme) keep only the LAST event.
    # Keyframes are already sorted by (time_s, viseme), so after tick-conversion,
    # same-tick events are contiguous; we walk backwards and take the first
    # occurrence of each (tick, viseme) which is the LAST in original order.
    seen: set[tuple[int, str]] = set()
    deduped: list[AbsEvent] = [abs_evts[0], abs_evts[1]]  # track_name + [lang en]
    for ev in reversed(abs_evts[2:]):
        key = (ev.abs_tick, ev.msg.text)
        # Extract viseme name from "[viseme weight]" format
        txt = ev.msg.text
        if txt.startswith("[") and " " in txt:
            vis = txt.split("[")[1].split(" ")[0]
            pair = (ev.abs_tick, vis)
            if pair in seen:
                continue
            seen.add(pair)
        deduped.append(ev)
    deduped.reverse()  # back to chronological
    abs_evts = deduped

    tr = to_track(abs_evts)
    tr.append(mido.MetaMessage("end_of_track", time=0))
    return tr


def _restore_ps_sysex(path: str) -> None:
    """Restore the Phase Shift sysex to their original state. mido only reads/writes
    bytes 0..127, so the difficulty byte 0xFF ("all difficulties") is clamped to 0x7F
    on load. Since 127 is never a valid PS difficulty (only 0–3 or 0xFF), the whole
    `50 53 00 00 7F` pattern is certainly a clamped 0xFF → it's restored to 0xFF,
    leaving the sysex byte-for-byte identical to the original (which YARG reads fine).
    The patch only changes that byte; length and structure stay intact."""
    with open(path, "rb") as f:
        data = f.read()
    fixed = data.replace(b"\x50\x53\x00\x00\x7f", b"\x50\x53\x00\x00\xff")
    if fixed != data:
        with open(path, "wb") as f:
            f.write(fixed)


# ── Folder utilities ──────────────────────────────────────────────────────────

def find_midis(folder: str) -> list[str]:
    result = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".mid") and not f.lower().endswith(".bak.mid"):
                result.append(os.path.join(root, f))
    return result


def find_charts(folder: str) -> list[str]:
    """.chart files still to process (ignores those that already have the .mid
    generated alongside — those were converted in a previous run and are processed
    as MIDI)."""
    result = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".chart") and not f.lower().endswith(".bak.chart"):
                result.append(os.path.join(root, f))
    return result


def _prepare_chart(path: str, log_fn) -> str | None:
    """Convert a .chart → .mid in place, back up .bak.chart and remove the original
    .chart (the game switches to using the .mid with venue/animations).
    Returns the path of the generated .mid, or None on error."""
    base = os.path.splitext(path)[0]
    backup = base + ".bak.chart"
    mid_path = base + ".mid"
    try:
        mid = chart_to_midi(path)
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        mid.save(mid_path)
        if os.path.abspath(path) != os.path.abspath(mid_path):
            os.remove(path)
        log_fn(f"  ♪ .chart → .mid: {os.path.basename(mid_path)}\n", "info")
        return mid_path
    except Exception as e:
        import traceback
        log_fn(f"  ✗ chart {os.path.basename(path)}: {e}\n", "err")
        log_fn(traceback.format_exc(), "err")
        return None


_BG_EXTS = (".png", ".jpg", ".jpeg")


_BG_HIDDEN_STEM = "dc_hidden_bg"   # hidden-background filename stem (see below)


def _hide_backgrounds(folder: str, log_fn) -> int:
    """Rename in-game background images (background.png/jpg/jpeg) to
    dc_hidden_bg.<ext> so they don't render as the stage background. Reversible
    by revert_folder. Skips files already hidden. Returns the count hidden.

    The previous scheme renamed to ``background.bak.<ext>``, but RB3/CH/YARG load
    backgrounds with a ``background*`` glob — so a name still STARTING with
    "background" kept rendering. The hidden name must therefore not contain the
    word "background" at all; revert maps ``dc_hidden_bg.<ext>`` (and the legacy
    ``background.bak.<ext>``) back to ``background.<ext>``."""
    hidden = 0
    for root, _, files in os.walk(folder):
        for f in files:
            name, ext = os.path.splitext(f)
            if name.lower() == "background" and ext.lower() in _BG_EXTS:
                src = os.path.join(root, f)
                dst = os.path.join(root, _BG_HIDDEN_STEM + ext)
                try:
                    if os.path.exists(dst):
                        continue                       # already hidden
                    os.rename(src, dst)
                    hidden += 1
                    log_fn(f"  ◇ background hidden: "
                           f"{os.path.relpath(src, folder)}\n", "info")
                except Exception as e:
                    log_fn(f"  ✗ {f}: {e}\n", "err")
    if hidden:
        log_fn(f"  ◇ {hidden} background image(s) hidden\n", "info")
    return hidden


def process_folder(
    folder: str,
    diffs_to_gen: list[str],
    do_expert_plus: bool,
    threshold_ms: float,
    log_fn,
    do_venue: bool = False,
    do_lipsync: bool = False,
    do_hide_bg: bool = False,
    do_talkies: bool = False,
    do_drum_anim: bool = True,
    cancel: object | None = None,
) -> None:
    # Hide in-game background images (background.png/jpg → .bak) — Venue sub-option.
    if do_hide_bg:
        _hide_backgrounds(folder, log_fn)

    # .chart: convert to .mid first (backup .bak.chart) and process as MIDI.
    charts = find_charts(folder)
    converted = []
    for cpath in charts:
        mp = _prepare_chart(cpath, log_fn)
        if mp:
            converted.append(os.path.abspath(mp))

    midis = find_midis(folder)
    if not midis:
        log_fn("⚠  No .mid found.\n", "warn")
        return
    log_fn(f"→ {len(midis)} file(s)\n", "info")
    errors = 0
    modified = 0
    skipped_total = 0
    venue_skipped_total = 0
    groove_fails: list[str] = []   # "song: PART DIFF groove X% (< Y%)"
    error_log: list[str] = []      # "song: <exception + traceback>"
    conv_set = set(converted)
    for path in midis:
        if cancel is not None and cancel.is_set():
            log_fn("\n  ⚡ Cancelled by user.\n", "warn")
            break
        base = os.path.splitext(path)[0]
        backup = base + ".bak.mid"
        from_chart = os.path.abspath(path) in conv_set
        name = os.path.relpath(path, folder)
        try:
            # A .mid coming from a .chart already has a .bak.chart backup → don't duplicate .bak.mid.
            if not from_chart and not os.path.exists(backup):
                shutil.copy2(path, backup)
            s = process_midi(path, path, diffs_to_gen, do_expert_plus,
                             threshold_ms, do_venue, None, do_lipsync, do_talkies,
                             do_drum_anim=do_drum_anim)
            modified += 1
            skipped_total += len(s.get("diffs_skipped", []))
            if s.get("venue_skipped"):
                venue_skipped_total += 1
            for w in s.get("groove_warnings", []):
                groove_fails.append(f"{name}: {w}")
            extra = f"  [2x: {s['converted_2x']}]" if do_expert_plus and s["total_kicks"] else ""
            log_fn(f"  ✓ {name}{extra}\n", "ok")
            if s.get("tracks_renamed"):
                ren = ", ".join(f"{o or '?'}→{n}" for o, n in s["tracks_renamed"])
                log_fn(f"    ◇ normalized track names: {ren}\n", "info")
            if s.get("sp_remapped"):
                log_fn(f"    ◇ FoF star-power: remapped {s['sp_remapped']} "
                       f"phrase(s) from note 103 → overdrive 116\n", "info")
            if do_venue and s.get("venue_events"):
                audio = " · audio✓" if s.get("audio_used") else ""
                extra = ""
                if s.get("audio_drums"):
                    extra += " +pseudo-drums"
                if s.get("audio_anim_instr"):
                    extra += f" +anim({','.join(s['audio_anim_instr'])})"
                if s.get("audio_lyrics"):
                    extra += " +lyrics"
                log_fn(f"    ◇ venue: {s['venue_events']} events "
                       f"(theme {s.get('venue_theme')})"
                       f" · animations: {s.get('anim_events', 0)}{audio}{extra}\n", "info")
            elif do_venue and s.get("venue_skipped"):
                log_fn(f"    ↷ venue: skipped (already authored)"
                       f" · animations: {s.get('anim_events', 0)}\n", "info")
            if s.get("vocals_charted"):
                ph = s.get("vocal_phrases_gen")
                ph_txt = f" + {ph} phrases" if ph else ""
                tr = s.get("vocals_trimmed")
                tr_txt = f" · {tr} trimmed by audio" if tr else ""
                log_fn(f"    ◇ talkies: {s['vocals_charted']} charted vocals"
                       f"{ph_txt}{tr_txt}\n", "info")
            if s.get("lipsync_events"):
                log_fn(f"    ◇ lipsync: {s['lipsync_events']} viseme keyframes\n", "info")
            for sk in s.get("diffs_skipped", []):
                log_fn(f"    ↷ skipped {sk} (already charted)\n", "info")
            # Groove-check warnings are recorded only in the session log file,
            # not shown in the GUI log.
        except Exception as e:
            errors += 1
            import traceback
            tb = traceback.format_exc()
            error_log.append(f"{name}: {e}\n{tb}")
            log_fn(f"  ✗ {name}: {e}\n", "err")
            log_fn(tb, "err")

    # ── Summary ──────────────────────────────────────────────────────────────
    log_fn("\n── Done ──\n", "info")
    log_fn(f"  modified: {modified}   skipped diffs: {skipped_total}"
           f"   skipped venues: {venue_skipped_total}"
           f"   errors: {errors}\n", "info")

    # Session log: write errors + groove-check failures to a file in the folder.
    if groove_fails or error_log:
        log_path = _write_session_log(folder, modified, skipped_total,
                                      venue_skipped_total, groove_fails, error_log)
        if log_path:
            log_fn(f"  log: {os.path.basename(log_path)}\n", "info")


def _write_session_log(folder: str, modified: int, skipped_total: int,
                       venue_skipped_total: int, groove_fails: list[str],
                       error_log: list[str]) -> str | None:
    """Write a per-session log (errors + groove-check failures) to the folder.
    Returns the path written, or None on failure."""
    import datetime
    ts = datetime.datetime.now()
    fname = f"downcharter_log_{ts:%Y%m%d-%H%M%S}.txt"
    path = os.path.join(folder, fname)
    lines = [
        f"Downcharter+ session log — {ts:%Y-%m-%d %H:%M:%S}",
        f"Folder: {folder}",
        f"Modified: {modified}   Skipped diffs: {skipped_total}"
        f"   Skipped venues: {venue_skipped_total}"
        f"   Groove fails: {len(groove_fails)}   Errors: {len(error_log)}",
        "",
    ]
    if groove_fails:
        lines.append("── Groove-check failures ──")
        lines += [f"  {g}" for g in groove_fails]
        lines.append("")
    if error_log:
        lines.append("── Errors ──")
        for e in error_log:
            lines.append(e.rstrip("\n"))
            lines.append("")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path
    except Exception:
        return None


def revert_folder(folder: str, log_fn,
                  cancel: object | None = None) -> None:
    reverted = 0
    for root, _, files in os.walk(folder):
        for f in files:
            name, ext = os.path.splitext(f)
            if (name.lower() in (_BG_HIDDEN_STEM, "background.bak")
                    and ext.lower() in _BG_EXTS):
                # Restore a hidden in-game background image (new dc_hidden_bg.*
                # scheme + legacy background.bak.* still recognised).
                backup = os.path.join(root, f)
                original = os.path.join(root, "background" + ext)
                try:
                    if os.path.exists(original):
                        os.remove(original)
                    os.rename(backup, original)
                    reverted += 1
                    log_fn(f"  ↩ {os.path.relpath(original, folder)}\n", "ok")
                except Exception as e:
                    log_fn(f"  ✗ {f}: {e}\n", "err")
            elif f.endswith(".bak.mid"):
                backup = os.path.join(root, f)
                original = backup[:-8] + ".mid"
                try:
                    if os.path.exists(original):
                        os.remove(original)
                    os.rename(backup, original)
                    reverted += 1
                    log_fn(f"  ↩ {os.path.relpath(original, folder)}\n", "ok")
                except Exception as e:
                    log_fn(f"  ✗ {f}: {e}\n", "err")
            elif f.endswith(".bak.chart"):
                # Original was a .chart: restore the .chart and remove the generated .mid.
                backup = os.path.join(root, f)
                base = backup[:-len(".bak.chart")]
                original = base + ".chart"
                gen_mid = base + ".mid"
                try:
                    if os.path.exists(gen_mid):
                        os.remove(gen_mid)
                    if os.path.exists(original):
                        os.remove(original)
                    os.rename(backup, original)
                    reverted += 1
                    log_fn(f"  ↩ {os.path.relpath(original, folder)}\n", "ok")
                except Exception as e:
                    log_fn(f"  ✗ {f}: {e}\n", "err")
    if reverted == 0:
        log_fn("⚠  No backup found.\n", "warn")
    else:
        log_fn(f"\n✓ {reverted} file(s) reverted.\n", "ok")

"""ps3build.py — assemble the native RB3 PS3 outputs (Phase B).

Goal of the whole effort: stop depending on Onyx to compile our LIPSYNC into a
.milo and pack the song. Onyx re-packed stale milos (two packs built around our
Phase-2 lipsync change came out byte-identical), so our lipsync wasn't reaching
the game. By building the milo ourselves (downcharter/milo.py) we guarantee the
lipsync we generate is in the file the game loads.

The .milo is built ONCE, here, at conversion time. Processing only authors the
audio-guided LIPSYNC1 track inside notes.mid (downcharter/processor `_apply_lipsync`);
it does NOT write any milo. When a folder is converted into a PS3 pack,
`build_ps3_song` reconstructs the same audio-guided syllable spans the LIPSYNC1
track was authored from — straight off the charted PART VOCALS talky tubes (whose
ends are already audio-trimmed) plus their lyrics, re-deriving the per-syllable
loudness gain from the vocal stem — and feeds them to `milo.build_milo_from_spans`
(the validated 30 fps path). One milo, made when the pack is assembled, guaranteed
to carry our lipsync.

The Xbox-360 CON/STFS packer is a later follow-up (downcharter/stfs.py) reusing the
same milo + dta + mogg.
"""
from __future__ import annotations
import os
import re
import shutil

import mido

from . import milo as _milo
from . import convert as _convert
from . import validate as _validate
from . import mogg as _mogg
from . import art as _art
from .midi_utils import build_tempo_map, tick_to_ms, to_abs, rescale_midi_tpb

# PART VOCALS talky pitch authored by processor._chart_vocals_from_lyrics.
_VOCAL_TALKY_PITCH = 50


def _noop_log(msg, tag=None):
    pass


# ── song.ini → songs.dta generation (YARG/CH source has no dta) ────────────────
# Clone Hero / YARG genre strings → the RB3 genre SYMBOL the game knows. RB3 only
# renders a fixed set of genre symbols; anything unknown falls back to "rock".
_GENRE_MAP = {
    "metal": "metal", "heavy metal": "metal", "metalcore": "metal",
    "death metal": "metal", "black metal": "metal", "thrash": "metal",
    "thrash metal": "metal", "nu metal": "metal", "nu-metal": "metal",
    "rock": "rock", "hard rock": "rock", "classic rock": "rock",
    "arena rock": "rock", "garage rock": "rock", "southern rock": "rock",
    "alternative": "alternative", "alt": "alternative", "alt rock": "alternative",
    "grunge": "grunge", "punk": "punk", "punk rock": "punk", "pop punk": "punk",
    "hardcore": "punk", "post-hardcore": "punk",
    "pop": "poprock", "pop rock": "poprock", "pop/rock": "poprock",
    "indie": "indierock", "indie rock": "indierock", "indierock": "indierock",
    "electronic": "popdanceelectronic", "electronica": "popdanceelectronic",
    "edm": "popdanceelectronic", "dance": "popdanceelectronic",
    "techno": "popdanceelectronic", "house": "popdanceelectronic",
    "synthpop": "popdanceelectronic", "synth-pop": "popdanceelectronic",
    "hip hop": "urban", "hip-hop": "urban", "hiphop": "urban", "rap": "urban",
    "r&b": "rbsoulfunk", "rnb": "rbsoulfunk", "soul": "rbsoulfunk",
    "funk": "rbsoulfunk", "classical": "classical", "orchestral": "classical",
    "jazz": "jazz", "blues": "blues", "country": "country", "folk": "country",
    "bluegrass": "country", "emo": "emo", "screamo": "emo",
    "prog": "prog", "progressive": "prog", "progressive rock": "prog",
    "progressive metal": "prog", "fusion": "fusion", "reggae": "reggaeska",
    "ska": "reggaeska", "latin": "latin", "world": "world",
    "new wave": "newwave", "novelty": "novelty", "soundtrack": "other",
    "video game": "other", "vgm": "other", "other": "other",
}

# RB3 valid genre symbols (so an already-RB-style genre in the ini passes through).
_RB_GENRES = {
    "alternative", "blues", "classical", "classicrock", "country", "emo",
    "fusion", "glam", "grunge", "indierock", "inspirational", "jazz", "jrock",
    "latin", "metal", "new", "newwave", "novelty", "numetal", "other", "popdanceelectronic",
    "poprock", "prog", "punk", "rbsoulfunk", "reggaeska", "rock", "southernrock",
    "world", "urban",
}


def _parse_song_ini(path: str) -> dict:
    """Parse a CH/YARG song.ini into a flat dict (lowercased keys).

    song.ini is a simple key=value INI: a leading ``[Song]`` (or ``[song]``)
    section header, then ``key = value`` lines. We keep it flat (one logical
    section) and lowercase every key, since the spec uses a single section."""
    meta: dict = {}
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("[") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                meta[k.strip().lower()] = v.strip()
    except Exception:
        pass
    return meta


def _dta_str(s: str) -> str:
    """Make `s` safe to embed inside a DTA double-quoted string literal.

    The Harmonix DTA format (per maxton/DtxCS) has NO escape sequences: a raw
    double quote terminates the string and a raw newline is taken literally. So
    any title/artist/album text must have its quotes neutralised (→ typographic
    quote) and newlines/control chars flattened to spaces, or the whole songs.dta
    fails to parse."""
    s = (s or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = s.replace('"', "”")          # " → ” (DTA can't escape a literal quote)
    s = "".join(ch for ch in s if ch >= " ")  # drop remaining control chars
    return re.sub(r"\s+", " ", s).strip()


def _ini_int(meta: dict, *keys) -> int | None:
    """First parseable integer among `keys` in `meta` (handles '2,003'-style)."""
    for k in keys:
        v = meta.get(k)
        if v is None or str(v).strip() == "":
            continue
        try:
            return int(float(re.sub(r"[^0-9.\-]", "", str(v)) or "x"))
        except ValueError:
            continue
    return None


def _sanitize_shortname(meta: dict, fallback: str, suffix: str) -> str:
    raw = (meta.get("artist", "") + meta.get("name", "")) or fallback
    short = re.sub(r"[^a-z0-9]", "", raw.lower()) or "song"
    # `suffix` is "" for a plain song or "2x" for a genuine double-kick variant,
    # so a real 2x build stays a distinct install while the normal one is unsuffixed.
    return f"{short[:28]}{suffix}"


def _song_id(shortname: str) -> int:
    """Stable positive 32-bit-ish numeric id from the shortname."""
    h = 0
    for c in shortname:
        h = (h * 131 + ord(c)) & 0x7FFFFFFF
    return 1000000000 + (h % 900000000)


# RB3 rank-tier thresholds (the Magma/C3 boundaries the game uses to turn a raw
# `rank` number into the number of difficulty "dots" it shows). Each list is the
# minimum rank for tiers 1..7, so a song.ini `diff_*` INTENSITY (0..6, where the
# Intensity page maps intensity n → difficulty tier n+1) indexes straight into it.
# These are PER-INSTRUMENT — the same intensity yields a different rank number per
# instrument, which is exactly why one shared table mis-displayed difficulties.
# Index 0 is 1 (not 0): in a DTA, rank 0 means "part absent", so the lowest
# *playable* tier must be ≥1.
_RANK_TIERS = {
    "drum":      (1, 124, 151, 178, 242, 345, 448),
    "guitar":    (1, 139, 176, 221, 267, 333, 409),
    "bass":      (1, 135, 181, 228, 293, 364, 436),
    "vocals":    (1, 132, 175, 218, 279, 353, 427),
    "keys":      (1, 153, 211, 269, 327, 385, 443),
    "real_keys": (1, 153, 211, 269, 327, 385, 443),
    "real_guitar": (1, 150, 205, 264, 323, 382, 442),
    "real_bass":   (1, 150, 208, 267, 325, 384, 442),
    "band":      (1, 163, 215, 243, 267, 292, 345),
}


def _intensity_to_rank(intensity, instrument: str, charted: bool = False) -> int:
    """song.ini `diff_*` intensity (0..6) → RB3 rank number for `instrument`, via
    that instrument's tier thresholds.

    A rank of 0 in a DTA means "part absent / not playable". So:
      * `charted=True` (the part IS in the chart, e.g. our generated unpitched
        vocals): a missing/unparseable/-1 intensity falls back to intensity 0 →
        the LOWEST playable tier (rank tiers[0] = 1). This is what Onyx does — an
        undocumented-but-charted part stays playable (e.g. Elegy's vocals → 1),
        never rank 0 (which would hide the part and can make YARG reject the song).
      * `charted=False`: a missing intensity stays rank 0 (genuinely absent)."""
    tiers = _RANK_TIERS.get(instrument, _RANK_TIERS["band"])
    try:
        i = int(intensity)
    except (TypeError, ValueError):
        i = None
    if i is None or i < 0:
        # No documented difficulty: lowest playable tier if charted, else absent.
        return tiers[0] if charted else 0
    return tiers[min(i, 6)]


def _audio_guided_spans(mid, folder: str) -> list:
    """Reconstruct the audio-guided syllable spans [(start_s, end_s, text, gain)]
    that the LIPSYNC1 track was authored from, so the milo built here matches the
    track exactly — no re-doing geometry.

    The Process tab already charted PART VOCALS as talky tubes (note 50) whose
    note ENDS were trimmed against the vocal stem, and tagged each lyric '#'. We
    read those tubes ONLY off the PART VOCALS track (not a merged view — drum
    animation notes also live in 24-51), pair each with its lyric, convert ticks
    → seconds via the chart's tempo map, and re-derive the per-syllable loudness
    gain from the vocal stem in `folder`. These are the same spans the LIPSYNC1
    keyframes came from, so `milo.build_milo_from_spans` reproduces our lipsync."""
    idx = next((i for i, t in enumerate(mid.tracks)
                if (t.name or "").strip().upper() == "PART VOCALS"), None)
    if idx is None:
        return []
    tempo_map = build_tempo_map(mid)
    tpb = mid.ticks_per_beat
    abs_evts = to_abs(mid.tracks[idx])

    # Collect vocal gems across the WHOLE vocal range (36-84), not just our talky
    # pitch (50): a song with PRE-CHARTED, pitched vocals carries real melody notes
    # there and no note-50 tubes, so keying off 50 alone produced zero spans → no
    # lipsync milo for those songs. Onyx's autoLipsync likewise reads any charted
    # vocal gem. Vocals are monophonic, so a single open note at a time is correct;
    # phrase markers (105/106) and percussion (96/97) fall outside 36-84 and are
    # excluded automatically.
    gems: list[tuple[int, int]] = []          # (start_tick, end_tick) vocal gems
    open_t = None
    for e in abs_evts:
        m = e.msg
        n = getattr(m, "note", None)
        if n is None or not (36 <= n <= 84):
            continue
        is_off = m.type == "note_off" or (m.type == "note_on"
                                          and getattr(m, "velocity", 0) == 0)
        if m.type == "note_on" and getattr(m, "velocity", 0) > 0:
            open_t = e.abs_tick
        elif is_off and open_t is not None:
            gems.append((open_t, e.abs_tick))
            open_t = None
    if not gems:
        return []

    lyr = [(e.abs_tick, e.msg.text.strip()) for e in abs_evts
           if e.msg.type in ("lyrics", "lyric", "text")
           and getattr(e.msg, "text", "") and not e.msg.text.strip().startswith("[")]
    lyr.sort(key=lambda lt: lt[0])

    # Vocal-stem loudness for the per-syllable gain (no-op without audio).
    va = None
    try:
        from . import audio as _audio
        if _audio.available():
            stems = _audio.find_vocal_stems(folder)
            if stems:
                va = _audio.voice_activity(stems)
    except Exception:
        va = None

    tol = max(1, tpb // 4)
    spans = []
    for s_t, e_t in gems:
        best = None
        for lt, tx in lyr:
            if abs(lt - s_t) <= tol and (best is None or abs(lt - s_t) < abs(best[0] - s_t)):
                best = (lt, tx)
        # strip the talky markers ('#'/'^') the chart carries; keep '-'/'=' (the
        # G2P syllable-grouping the milo path expects).
        text = best[1].rstrip("#^ ").strip() if best else ""
        if not text or text in ("+", "*", "%"):
            continue
        s_s = tick_to_ms(s_t, tempo_map, tpb) / 1000.0
        e_s = tick_to_ms(e_t, tempo_map, tpb) / 1000.0
        gain = 1.0
        if va is not None:
            try:
                gain = _audio.syllable_gain(va, s_s, e_s)
            except Exception:
                gain = 1.0
        spans.append((s_s, e_s, text, gain))
    return spans


def _charted_instruments(mid) -> set:
    """Which RB instruments are actually charted in `mid` (by PART track + gems).
    Returns a subset of {drum, bass, guitar, keys, vocals}. An instrument audio
    stem with no chart (e.g. bass audio but no PART BASS gems) must NOT appear as
    a playable track in the dta — it just becomes backing audio."""
    name_map = [("DRUM", "drum"), ("BASS", "bass"), ("GUITAR", "guitar"),
                ("KEYS", "keys"), ("VOCAL", "vocals")]
    charted = set()
    for tr in mid.tracks:
        nm = (tr.name or "").upper()
        for key, inst in name_map:
            if key not in nm:
                continue
            if inst == "vocals":
                if any(m.type == "lyrics" for m in tr) or \
                   any(m.type == "note_on" and m.velocity > 0 for m in tr):
                    charted.add("vocals")
            else:
                # A real chart has expert gems in the 96..100 lane.
                if any(m.type == "note_on" and m.velocity > 0 and 96 <= m.note <= 100
                       for m in tr):
                    charted.add(inst)
            break
    return charted


def _build_dta(meta: dict, shortname: str, layout, is_2x: bool,
               charted: set | None = None, has_art: bool = False) -> tuple[str, str]:
    """Generate a Rock Band 3 songs.dta from song.ini metadata + the mogg channel
    layout. `layout` = [(track_name, [ch...])] from mogg.build_mogg_from_stems.
    `charted` limits which instruments are exposed as playable (ranked) parts;
    audio stems for un-charted instruments stay as backing channels.
    `is_2x` True only for a genuine double-kick variant (drives the label + flag).

    Returns (dta_text, codec) — write the text with `codec` so the file's bytes
    match its declared (encoding ...) node (Latin1 normally, UTF-8 if needed)."""
    # Only a genuine 2x variant is tagged in the display name; the plain (1x /
    # intact) song keeps its bare title — no "(1x Bass Pedal)" suffix.
    name_suffix = " (2x Bass Pedal)" if is_2x else ""
    title = _dta_str(meta.get("name") or shortname)
    artist = _dta_str(meta.get("album_artist") or meta.get("artist") or "Unknown")
    album = _dta_str(meta.get("album") or "")
    charter = _dta_str(meta.get("charter") or meta.get("frets") or "")
    # genre: map the CH/YARG string → an RB3 genre symbol; if it's already a valid
    # RB symbol, keep it; otherwise fall back to rock.
    g_raw = (meta.get("genre") or "").strip().lower()
    g_key = re.sub(r"[^a-z0-9]", "", g_raw)
    genre = _GENRE_MAP.get(g_raw) or (g_key if g_key in _RB_GENRES else "rock")
    year = _ini_int(meta, "year") or 2020
    song_len = _ini_int(meta, "song_length") or 0
    # vocal_gender: female if the ini says so, else male (RB3 default).
    vgender = "female" if (meta.get("vocal_gender") or "").strip().lower() \
        .startswith("f") else "male"
    album_track = _ini_int(meta, "album_track", "track") or 1

    # Only instruments that are actually charted are exposed as playable tracks.
    # If `charted` is None (caller didn't detect), fall back to "all instruments
    # that have a stem" (legacy behaviour).
    inst_tracks = charted if charted is not None else {"drum", "bass", "guitar", "keys", "vocals"}
    total_ch = sum(len(idxs) for _, idxs in layout)
    cores = []
    for track, idxs in layout:
        for _ in idxs:
            cores.append("1" if track == "guitar" else "-1")
    # Pans: stereo pairs → -1 / 1; lone channel → 0.
    pans = []
    for track, idxs in layout:
        if len(idxs) == 2:
            pans += ["-1.0", "1.0"]
        elif len(idxs) == 1:
            pans += ["0.0"]
        else:
            for i in range(len(idxs)):
                pans.append("-1.0" if i % 2 == 0 else "1.0")

    track_lines = []
    for track, idxs in layout:
        if track in inst_tracks and idxs:
            chans = " ".join(str(i) for i in idxs)
            track_lines.append(f"            ({track} ({chans}))")
    tracks_block = "\n".join(track_lines)

    # A rank of 0 = instrument not playable. Only rank the charted instruments.
    # Each diff_* in the ini is an independent 0-6 intensity — never borrow one
    # instrument's tier for another.
    has_vox = "vocals" in inst_tracks
    rank_g = _intensity_to_rank(meta.get("diff_guitar"), "guitar", charted=True) if "guitar" in inst_tracks else 0
    rank_b = _intensity_to_rank(meta.get("diff_bass"), "bass", charted=True) if "bass" in inst_tracks else 0
    rank_d = _intensity_to_rank(meta.get("diff_drums"), "drum", charted=True) if "drum" in inst_tracks else 0
    rank_k = _intensity_to_rank(meta.get("diff_keys"), "keys", charted=True) if "keys" in inst_tracks else 0
    rank_v = _intensity_to_rank(meta.get("diff_vocals"), "vocals", charted=True) if has_vox else 0
    # Band rank: honour an explicit diff_band intensity; otherwise derive it from
    # the AVERAGE INTENSITY of the charted instruments (not the average of their
    # ranks — those live on different per-instrument scales) and map through the
    # band tier table. _intensity_to_rank returns 0 only for an explicitly-absent
    # (-1) part, so filter those out of the average.
    _intensities = []
    for ini_key, inst in (("diff_guitar", "guitar"), ("diff_bass", "bass"),
                          ("diff_drums", "drum"), ("diff_keys", "keys"),
                          ("diff_vocals", "vocals")):
        if inst not in inst_tracks and not (inst == "vocals" and has_vox):
            continue
        iv = _ini_int(meta, ini_key)
        if iv is not None and iv >= 0:
            _intensities.append(iv)
    if meta.get("diff_band") not in (None, ""):
        rank_band = _intensity_to_rank(meta.get("diff_band"), "band")
    elif _intensities:
        rank_band = _intensity_to_rank(round(sum(_intensities) / len(_intensities)), "band")
    else:
        rank_band = _RANK_TIERS["band"][3]
    has_keys = rank_k > 0

    sid = _song_id(shortname)
    pans_s = " ".join(pans)
    vols_s = " ".join("0.0" for _ in range(total_ch))
    cores_s = " ".join(cores)
    # preview: honour the ini's preview_start_time/preview_end_time (ms) when set;
    # otherwise start a 30 s clip a third of the way in.
    preview_a = _ini_int(meta, "preview_start_time")
    if preview_a is None or preview_a < 0:
        preview_a = min(30000, max(0, song_len // 3)) if song_len else 30000
    preview_b = _ini_int(meta, "preview_end_time")
    if preview_b is None or preview_b <= preview_a:
        preview_b = preview_a + 30000

    # rank block: keys line only when keys are actually charted.
    rank_lines = [f"      (drum {rank_d})", f"      (guitar {rank_g})",
                  f"      (bass {rank_b})", f"      (vocals {rank_v})"]
    if has_keys:
        rank_lines.append(f"      (keys {rank_k})")
        rank_lines.append(f"      (real_keys {rank_k})")
    rank_lines.append(f"      (band {rank_band})")
    rank_block = "\n".join(rank_lines)
    # Content rating. song.ini (YARG) uses 1=FF 2=SR 3=MC 4=NR 5=SC; RB3's dta
    # only has 1=FF 2=SR 3=Mature 4=No-rating. Map 1:1 for 1-4; YARG's 5 (Sensitive
    # Content) has no RB slot → treat as Supervision Recommended (2). Default 4.
    rating = _ini_int(meta, "rating")
    if rating == 5:
        rating = 2
    elif rating is None or not (1 <= rating <= 4):
        rating = 4
    # optional author/charter credit (RB3DX surfaces it; harmless on stock RB3).
    author_line = f'\n   (author "{charter}")' if charter else ""

    # Encoding: YARG/RB detect a .dta's encoding from its BOM, NOT this node — and
    # a BOM-less UTF-8 file is read as Latin1, so an accented title written UTF-8
    # would show as mojibake in YARG. Match Onyx: write Latin1 (and declare it)
    # whenever every text field fits Latin1 (the common case); fall back to UTF-8
    # only for genuinely non-Latin1 text (e.g. Cyrillic/CJK titles).
    try:
        "".join((title, artist, album, charter, genre)).encode("latin1")
        enc_token, codec = "latin1", "latin1"
    except UnicodeEncodeError:
        enc_token, codec = "utf8", "utf-8"

    dta = f"""({shortname}
   (name "{title}{name_suffix}")
   (artist "{artist}")
   (master TRUE)
   (song_id {sid})
   (song
      (name "songs/{shortname}/{shortname}")
      (tracks
         (
{tracks_block}
         )
      )
      (pans ({pans_s}))
      (vols ({vols_s}))
      (cores ({cores_s}))
      (vocal_parts {1 if has_vox else 0})
      (drum_solo (seqs (kick.cue snare.cue tom1.cue tom2.cue crash.cue)))
      (drum_freestyle (seqs (kick.cue snare.cue hat.cue ride.cue crash.cue)))
   )
   (bank "sfx/tambourine_bank.milo")
   (drum_bank "sfx/kit01_bank.milo")
   (anim_tempo kTempoMedium)
   (song_scroll_speed 2300)
   (preview {preview_a} {preview_b})
   (song_length {song_len})
   (rank
{rank_block}
   )
   (format 10)
   (version 30)
   (game_origin ugc_plus)
   (rating {rating})
   (genre {genre})
   (vocal_gender {vgender})
   (year_released {year})
   (album_art {'TRUE' if has_art else 'FALSE'})
   (album_name "{album}")
   (album_track_number {album_track}){author_line}
   (encoding {enc_token})
   ;2xBass={'1' if is_2x else '0'}
)
"""
    return dta, codec


# ── source-folder discovery ──────────────────────────────────────────────────
def _find_one(folder: str, predicate) -> str | None:
    """First file under `folder` (recursive) satisfying predicate(full_path)."""
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            p = os.path.join(root, f)
            if predicate(p):
                return p
    return None


def _find_source_mid(folder: str) -> str | None:
    low = lambda p: os.path.basename(p).lower()
    # Prefer notes.mid, then any plain .mid (never .bak.mid, never .mid.edat).
    nm = _find_one(folder, lambda p: low(p) == "notes.mid")
    if nm:
        return nm
    return _find_one(folder, lambda p: low(p).endswith(".mid")
                     and not low(p).endswith(".bak.mid"))


def _find_source_mogg(folder: str) -> str | None:
    return _find_one(folder, lambda p: p.lower().endswith(".mogg"))


def _find_source_dta(folder: str) -> str | None:
    return _find_one(folder, lambda p: os.path.basename(p).lower() == "songs.dta") \
        or _find_one(folder, lambda p: p.lower().endswith(".dta"))


def _find_source_milo(folder: str, mid_path: str) -> str | None:
    # Prefer the sidecar the Process tab wrote next to the mid.
    sidecar = os.path.splitext(mid_path)[0] + ".milo_ps3"
    if os.path.isfile(sidecar):
        return sidecar
    return _find_one(folder, lambda p: p.lower().endswith(".milo_ps3"))


def _find_source_art(folder: str) -> str | None:
    return _find_one(folder, lambda p: p.lower().endswith(".png_ps3"))


# ── songs.dta helpers ─────────────────────────────────────────────────────────
def _dta_shortname(dta_text: str) -> str | None:
    """The internal song shortname — the first quoted symbol in the DTA."""
    m = re.search(r"\(\s*'([A-Za-z0-9_]+)'", dta_text)
    return m.group(1) if m else None


def _patch_dta(dta_text: str, shortname: str, is_2x: bool) -> str:
    """Adjust a songs.dta for the chosen pedal mode: rename the song with the
    pedal suffix and set the 2xBass comment flag. Internal paths are rewritten
    to point at `shortname` (in case the source used a different id).
    `is_2x` True only for a genuine double-kick variant."""
    # Song display name: strip any existing "(.. Bass Pedal)" then, for a genuine
    # 2x variant only, append "(2x Bass Pedal)". The plain song keeps its bare title.
    def _rename(m):
        title = m.group(1)
        title = re.sub(r"\s*\((?:1x|2x)\s*Bass Pedal\)\s*$", "", title).strip()
        if is_2x:
            title = f"{title} (2x Bass Pedal)"
        return f'\'name\'\n      "{title}"'
    dta_text = re.sub(r"'name'\s*\n\s*\"([^\"]*)\"", _rename, dta_text, count=1)
    # 2xBass comment flag (informational; YARG/manager scanners read it).
    if re.search(r";2xBass=", dta_text):
        dta_text = re.sub(r";2xBass=\d", f";2xBass={'1' if is_2x else '0'}", dta_text)
    return dta_text


def _pkg_folder_name(shortname: str, dta_text: str, suffix: str) -> str:
    """A unique PS3 package folder name, e.g. ELEGY_2X. `suffix` is "" for a plain
    song (no pedal tag) or "2x" for a genuine double-kick variant. If the shortname
    already carries the tag (generated names do), don't append it twice."""
    base = shortname.upper()
    if not suffix:
        return base
    tag = suffix.upper()
    if base.endswith(tag):
        return base
    return f"{base}_{tag}"


def source_has_double_kicks(src_folder: str) -> bool:
    """True if the source chart carries Expert+ double-kicks (note-95 markers).

    Used by the 'both' orchestration: a song with no doubles has nothing for the
    2x variant to convert, so only the plain (1x) folder should be produced."""
    mid_path = _find_source_mid(src_folder)
    if not mid_path:
        return False
    try:
        return _convert.count_double_kicks(mido.MidiFile(mid_path)) > 0
    except Exception:
        return False


# ── full PS3 song folder ───────────────────────────────────────────────────────
def build_ps3_song(src_folder: str, mode: str, log_fn=None, art_size: int = 512,
                   out_base: str | None = None) -> str:
    """Assemble a native unencrypted RPCS3 PS3 song folder from `src_folder`.

    `src_folder` is expected to hold a Downcharter-processed song: a plain
    notes.mid (with our Expert+ note-95 markers + LIPSYNC1), the `.milo_ps3`
    sidecar the Process tab wrote next to it, a multichannel `.mogg`, a
    `songs.dta`, and optionally the album-art `*.png_ps3`.

    `mode` is "1x" or "2x" (the bass-pedal variant — see convert.apply_pedal_variant).

    Layout produced (next to the source folder):
        <PKG>/songs/songs.dta
        <PKG>/songs/<id>/<id>.mid          (plain, pedal-adjusted)
        <PKG>/songs/<id>/<id>.mogg         (copied verbatim)
        <PKG>/songs/<id>/gen/<id>.milo_ps3 (our milo)
        <PKG>/songs/<id>/gen/<id>_keep.png_ps3 (art, if found)

    Returns the package folder path. Raises on missing essentials.
    """
    log = log_fn or _noop_log
    if mode not in ("1x", "2x"):
        raise ValueError(f"mode must be '1x' or '2x', got {mode!r}")

    mid_path = _find_source_mid(src_folder)
    if not mid_path:
        raise FileNotFoundError("no plain .mid found in source folder")
    mogg_path = _find_source_mogg(src_folder)          # may be None (YARG/CH stems)
    dta_path = _find_source_dta(src_folder)            # may be None (YARG/CH)
    milo_path = _find_source_milo(src_folder, mid_path)
    art_path = _find_source_art(src_folder)
    ini_path = _find_one(src_folder, lambda p: os.path.basename(p).lower() == "song.ini")
    meta = _parse_song_ini(ini_path) if ini_path else {}

    # Load the source chart up front so naming can know whether this song actually
    # has double-kicks. A "2x" build only earns the 2x name/label when there are
    # note-95 markers to convert; otherwise it is byte-for-byte the plain 1x chart,
    # so it stays unsuffixed. The 1x mode is always the plain, unsuffixed song.
    try:
        src_mid = mido.MidiFile(mid_path)
    except (ValueError, IOError):
        # Phase Shift sysex carry an illegal 0xFF byte; strict parsing chokes.
        # clip=True clamps it (we strip the sysex entirely below anyway).
        src_mid = mido.MidiFile(mid_path, clip=True)
    # RB3 crashes on any time division other than 480 TPB. Our processed charts
    # are already 480, but a song packed without going through processing (or an
    # odd source .mid) may not be — force it here so every package is 480.
    rescale_midi_tpb(src_mid, 480)
    has_2x = _convert.count_double_kicks(src_mid) > 0
    name_2x = (mode == "2x" and has_2x)
    suffix = "2x" if name_2x else ""

    # Existing dta (Onyx-style source) gives us the shortname; otherwise we mint
    # one from song.ini (artist+title+suffix) so a real 2x installs beside the 1x.
    dta_text = ""
    if dta_path:
        with open(dta_path, "r", encoding="latin1") as f:
            dta_text = f.read()
    if dta_text and _dta_shortname(dta_text):
        shortname = _dta_shortname(dta_text)
    else:
        fallback = os.path.splitext(os.path.basename(mid_path))[0]
        shortname = _sanitize_shortname(meta, fallback, suffix)

    pkg = _pkg_folder_name(shortname, dta_text, suffix)
    # Where the package folder is written: the user-chosen output folder if one
    # was supplied (Convert tab), otherwise next to the source folder (default).
    base_dir = os.path.abspath(out_base) if out_base \
        else os.path.dirname(os.path.abspath(src_folder))
    out_root = os.path.join(base_dir, pkg)
    # Guard against the package folder colliding with the SOURCE folder. On a
    # case-insensitive filesystem (Windows/macOS) "Elegy" and "ELEGY" are the same
    # directory, so a bare-named source (no song.ini → shortname == folder name)
    # would have the package written straight into the source, mixing/overwriting
    # the user's chart + stems. Never write inside the source: suffix the package.
    if os.path.normcase(os.path.abspath(out_root)) == \
            os.path.normcase(os.path.abspath(src_folder)):
        out_root += "_PS3"
        log("  (package name matched the source folder — using "
            f"{os.path.basename(out_root)} to avoid overwriting it)\n", "warn")
    song_dir = os.path.join(out_root, "songs", shortname)
    gen_dir = os.path.join(song_dir, "gen")
    os.makedirs(gen_dir, exist_ok=True)

    log(f"  → {pkg}\n", "info")

    # 1) MIDI: pedal-variant transform → <id>.mid.edat (DECRYPTED / plain).
    #    The chart must live at <id>.mid.edat, but it is written as a plain,
    #    unencrypted MIDI — NOT wrapped in an NPD/EDAT container.
    #
    #    Why no NPD wrapper: both readers reject/avoid it. YARG's UnpackedRBPKG
    #    flags any .mid.edat whose first 3 bytes are "NPD" as encrypted
    #    (ScanResult.EdatMidiEncrypted) and drops the song; otherwise it loads the
    #    file verbatim as a MIDI. RB3DX nightly likewise reads a decrypted chart.
    #    So a plain MIDI named .mid.edat is the format both accept ("desde que
    #    esteja desincriptado"). The NPD debug-EDAT wrapper (downcharter/edat.py,
    #    added in 91839e5) made YARG see the NPD magic and reject the folder — it
    #    is intentionally no longer used on this path.
    # a) open strums → green gem (RB3 ignores open notes)
    src_mid, os_stats = _convert.convert_open_notes(src_mid)
    if os_stats["converted"]:
        log(f"    ◇ mid: {os_stats['converted']} open note(s) remapped to green\n", "info")
    # b) bass-pedal variant on the drums kick lane
    out_mid, ks = _convert.apply_pedal_variant(src_mid, mode)
    # b2) RB3 crash-safety: fix overlapping/stuck same-pitch notes and strip the
    #     Phase Shift sysex (open/tap markers) the YARG/CH path keeps. Without this
    #     a broken-chord/hung-sustain chart hangs RB3 in-game.
    out_mid, san = _convert.sanitize_for_rb(out_mid)
    if san["overlaps_fixed"] or san["sysex_removed"] or san["tap_removed"]:
        log(f"    ◇ mid: RB-safety — fixed {san['overlaps_fixed']} overlapping "
            f"note(s), removed {san['sysex_removed']} Phase Shift sysex, "
            f"{san['tap_removed']} tap marker(s)\n", "info")
    # b3) Onyx no-Magma fixups: remove note-less overdrive phrases (fixNotelessOD)
    #     and add missing drum [mix] events (drumsComplete).
    out_mid, fx = _convert.apply_rb_fixups(out_mid)
    if fx["noteless_od_removed"] or fx["drum_mix_added"]:
        log(f"    ◇ mid: removed {fx['noteless_od_removed']} empty overdrive "
            f"phrase(s), added {fx['drum_mix_added']} drum mix event(s)\n", "info")
    if fx["end_added"] or fx["beat_added"]:
        log(f"    ◇ mid: basicTiming — "
            f"{'added [end]; ' if fx['end_added'] else ''}"
            f"{('generated BEAT track (%d beats)' % fx['beat_added']) if fx['beat_added'] else 'BEAT present'}\n",
            "info")
    # b4) Lead-in pad (Onyx magmaPad): RB3 needs >=2 beats before the first gem or
    #     it can reject/hang. We can only pad when we BUILD the mogg from stems
    #     (so matching silence is prepended); a verbatim source mogg is left as-is.
    pad_seconds = 0.0
    if mogg_path is None:
        pad_ticks = _convert.lead_in_pad_ticks(out_mid, min_beats=2.0)
        if pad_ticks > 0:
            tmap = build_tempo_map(out_mid)
            init_us = tmap[0][1] if tmap else 500000
            pad_seconds = pad_ticks / out_mid.ticks_per_beat * (init_us / 1_000_000.0)
            out_mid = _convert.pad_start(out_mid, pad_ticks)
            log(f"    ◇ mid: padded {pad_ticks} tick(s) ({pad_seconds:.3f}s) of "
                f"lead-in before the first gem\n", "info")
    elif _convert.lead_in_pad_ticks(out_mid, min_beats=2.0) > 0:
        log("    ⚠ mid: short lead-in but source mogg is reused verbatim — "
            "cannot pad audio, left unpadded\n", "warn")
    # NOTE: drummer limb animations (PART DRUMS 24-51) are authored during MIDI
    # processing (processor → convert.generate_drum_animations), so the notes.mid
    # already carries them. We do NOT synthesise them here at conversion time.
    charted = _charted_instruments(out_mid)
    # c) crash-relevant MIDI sanity gate — only at pack time (PS3/Xbox), never in
    #    processing. Non-fatal: we log issues (errors + warnings) but still build,
    #    so the user can decide. Mirrors quality.groove_check's advisory stance.
    try:
        for level, msg in _validate.validate_rb_midi(out_mid):
            mark = "✗" if level == "error" else "⚠"
            log(f"    {mark} check: {msg}\n", level)
    except Exception as e:
        log(f"    ⚠ check: MIDI validation skipped ({e})\n", "warn")
    # song.ini rarely carries song_length; derive it from the chart itself.
    if not meta.get("song_length"):
        try:
            meta["song_length"] = str(int(out_mid.length * 1000))
        except Exception:
            pass
    out_mid.save(os.path.join(song_dir, f"{shortname}.mid.edat"))
    if mode == "2x":
        log(f"    ◇ mid: {ks['converted']} double-kick(s) forced to single lane\n", "info")
    else:
        log(f"    ◇ mid: {ks['removed']} double-kick(s) removed (1x playable)\n", "info")
    log(f"    ◇ charted: {', '.join(sorted(charted)) or 'none'}\n", "info")

    # 2) MOGG: reuse the source mogg verbatim if present, else build one from the
    #    separate YARG/CH stems. The built layout drives the dta channel lists.
    mogg_layout = None
    out_mogg = os.path.join(song_dir, f"{shortname}.mogg")
    if mogg_path:
        # Copy verbatim — but re-encode to 44.1 kHz first if the source isn't,
        # since RB3 crashes at song LOAD on any other mogg sample rate. Channel
        # count is preserved, so a pre-existing/patched dta stays valid.
        _mogg.ensure_mogg_44100(mogg_path, out_mogg, log)
    else:
        mogg_layout = _mogg.build_mogg_from_stems(src_folder, out_mogg, log,
                                                  pad_seconds=pad_seconds)

    # 3) MILO: build OUR lipsync milo here — this is the ONLY place a milo is made.
    #    Processing already authored the audio-guided LIPSYNC1 track; we reconstruct
    #    the very spans it came from (charted PART VOCALS talky tubes + lyrics +
    #    vocal-stem gain) and serialise the milo with the validated 30 fps path. A
    #    pre-existing source milo is used only as a fallback for instrumental charts.
    milo_out = os.path.join(gen_dir, f"{shortname}.milo_ps3")
    try:
        spans = _audio_guided_spans(out_mid, src_folder)
        if spans:
            with open(milo_out, "wb") as f:
                f.write(_milo.build_milo_from_spans(spans, out_mid.length))
            log(f"    ◇ milo: built from {len(spans)} audio-guided syllable(s)\n", "info")
        elif milo_path:
            shutil.copy2(milo_path, milo_out)
            log(f"    ◇ milo: no charted vocals — reused source milo\n", "info")
        else:
            log(f"    ⚠ milo: no charted vocals — skipped (no lipsync)\n", "warn")
    except Exception as e:
        log(f"    ⚠ milo: lipsync build failed ({e}) — skipped\n", "warn")

    # 4) Album art. Prefer a pre-converted .png_ps3 from the source; otherwise
    #    generate one natively from the cover (album.png/cover.jpg) — a 256×256
    #    DXT1 HMX texture — so YARG/CH sources no longer depend on Onyx's art.
    art_out = os.path.join(gen_dir, f"{shortname}_keep.png_ps3")
    has_art = False
    if art_path:
        shutil.copy2(art_path, art_out)
        has_art = True
        log(f"    ◇ art: copied (.png_ps3)\n", "info")
    else:
        cover = _art.find_cover(src_folder)
        if cover and _art.available():
            try:
                with open(art_out, "wb") as f:
                    f.write(_art.build_png_ps3(cover, art_size))
                has_art = True
                log(f"    ◇ art: generated from {os.path.basename(cover)} "
                    f"({art_size}×{art_size})\n", "info")
            except Exception as e:
                log(f"    ⚠ art: cover convert failed ({e}) — skipped\n", "warn")
        elif cover and not _art.available():
            log(f"    ⚠ art: cover found but Pillow/numpy missing — skipped\n", "warn")
        else:
            log(f"    ⚠ art: no cover image found — skipped\n", "warn")

    # 5) songs.dta: patch an existing one, else generate from song.ini + layout.
    out_dta_path = os.path.join(out_root, "songs", "songs.dta")
    if dta_text:
        out_dta = _patch_dta(dta_text, shortname, name_2x)
        with open(out_dta_path, "w", encoding="latin1") as f:
            f.write(out_dta)
        log(f"    ◇ dta: patched ({'2x' if name_2x else '1x'})\n", "info")
    elif mogg_layout is not None:
        out_dta, dta_codec = _build_dta(meta, shortname, mogg_layout, name_2x,
                                        charted, has_art=has_art)
        # Write with the codec _build_dta chose to match its (encoding ...) node:
        # Latin1 for normal titles (so YARG, which reads BOM-less files as Latin1,
        # gets them right), UTF-8 only for genuinely non-Latin1 text.
        with open(out_dta_path, "w", encoding=dta_codec) as f:
            f.write(out_dta)
        log(f"    ◇ dta: generated from song.ini ({'2x' if name_2x else '1x'})\n", "info")
    else:
        log(f"    ⚠ dta: no songs.dta and no built mogg layout — skipped\n", "warn")

    log(f"  ✓ {pkg}\n", "ok")
    return out_root

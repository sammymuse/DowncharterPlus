"""ps3build.py — assemble the native RB3 PS3 outputs (Phase B).

Goal of the whole effort: stop depending on Onyx to compile our LIPSYNC into a
.milo and pack the song. Onyx re-packed stale milos (two packs built around our
Phase-2 lipsync change came out byte-identical), so our lipsync wasn't reaching
the game. By building the milo ourselves (downcharter/milo.py) we guarantee the
lipsync we generate is in the file the game loads.

This module wires that milo into the file system. It is being rolled out in two
steps, smallest-blast-radius first:

  1. `write_milo_sidecar` (LIVE): during normal processing, write a
     `<song>.milo_ps3` next to the processed notes.mid, built from the same
     audio-guided syllable spans as the LIPSYNC1 track. A PS3 .milo_ps3 and an
     Xbox .milo_xbox have identical bodies, so the same bytes serve both — we
     also drop a `.milo_xbox` copy. This lets us A/B test the milo IN-GAME by
     swapping it into a known-good RPCS3 pack's `gen/` folder, BEFORE investing
     in from-scratch dta/mogg/folder generation. (Decided test-first: confirm
     the milo carries our lipsync in-game first.)

  2. `build_ps3_song` (TODO, pending step 1's in-game validation + the
     unencrypted .mid / .mogg-version questions): lay out the full unencrypted
     PS3 song folder — `<ID>/songs/<id>/<id>.mid` (plain), `<id>.mogg` (reuse the
     source mogg verbatim, per the decision to keep it as-is), `gen/<id>.milo_ps3`,
     and a generated `songs/songs.dta`. The Xbox-360 CON/STFS packer is a later
     follow-up (downcharter/stfs.py) reusing the same milo + dta + mogg.
"""
from __future__ import annotations
import os
import re
import shutil

import mido

from . import milo as _milo
from . import convert as _convert
from . import mogg as _mogg


def _noop_log(msg, tag=None):
    pass


# ── song.ini → songs.dta generation (YARG/CH source has no dta) ────────────────
_GENRE_MAP = {
    "metal": "metal", "heavy metal": "metal", "metalcore": "metal",
    "rock": "rock", "hard rock": "rock", "alternative": "alternative",
    "punk": "punk", "pop": "pop", "indie": "indierock", "indie rock": "indierock",
    "electronic": "electronic", "hip hop": "hiphop", "rap": "hiphop",
    "classical": "classical", "jazz": "jazz", "blues": "blues",
    "country": "country", "emo": "emo", "prog": "prog",
}


def _parse_song_ini(path: str) -> dict:
    """Parse a CH/YARG song.ini into a flat dict (lowercased keys)."""
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


def _sanitize_shortname(meta: dict, fallback: str, mode: str) -> str:
    raw = (meta.get("artist", "") + meta.get("name", "")) or fallback
    short = re.sub(r"[^a-z0-9]", "", raw.lower()) or "song"
    return f"{short[:28]}{mode}"   # mode suffix → 1x and 2x stay distinct installs


def _song_id(shortname: str) -> int:
    """Stable positive 32-bit-ish numeric id from the shortname."""
    h = 0
    for c in shortname:
        h = (h * 131 + ord(c)) & 0x7FFFFFFF
    return 1000000000 + (h % 900000000)


def _tier_to_rank(tier) -> int:
    table = {0: 1, 1: 130, 2: 200, 3: 270, 4: 350, 5: 450, 6: 600}
    try:
        return table.get(int(tier), 270)
    except (TypeError, ValueError):
        return 270


def _build_dta(meta: dict, shortname: str, layout, mode: str) -> str:
    """Generate a Rock Band 3 songs.dta from song.ini metadata + the mogg channel
    layout. `layout` = [(track_name, [ch...])] from mogg.build_mogg_from_stems."""
    label = "2x Bass Pedal" if mode == "2x" else "1x Bass Pedal"
    title = (meta.get("name") or shortname).strip()
    artist = (meta.get("album_artist") or meta.get("artist") or "Unknown").strip()
    album = (meta.get("album") or "").strip()
    genre = _GENRE_MAP.get((meta.get("genre") or "").strip().lower(), "rock")
    try:
        year = int(re.sub(r"[^0-9]", "", meta.get("year", "")) or 0) or 2020
    except ValueError:
        year = 2020
    song_len = 0
    try:
        song_len = int(float(meta.get("song_length") or 0))
    except (TypeError, ValueError):
        song_len = 0

    # Channel lists.
    inst_tracks = {"drum", "bass", "guitar", "keys", "vocals"}
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

    has_vox = any(t == "vocals" for t, _ in layout)
    rank_band = _tier_to_rank(meta.get("diff_band", meta.get("diff_guitar")))
    rank_g = _tier_to_rank(meta.get("diff_guitar"))
    rank_b = _tier_to_rank(meta.get("diff_bass", meta.get("diff_guitar")))
    rank_d = _tier_to_rank(meta.get("diff_drums"))
    rank_v = _tier_to_rank(meta.get("diff_vocals", 0)) if has_vox else 0

    sid = _song_id(shortname)
    pans_s = " ".join(pans)
    vols_s = " ".join("0.0" for _ in range(total_ch))
    cores_s = " ".join(cores)
    preview_a = min(30000, max(0, song_len // 3)) if song_len else 30000
    preview_b = preview_a + 30000

    dta = f"""({shortname}
   (name "{title} ({label})")
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
      (drum {rank_d})
      (guitar {rank_g})
      (bass {rank_b})
      (vocals {rank_v})
      (band {rank_band})
   )
   (format 10)
   (version 30)
   (game_origin ugc_plus)
   (rating 4)
   (genre {genre})
   (vocal_gender male)
   (year_released {year})
   (album_art TRUE)
   (album_name "{album}")
   (album_track_number 1)
   (encoding utf8)
   ;2xBass={'1' if mode == '2x' else '0'}
)
"""
    return dta


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


def _patch_dta(dta_text: str, shortname: str, mode: str) -> str:
    """Adjust a songs.dta for the chosen pedal mode: rename the song with the
    pedal suffix and set the 2xBass comment flag. Internal paths are rewritten
    to point at `shortname` (in case the source used a different id)."""
    label = "2x Bass Pedal" if mode == "2x" else "1x Bass Pedal"
    # Song display name: strip any existing "(.. Bass Pedal)" then append ours.
    def _rename(m):
        title = m.group(1)
        title = re.sub(r"\s*\((?:1x|2x)\s*Bass Pedal\)\s*$", "", title).strip()
        return f'\'name\'\n      "{title} ({label})"'
    dta_text = re.sub(r"'name'\s*\n\s*\"([^\"]*)\"", _rename, dta_text, count=1)
    # 2xBass comment flag (informational; YARG/manager scanners read it).
    if re.search(r";2xBass=", dta_text):
        dta_text = re.sub(r";2xBass=\d", f";2xBass={'1' if mode=='2x' else '0'}", dta_text)
    return dta_text


def _pkg_folder_name(shortname: str, dta_text: str, mode: str) -> str:
    """A unique PS3 package folder name, e.g. ELEGY_2X. If the shortname already
    carries the pedal mode (generated names do), don't append it twice."""
    base = shortname.upper()
    if base.endswith(mode.upper()):
        return base
    return f"{base}_{mode.upper()}"


# ── full PS3 song folder ───────────────────────────────────────────────────────
def build_ps3_song(src_folder: str, mode: str, log_fn=None) -> str:
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

    # Existing dta (Onyx-style source) gives us the shortname; otherwise we mint
    # one from song.ini (artist+title+mode) so 1x and 2x install side by side.
    dta_text = ""
    if dta_path:
        with open(dta_path, "r", encoding="latin1") as f:
            dta_text = f.read()
    if dta_text and _dta_shortname(dta_text):
        shortname = _dta_shortname(dta_text)
    else:
        fallback = os.path.splitext(os.path.basename(mid_path))[0]
        shortname = _sanitize_shortname(meta, fallback, mode)

    pkg = _pkg_folder_name(shortname, dta_text, mode)
    out_root = os.path.join(os.path.dirname(os.path.abspath(src_folder)), pkg)
    song_dir = os.path.join(out_root, "songs", shortname)
    gen_dir = os.path.join(song_dir, "gen")
    os.makedirs(gen_dir, exist_ok=True)

    log(f"  → {pkg}\n", "info")

    # 1) MIDI: pedal-variant transform → plain unencrypted <id>.mid
    src_mid = mido.MidiFile(mid_path)
    out_mid, ks = _convert.apply_pedal_variant(src_mid, mode)
    out_mid.save(os.path.join(song_dir, f"{shortname}.mid"))
    if mode == "2x":
        log(f"    ◇ mid: {ks['converted']} double-kick(s) forced to single lane\n", "info")
    else:
        log(f"    ◇ mid: {ks['removed']} double-kick(s) removed (1x playable)\n", "info")

    # 2) MOGG: reuse the source mogg verbatim if present, else build one from the
    #    separate YARG/CH stems. The built layout drives the dta channel lists.
    mogg_layout = None
    out_mogg = os.path.join(song_dir, f"{shortname}.mogg")
    if mogg_path:
        shutil.copy2(mogg_path, out_mogg)
        log(f"    ◇ mogg: copied ({os.path.basename(mogg_path)})\n", "info")
    else:
        mogg_layout = _mogg.build_mogg_from_stems(src_folder, out_mogg, log)

    # 3) MILO: our native lipsync milo
    if milo_path:
        shutil.copy2(milo_path, os.path.join(gen_dir, f"{shortname}.milo_ps3"))
        tag = "sidecar (our lipsync)" if milo_path.endswith(".milo_ps3") \
            and os.path.dirname(milo_path) == os.path.dirname(mid_path) else "reused"
        log(f"    ◇ milo: {tag}\n", "info")
    else:
        log(f"    ⚠ milo: none found — run the Process tab (talkies) first\n", "warn")

    # 4) Album art (optional)
    if art_path:
        shutil.copy2(art_path, os.path.join(gen_dir, f"{shortname}_keep.png_ps3"))
        log(f"    ◇ art: copied\n", "info")

    # 5) songs.dta: patch an existing one, else generate from song.ini + layout.
    out_dta_path = os.path.join(out_root, "songs", "songs.dta")
    if dta_text:
        out_dta = _patch_dta(dta_text, shortname, mode)
        with open(out_dta_path, "w", encoding="latin1") as f:
            f.write(out_dta)
        log(f"    ◇ dta: patched ({mode})\n", "info")
    elif mogg_layout is not None:
        out_dta = _build_dta(meta, shortname, mogg_layout, mode)
        with open(out_dta_path, "w", encoding="latin1") as f:
            f.write(out_dta)
        log(f"    ◇ dta: generated from song.ini ({mode})\n", "info")
    else:
        log(f"    ⚠ dta: no songs.dta and no built mogg layout — skipped\n", "warn")

    log(f"  ✓ {pkg}\n", "ok")
    return out_root


def write_milo_sidecar(dst_mid_path: str, spans, song_len_s: float,
                       lang: str = "en") -> list[str]:
    """Build the .milo from the lipsync spans and write it next to the MIDI.

    `spans` = [(start_s, end_s, text, gain)] (the same audio-guided syllable
    spans that drive the LIPSYNC1 track). Writes `<base>.milo_ps3` and an
    identical `<base>.milo_xbox` (the body is platform-independent; only the
    outer CON/STFS wrapper differs). Returns the paths written (empty if there's
    nothing to build). Never raises into the caller's pipeline."""
    if not spans or song_len_s <= 0:
        return []
    milo_bytes = _milo.build_milo_from_spans(spans, song_len_s, lang)
    base = os.path.splitext(dst_mid_path)[0]
    written: list[str] = []
    for ext in (".milo_ps3", ".milo_xbox"):
        path = base + ext
        with open(path, "wb") as f:
            f.write(milo_bytes)
        written.append(path)
    return written

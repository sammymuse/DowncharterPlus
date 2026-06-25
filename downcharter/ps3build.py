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


def _noop_log(msg, tag=None):
    pass


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
    """A unique PS3 package folder name, e.g. O123_ELEGY_2X."""
    base = shortname.upper()
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
    mogg_path = _find_source_mogg(src_folder)
    if not mogg_path:
        raise FileNotFoundError("no .mogg found in source folder")
    dta_path = _find_source_dta(src_folder)
    milo_path = _find_source_milo(src_folder, mid_path)
    art_path = _find_source_art(src_folder)

    dta_text = ""
    if dta_path:
        with open(dta_path, "r", encoding="latin1") as f:
            dta_text = f.read()
    shortname = (_dta_shortname(dta_text) if dta_text else None) \
        or re.sub(r"[^a-z0-9_]", "_", os.path.splitext(os.path.basename(mid_path))[0].lower())

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

    # 2) MOGG: copy verbatim (per the keep-as-is decision)
    shutil.copy2(mogg_path, os.path.join(song_dir, f"{shortname}.mogg"))
    log(f"    ◇ mogg: copied ({os.path.basename(mogg_path)})\n", "info")

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

    # 5) songs.dta (patched for the pedal mode)
    if dta_text:
        out_dta = _patch_dta(dta_text, shortname, mode)
        with open(os.path.join(out_root, "songs", "songs.dta"), "w",
                  encoding="latin1") as f:
            f.write(out_dta)
        log(f"    ◇ dta: written ({mode})\n", "info")
    else:
        log(f"    ⚠ dta: no songs.dta in source — skipped\n", "warn")

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

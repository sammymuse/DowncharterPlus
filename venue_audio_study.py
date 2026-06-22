"""Venue placement study: cross each official VENUE event with the song-relative
AUDIO context (loudness / flux / brightness percentile) at the same instant.

Goal: learn WHERE and WHEN each effect is used in the 20 ground-truth venues,
in audio terms, instead of guessing. Pairs midi_files/*/<name>.mid with the
matching mogg_files/<name>.mogg.
"""
import os, glob, struct, io, collections, statistics
import numpy as np
import mido

ROOT = "midis/Venue learn songs"
MIDI_DIR = os.path.join(ROOT, "midi_files")
MOGG_DIR = os.path.join(ROOT, "mogg_files")

HOP, WIN = 1024, 2048


def load_mono(path):
    import soundfile as sf
    raw = open(path, "rb").read()
    version = struct.unpack("<I", raw[:4])[0]
    if version != 0x0A:
        raise ValueError("encrypted mogg")
    offset = struct.unpack("<I", raw[4:8])[0]
    data, sr = sf.read(io.BytesIO(raw[offset:]), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def stft_mag(mono, sr, hop=HOP, win=WIN):
    n = len(mono)
    if n < win:
        return None
    w = np.hanning(win).astype(np.float32)
    nf = 1 + (n - win) // hop
    frames = np.empty((nf, win), dtype=np.float32)
    for i in range(nf):
        frames[i] = mono[i * hop:i * hop + win] * w
    spec = np.fft.rfft(frames, axis=1)
    return np.abs(spec).astype(np.float32)


def tempo_map(mid):
    # absolute (tick, tempo_us_per_beat) breakpoints from track 0/all
    events = []
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                events.append((t, msg.tempo))
    events.sort()
    if not events or events[0][0] != 0:
        events.insert(0, (0, 500000))
    # dedupe by tick keep last
    out = []
    for tk, tp in events:
        if out and out[-1][0] == tk:
            out[-1] = (tk, tp)
        else:
            out.append((tk, tp))
    return out


def tick_to_ms(tick, tmap, tpb):
    ms = 0.0
    for i, (tk, tp) in enumerate(tmap):
        nxt = tmap[i + 1][0] if i + 1 < len(tmap) else None
        if nxt is not None and tick > nxt:
            ms += (nxt - tk) * tp / 1000.0 / tpb
        else:
            ms += (tick - tk) * tp / 1000.0 / tpb
            break
    return ms


def categorize(text):
    t = text.strip("[]")
    if t.startswith("lighting"):
        inner = t[t.find("(") + 1:t.find(")")] if "(" in t else ""
        if "blackout" in inner:
            return ("light_blackout", inner)
        if "strobe" in inner:
            return ("light_strobe", inner)
        if inner in ("manual_warm", "loop_warm", "flare_fast", "flare_slow"):
            return ("light_warm", inner)
        if inner in ("manual_cool", "loop_cool", "searchlights"):
            return ("light_cool", inner)
        if inner in ("frenzy", "silhouettes", "silhouettes_spot", "dischord"):
            return ("light_intense", inner)
        return ("light_other", inner)
    if t.startswith("bonusfx"):
        return ("pyro", t)
    if t == "next" or t == "prev" or t == "first":
        return ("keyframe", t)
    if t.startswith("coop_") or t.startswith("directed_"):
        return ("camera", t)
    if t.endswith(".pp"):
        return ("postproc", t)
    return ("other", t)


def pct_of(arr_sorted, val):
    # percentile rank of val within sorted array
    import bisect
    i = bisect.bisect_right(arr_sorted, val)
    return 100.0 * i / max(1, len(arr_sorted))


def analyze_song(mid_path, mogg_path):
    mid = mido.MidiFile(mid_path)
    tpb = mid.ticks_per_beat
    tmap = tempo_map(mid)
    venue = None
    for tr in mid.tracks:
        if tr.name and tr.name.upper() == "VENUE":
            venue = tr
            break
    if venue is None:
        return None
    # collect events (tick, category, sub)
    evts = []
    t = 0
    for msg in venue:
        t += msg.time
        if msg.type == "text":
            cat, sub = categorize(msg.text)
            evts.append((t, cat, sub))

    # audio features
    mono, sr = load_mono(mogg_path)
    mag = stft_mag(mono, sr)
    if mag is None:
        return None
    stft_s = HOP / sr
    freqs = np.fft.rfftfreq(WIN, 1.0 / sr)
    rms = np.sqrt((mag ** 2).mean(axis=1) + 1e-9)
    flux = np.concatenate([[0], np.maximum(0, np.diff(mag, axis=0)).sum(axis=1)])
    centroid = (mag * freqs).sum(axis=1) / (mag.sum(axis=1) + 1e-9)
    # smoothed intensity for drop detection
    inten = rms / (rms.max() + 1e-9) + flux / (flux.max() + 1e-9)
    w = max(1, int(0.2 / stft_s))
    inten_s = np.convolve(inten, np.ones(w) / w, mode="same")

    rms_s = np.sort(rms)
    flux_s = np.sort(flux)
    cen_s = np.sort(centroid)

    def frame_at(ms):
        return int(round(ms / 1000.0 / stft_s))

    win_f = max(1, int(0.5 / stft_s))  # 0.5 s drop window

    rows = []
    for tk, cat, sub in evts:
        ms = tick_to_ms(tk, tmap, tpb)
        fi = frame_at(ms)
        if fi < 0 or fi >= len(rms):
            continue
        loud_p = pct_of(rms_s, rms[fi])
        flux_p = pct_of(flux_s, flux[fi])
        bright_p = pct_of(cen_s, centroid[fi])
        # local drop ratio: mean next window / mean prev window
        a = inten_s[max(0, fi - win_f):fi + 1].mean()
        b = inten_s[fi:fi + win_f + 1].mean()
        drop = b / (a + 1e-9)
        rows.append((cat, sub, loud_p, flux_p, bright_p, drop))
    return rows


def main():
    moggs = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(MOGG_DIR, "*.mogg"))}
    all_rows = []
    for d in sorted(glob.glob(os.path.join(MIDI_DIR, "*"))):
        mids = glob.glob(os.path.join(d, "*.mid"))
        if not mids:
            continue
        mid_path = mids[0]
        name = os.path.splitext(os.path.basename(mid_path))[0]
        mogg = moggs.get(name)
        if not mogg:
            print("NO MOGG for", name)
            continue
        try:
            rows = analyze_song(mid_path, mogg)
        except Exception as e:
            print("ERR", name, e)
            continue
        if rows:
            all_rows.extend(rows)
            print(f"ok {name[:40]:40s} {len(rows)} events")

    print("\n==== AUDIO CONTEXT BY CATEGORY (percentile rank within each song) ====")
    print(f"{'category':16s} {'n':>5s} {'loud_p':>8s} {'flux_p':>8s} {'bright_p':>9s} {'drop':>7s}")
    cats = collections.defaultdict(list)
    for r in all_rows:
        cats[r[0]].append(r)
    base_loud = statistics.median([r[2] for r in all_rows])
    for cat in sorted(cats):
        rs = cats[cat]
        med = lambda i: statistics.median([r[i] for r in rs])
        print(f"{cat:16s} {len(rs):5d} {med(2):8.1f} {med(3):8.1f} {med(4):9.1f} {med(5):7.2f}")
    print(f"\n(overall median loud_p across all events = {base_loud:.1f}; "
          f"random baseline would be ~50)")

    # warm vs cool brightness AND loudness contrast
    def med(rows, i):
        return statistics.median([r[i] for r in rows]) if rows else float("nan")
    warm = [r for r in all_rows if r[0] == "light_warm"]
    cool = [r for r in all_rows if r[0] == "light_cool"]
    print(f"\nWARM  bright_p={med(warm,4):.1f} loud_p={med(warm,2):.1f}  "
          f"COOL  bright_p={med(cool,4):.1f} loud_p={med(cool,2):.1f}  "
          f"(my feature #2 assumes dark->warm, bright->cool i.e. cool bright_p > warm)")

    # distribution: fraction of each key category in loud tertiles
    print("\n==== loudness-percentile distribution (where does each effect sit?) ====")
    print(f"{'category':16s} {'%bottom33':>10s} {'%mid':>6s} {'%top33':>8s}")
    for cat in ("pyro", "light_strobe", "light_blackout", "light_warm",
                "light_cool", "light_intense", "keyframe"):
        rs = [r for r in all_rows if r[0] == cat]
        if not rs:
            continue
        lo = sum(1 for r in rs if r[2] < 33.3) / len(rs) * 100
        mi = sum(1 for r in rs if 33.3 <= r[2] < 66.6) / len(rs) * 100
        hi = sum(1 for r in rs if r[2] >= 66.6) / len(rs) * 100
        print(f"{cat:16s} {lo:10.0f} {mi:6.0f} {hi:8.0f}")


if __name__ == "__main__":
    main()

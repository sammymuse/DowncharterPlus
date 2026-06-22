"""Animation mood-marker placement study: audio context (loudness percentile) of
each per-instrument mood marker across the 20 official venues. Tells us the energy
ordering and when idle vs idle_intense vs play vs intense vs mellow is used."""
import os, glob, collections, statistics
import numpy as np
import mido
from venue_audio_study import (load_mono, stft_mag, tempo_map, tick_to_ms,
                               pct_of, MIDI_DIR, MOGG_DIR, HOP, WIN)

MOODS = {"[intense]", "[play]", "[mellow]", "[idle]", "[idle_realtime]",
         "[idle_intense]", "[play_solo]"}


def analyze(mid_path, mogg_path):
    mid = mido.MidiFile(mid_path)
    tpb = mid.ticks_per_beat
    tmap = tempo_map(mid)
    events = []  # (tick, marker, inst)
    for tr in mid.tracks:
        nm = (tr.name or "").upper()
        if not nm.startswith("PART"):
            continue
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "text" and msg.text.strip() in MOODS:
                events.append((t, msg.text.strip(), nm))
    if not events:
        return None
    mono, sr = load_mono(mogg_path)
    mag = stft_mag(mono, sr)
    if mag is None:
        return None
    stft_s = HOP / sr
    rms = np.sqrt((mag ** 2).mean(axis=1) + 1e-9)
    rms_s = np.sort(rms)
    rows = []
    for tk, mk, inst in events:
        fi = int(round(tick_to_ms(tk, tmap, tpb) / 1000.0 / stft_s))
        if 0 <= fi < len(rms):
            rows.append((mk, pct_of(rms_s, rms[fi]), inst))
    return rows


def main():
    moggs = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(MOGG_DIR, "*.mogg"))}
    allrows = []
    # also count consecutive-repeat runs of the SAME marker per track (monotony)
    for d in sorted(glob.glob(os.path.join(MIDI_DIR, "*"))):
        mids = glob.glob(os.path.join(d, "*.mid"))
        if not mids:
            continue
        name = os.path.splitext(os.path.basename(mids[0]))[0]
        mogg = moggs.get(name)
        if not mogg:
            continue
        try:
            rows = analyze(mids[0], mogg)
        except Exception as e:
            print("ERR", name, e); continue
        if rows:
            allrows.extend(rows)

    print(f"total mood markers: {len(allrows)}\n")
    print(f"{'marker':18s} {'n':>5s} {'loud_p median':>14s}")
    fam = collections.defaultdict(list)
    for r in allrows:
        fam[r[0]].append(r[1])
    for mk in sorted(fam, key=lambda k: -statistics.median(fam[k])):
        print(f"{mk:18s} {len(fam[mk]):5d} {statistics.median(fam[mk]):14.1f}")

    # how often does idle_intense vs idle appear, by loud context
    ii = fam.get("[idle_intense]", [])
    il = fam.get("[idle]", [])
    if ii and il:
        print(f"\nidle_intense loud_p median={statistics.median(ii):.1f}  "
              f"idle loud_p median={statistics.median(il):.1f}  "
              f"(idle_intense should sit in louder audio)")


if __name__ == "__main__":
    main()

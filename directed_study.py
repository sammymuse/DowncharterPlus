"""Directed-cut variety study over the 20 official venues: how many DISTINCT directed
cuts per song and how dominant the top one is. Ground-truth for the anti-monotony
rule in build_camera (the official ones use ~14 distinct/song, top-cut ~20%)."""
import glob, os, collections, statistics
import mido

MIDI_DIR = "midis/Venue learn songs/midi_files"


def main():
    agg = collections.Counter()
    distincts, tops = [], []
    print(f'{"song":34s} {"#dir":>5s} {"distinct":>8s} {"top%":>5s}')
    for d in sorted(glob.glob(os.path.join(MIDI_DIR, "*"))):
        mids = glob.glob(os.path.join(d, "*.mid"))
        if not mids:
            continue
        ven = next((t for t in mido.MidiFile(mids[0]).tracks
                    if (t.name or "").upper() == "VENUE"), None)
        if not ven:
            continue
        c = collections.Counter()
        for msg in ven:
            if msg.type == "text":
                t = msg.text.strip("[]")
                if t.startswith("directed_"):
                    c[t] += 1
                    agg[t] += 1
        if not c:
            continue
        tot = sum(c.values())
        topn = c.most_common(1)[0][1]
        distincts.append(len(c))
        tops.append(100 * topn / tot)
        print(f'{os.path.basename(d)[:34]:34s} {tot:5d} {len(c):8d} {100*topn/tot:4.0f}%')
    print(f'\nMEDIAN distinct/song = {statistics.median(distincts):.0f}  '
          f'top-cut share = {statistics.median(tops):.0f}%')
    print("\nTOP directed cuts across the corpus:")
    for k, v in agg.most_common(15):
        print(f'  {v:4d}  {k}')


if __name__ == "__main__":
    main()

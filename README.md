<p align="center">
  <img src="assets/DC+_icon.png" alt="Downcharter+" width="120">
</p>

# Downcharter+

<p align="center">
  <a href="https://github.com/sammymuse/Downcharter/releases/latest"><img src="https://img.shields.io/github/v/release/sammymuse/Downcharter?label=release" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/sammymuse/Downcharter" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/platform-Windows-blue" alt="Platform: Windows">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
</p>

Chart reducer, venue generator and native Rock Band 3 packager for
[YARG](https://yarg.in/), [Rock Band 3](https://en.wikipedia.org/wiki/Rock_Band_3)
and [Clone Hero](https://clonehero.net/).

Most customs ship with only an Expert chart. Downcharter+ takes that `.mid` or
`.chart` and fills in the rest: the lower difficulties for every instrument, a
venue with camera and lights, character animations, talky vocals from the
lyrics, and audio-driven lipsync. It then packages the result natively for the
platform you play on — a PS3/RPCS3 folder, an Xbox 360 CON, or a YARG/Clone
Hero `.sng` — with no external converter needed.

What it covers:

- Difficulty reduction for guitar, bass and keys — Hard, Medium and Easy
- Drum reduction across all three difficulties, plus Expert+ (2× kick) detection
- Venue generation — camera cuts, lights, post-processing, pyro and animations
- Talkies charted from the lyrics, with note ends trimmed against the vocal stem
- Audio-driven lipsync (mouth + blink/brow/squint), packaged in a native `.milo`
  and validated frame-by-frame against official RB3 lipsync
- Vocal stem separation (MDX-NET) when the song has no isolated stems —
  GPU (DirectML) with CPU fallback
- Native packaging: PS3/RPCS3 song folder, Xbox 360 CON, YARG/CH `.sng`

The reductions try to stay close to what a human charter would do — they keep the
groove and the riff intact instead of quantizing everything to a grid, preserve
sustains and pitch contour, and keep the BEAT track running to the end so the
characters don't freeze mid-song.

---

## Guitar & Drum reductions

Difficulties are worked out from the beat grid rather than hard-coded for a given
BPM, and they cascade down from Expert (`Expert → Hard → Medium → Easy`) so each
level stays consistent with the one above it.

**Guitar / Bass / Keys** — pitch contour preserved:

![Guitar reduction](docs/images/reductions_guitar.png)

| Difficulty | Rules |
|---|---|
| **Hard** | from Expert · real HOPOs · no green+orange or 3-note chords · sustain gap 1/16 |
| **Medium** | min spacing 1/4 · gap-fill + beat-snap · all strum · sustain gap 1/4 |
| **Easy** | min spacing dotted-1/4 (1.5 beat) · no chords · all strum · sustain gap 1/4 |

**Drums** — groove-preserving:

![Drum reduction](docs/images/reductions_drums.png)

- Kick / snare / pad thinning on an adaptive grid, fills collapsed per difficulty
- Doubles preserved · 3+ fast kicks → alternating = Expert+
- A **groove-check** quality guard flags any reduction that loses the feel (logged to a session file)

For Rock Band, the Expert chart also gets a hand-position map (force-HOPO/strum
markers derived from 100 official charts) and full drum limb animations with
realistic sticking, including strict alternation through fills.

---

## Venue generation

![Venue generation](docs/images/venue.jpg)

▶️ **[Watch the full demo](https://youtu.be/stHylYlXqAs)**

Generates camera cuts, lights, post-processing, spotlights, pyro and character
animations, with a theme chosen to fit the song. Everything is calibrated
against 100 official RB3 charts: a director pass makes song-level decisions
once (repeated sections reuse their look, section boundaries share anchors, the
last chorus gets the official 1.5× lighting climax), and the RNG is seeded from
the song's content so the same song always gets the same venue.

The camera won't point at an instrument that isn't playing — and the vocalist's
cuts and spotlight only fire on **real** sung vocals (chart/lyrics), never on an
audio-analysis guess. If a song already has a hand-authored venue, it's left
alone.

If audio is present (`.ogg`, `.opus`, `.wav`, `.flac`, `.mp3`, `.mogg`), it's
used to read the actual loudness of each section — a loud chorus vs. a quiet
verse — which feeds the cuts and lights.

---

## Talkies & lipsync

![Talkies generation](docs/images/talkies.png)

Charts unpitched (talky) vocals in `PART VOCALS` straight from the lyrics, so the
characters sing and lip-sync even on songs that only have lyrics and no vocal
chart. These are real vocal tubes the engine drives from, not just a lipsync
track.

Without pitch info, the tricky part is knowing how long each note should last. If
an isolated vocal stem is available, Downcharter+ reads the voice's RMS envelope
and ends each note where the singer actually stops. With no separate stem it
extracts the vocal channels from an unencrypted `.mogg`, or separates one from
the mix with MDX-NET (GPU via DirectML, CPU fallback). A real stem is worth
having — it drives both the sustain trimming and the per-syllable mouth weight.

The lipsync itself is phoneme-based (CMUdict + rule G2P → visemes) and follows
the official articulation model: the mouth opens ~0.12 s before each tube and
closes ~0.12 s after it, vowels hold for the tube's whole length, melisma (`+`)
tubes keep it open, and back-to-back syllables connect legato instead of
punching shut. Blinks, brow and squint keyframes ride along. Compared
frame-by-frame against 15 official RB3 milos, the generated lipsync agrees on
mouth state 88% of the time (96% open-when-they're-open), with episode lengths
matching the official envelope.

---

## Rock Band 3 packaging (native)

The Convert tab turns a processed song folder into a ready-to-install package —
no Magma, no Onyx:

- **PS3 / RPCS3 folder** — `songs/<id>/<id>.mid.edat` (plain MIDI), `.mogg`,
  `gen/<id>.milo_ps3` (our lipsync), `.png_ps3` album art (native DXT1 encoder),
  and a generated `songs.dta` from `song.ini`.
- **Xbox 360 CON** — the same payload in a single STFS container (hash tables,
  volume descriptor), signable for retail hardware; YARG loads it unsigned. The
  inner `.mid` is byte-identical to the PS3 one — both builders share the same
  conversion chain.
- **YARG / Clone Hero `.sng`** — the modern single-file format, with the
  original chart data preserved (Phase Shift sysex, tap notes, LIPSYNC tracks).

The RB conversion chain ports what Onyx/Magma would do: open notes → green,
1×/2× bass pedal variants, crash-safety sanitizing (overlaps, unknown tracks),
hand-map + init force markers, no-Magma fixups ([end], BEAT, drum [mix],
unisons), a ≥6-beat lead-in pad on both MIDI and audio, and a preview window
that starts 2 s before the first chorus when `song.ini` doesn't set one. A
non-fatal validation gate checks the result against every RB3 crash class we
know before it ships.

---

## Install and use (prebuilt version)

1. Go to **[Releases](../../releases)** and download `Downcharter+.zip`.
2. Unzip it into a folder of your choice.
3. Run **`Downcharter+.exe`** — no Python install needed.

### In the app
1. **Open…** → pick the folder with your charts (subfolders included).
2. Toggle what to generate: Expert+, difficulties (Hard/Medium/Easy), venue, talkies,
   lipsync, hide in-game background.
3. **Process folder** — originals are backed up as `.bak.mid` / `.bak.chart`.
4. **Convert** — pick PS3 folder, CON or `.sng` and the pedal variant (1×/2×).
5. Changed your mind? **Revert** restores the originals.

---

## Run from source

```bash
pip install -r requirements.txt
python main.py
```

Requires Python 3.10+. For GPU vocal separation install
`onnxruntime-directml` (already listed in `requirements.txt`) — never together
with plain `onnxruntime`.

---

## Build the `.exe`

```powershell
pip install -r requirements-build.txt
powershell -ExecutionPolicy Bypass -File build.ps1
```

Result:
- `dist/Downcharter+/Downcharter+.exe` — executable + dependencies (onedir)
- `dist/Downcharter+.zip` — ready to publish in Releases

Packaging uses PyInstaller via `downcharter.spec`, which bundles soundfile's
`libsndfile`, the CMU pronunciation dictionary and the onnxruntime-directml
runtime (DirectML.dll) for GPU vocal separation.

---

## Structure

```
main.py              GUI (tkinter)
downcharter/         engine package
  processor.py         orchestrates the folder (process_folder / revert_folder)
  guitar.py guitar2.py guitar/bass/keys reduction
  guitar_handmap.py    force-HOPO/strum hand-position map (RB)
  drums.py             drums reduction + Expert+
  venue.py             camera, lights, post-proc, BEAT track, animations
  venue_director.py    song-level venue decisions (repetition, anchors, climax)
  cut_events.py        directed camera cuts (probabilistic, official-calibrated)
  audio.py             audio analysis (RMS / bands / voice activity / feel)
  separate.py          MDX-NET vocal separation (DirectML GPU / CPU)
  lipsync.py           phonemes → visemes → keyframes (mouth + facial)
  milo.py              native .milo builder (CharLipSync container)
  chart.py             reading .chart → .mid
  convert.py           RB conversion chain (open notes, sanitize, fixups, pad)
  ps3build.py          PS3/RPCS3 folder builder (+ songs.dta from song.ini)
  stfs.py              Xbox 360 CON/STFS packer
  sng.py               YARG/CH .sng packer
  validate.py          pre-pack RB3 crash-class checks (non-fatal)
  mogg.py              .mogg builder (stems → multichannel, 44.1 kHz)
  art.py               .png_ps3/.png_xbox album art (DXT1)
  midi_utils.py        tempo / MIDI event helpers
downcharter.spec     PyInstaller recipe
build.ps1            build script → dist/
tests/               pytest suite (185+ tests)
```

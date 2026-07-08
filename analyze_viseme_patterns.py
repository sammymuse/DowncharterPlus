"""Analyze viseme patterns: which visemes appear together, weights, timing."""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from downcharter import milo

MIDI_DIR = Path(__file__).parent / "midis" / "Lipsync learn songs" / "milo_xbox_files"

MOUTH_VISEMES = {
    'Bump_hi', 'Bump_lo', 'Cage_hi', 'Cage_lo', 'Church_hi', 'Church_lo',
    'Earth_hi', 'Earth_lo', 'Eat_hi', 'Eat_lo', 'Fave_hi', 'Fave_lo',
    'If_hi', 'If_lo', 'New_hi', 'New_lo', 'Oat_hi', 'Oat_lo', 'Ox_hi',
    'Ox_lo', 'Roar_hi', 'Roar_lo', 'Size_hi', 'Size_lo', 'Though_hi',
    'Though_lo', 'Told_hi', 'Told_lo', 'Wet_hi', 'Wet_lo'
}

FACIAL_VISEMES = {'Blink', 'Squint', 'Brow_down', 'Brow_up', 'Brow_aggressive', 
                  'Brow_pouty', 'Brow_dramatic', 'Wide_eyed'}

def analyze_viseme_combos(path: Path) -> dict:
    """Analyze which visemes appear together and their weights."""
    try:
        data = milo.parse_song_lipsync(path.read_bytes())
    except Exception as e:
        return {"error": str(e)}
    
    frames = data["frames"]
    n_frames = data["n_frames"]
    
    # Count viseme combinations (mouth only)
    mouth_combos = defaultdict(int)
    facial_combos = defaultdict(int)
    
    # Track viseme co-occurrence
    viseme_pairs = defaultdict(int)
    
    # Track weight distributions per viseme
    viseme_weights = defaultdict(list)
    
    for fr, state in frames.items():
        mouth_vis = sorted([v for v in state if v in MOUTH_VISEMES])
        facial_vis = sorted([v for v in state if v in FACIAL_VISEMES])
        
        if mouth_vis:
            mouth_combos[tuple(mouth_vis)] += 1
        if facial_vis:
            facial_combos[tuple(facial_vis)] += 1
        
        # Track pairs
        for i, v1 in enumerate(mouth_vis):
            for v2 in mouth_vis[i+1:]:
                viseme_pairs[(v1, v2)] += 1
        
        # Track weights
        for vis, w in state.items():
            viseme_weights[vis].append(w)
    
    return {
        "path": path.name,
        "n_frames": n_frames,
        "mouth_combos": dict(mouth_combos),
        "facial_combos": dict(facial_combos),
        "viseme_pairs": dict(viseme_pairs),
        "viseme_weights": {k: (sum(v)/len(v), min(v), max(v), len(v)) 
                          for k, v in viseme_weights.items()},
    }

def main():
    milos = sorted(Path(MIDI_DIR).rglob("*.milo_xbox"))[:10]  # First 10 for speed
    print(f"Analyzing viseme patterns in {len(milos)} milos...\n")
    
    all_mouth_combos = defaultdict(int)
    all_facial_combos = defaultdict(int)
    all_pairs = defaultdict(int)
    all_weights = defaultdict(list)
    
    for m in milos:
        stats = analyze_viseme_combos(m)
        if "error" in stats:
            continue
        
        for combo, cnt in stats["mouth_combos"].items():
            all_mouth_combos[combo] += cnt
        for combo, cnt in stats["facial_combos"].items():
            all_facial_combos[combo] += cnt
        for pair, cnt in stats["viseme_pairs"].items():
            all_pairs[pair] += cnt
        for vis, (avg, mn, mx, n) in stats["viseme_weights"].items():
            all_weights[vis].extend([avg] * n)
    
    print("=" * 80)
    print("TOP MOUTH VISEME COMBINATIONS")
    print("=" * 80)
    top_mouth = sorted(all_mouth_combos.items(), key=lambda x: -x[1])[:20]
    for combo, cnt in top_mouth:
        print(f"  {str(combo):60s} {cnt:6d}")
    print()
    
    print("=" * 80)
    print("TOP FACIAL EXPRESSION COMBINATIONS")
    print("=" * 80)
    top_facial = sorted(all_facial_combos.items(), key=lambda x: -x[1])[:15]
    for combo, cnt in top_facial:
        print(f"  {str(combo):60s} {cnt:6d}")
    print()
    
    print("=" * 80)
    print("TOP VISEME PAIRS (co-occurrence)")
    print("=" * 80)
    top_pairs = sorted(all_pairs.items(), key=lambda x: -x[1])[:15]
    for (v1, v2), cnt in top_pairs:
        print(f"  {v1:15s} + {v2:15s}  {cnt:6d}")
    print()
    
    print("=" * 80)
    print("VISEME WEIGHT STATS (avg, min, max, count)")
    print("=" * 80)
    for vis in sorted(all_weights.keys()):
        vals = all_weights[vis]
        avg = sum(vals) / len(vals)
        print(f"  {vis:20s}  avg={avg:6.1f}  min={min(vals):3.0f}  max={max(vals):3.0f}  n={len(vals):6d}")

if __name__ == "__main__":
    main()

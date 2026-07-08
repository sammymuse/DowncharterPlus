"""Check viseme weights in generated lipsync vs official."""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from downcharter import milo, lipsync

MIDI_DIR = Path(__file__).parent / "midis" / "Lipsync learn songs" / "milo_xbox_files"

def analyze_official_weights(path: Path) -> dict:
    """Extract viseme weight stats from an official milo."""
    try:
        data = milo.parse_song_lipsync(path.read_bytes())
    except Exception as e:
        return {"error": str(e)}
    
    frames = data["frames"]
    viseme_weights = defaultdict(list)
    
    for fr, state in frames.items():
        for vis, w in state.items():
            viseme_weights[vis].append(w)
    
    return {
        "path": path.name,
        "viseme_weights": {k: (sum(v)/len(v), min(v), max(v), len(v)) 
                          for k, v in viseme_weights.items()},
    }

def analyze_our_weights() -> dict:
    """Generate a realistic lipsync and extract viseme weight stats."""
    # Create a more realistic song with many syllables
    spans = []
    t = 10.0
    words = ["Hello", "world", "this", "is", "a", "test", "of", "the", "lipsync",
             "system", "with", "many", "syllables", "to", "analyze", "properly",
             "the", "viseme", "weights", "and", "transitions", "between", "them"]
    
    for word in words:
        dur = 0.3 + (len(word) * 0.05)
        spans.append((t, t + dur, word, 0.8))
        t += dur + 0.2
    
    song_len_s = t + 10.0
    
    # Mock vocal notes
    vocal_notes = [(sp[0], 60 + i % 12) for i, sp in enumerate(spans)]
    phrase_ends = [spans[i][1] for i in range(0, len(spans), 4)]
    
    # Generate frames
    frames, n_frames = lipsync.frames_from_spans(
        spans, song_len_s, lang="en", facial_seed=42,
        vocal_notes=vocal_notes, phrase_ends=phrase_ends
    )
    
    # Extract weights
    viseme_weights = defaultdict(list)
    for fr, state in frames.items():
        for vis, w in state.items():
            viseme_weights[vis].append(w)
    
    return {
        "viseme_weights": {k: (sum(v)/len(v), min(v), max(v), len(v)) 
                          for k, v in viseme_weights.items()},
    }

def main():
    print("=" * 80)
    print("VISEME WEIGHT COMPARISON")
    print("=" * 80)
    print()
    
    # Analyze first 5 official milos
    milos = sorted(Path(MIDI_DIR).rglob("*.milo_xbox"))[:5]
    official_weights = defaultdict(list)
    
    for m in milos:
        stats = analyze_official_weights(m)
        if "error" in stats:
            continue
        for vis, (avg, mn, mx, n) in stats["viseme_weights"].items():
            official_weights[vis].append((avg, mn, mx))
    
    # Average official weights
    print("OFFICIAL MILOS (avg of 5 songs):")
    print(f"{'Viseme':<20} {'Avg':>6} {'Min':>4} {'Max':>4}")
    print("-" * 40)
    for vis in sorted(official_weights.keys()):
        entries = official_weights[vis]
        avg_avg = sum(e[0] for e in entries) / len(entries)
        global_min = min(e[1] for e in entries)
        global_max = max(e[2] for e in entries)
        print(f"{vis:<20} {avg_avg:6.1f} {global_min:4d} {global_max:4d}")
    print()
    
    # Analyze our generated lipsync
    our_stats = analyze_our_weights()
    
    print("OUR GENERATED LIPSYNC:")
    print(f"{'Viseme':<20} {'Avg':>6} {'Min':>4} {'Max':>4}")
    print("-" * 40)
    for vis in sorted(our_stats["viseme_weights"].keys()):
        avg, mn, mx, n = our_stats["viseme_weights"][vis]
        print(f"{vis:<20} {avg:6.1f} {mn:4d} {mx:4d}")

if __name__ == "__main__":
    main()

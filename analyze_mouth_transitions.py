"""Deep analysis: how mouth visemes transition in official milos."""
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from downcharter import milo

MIDI_DIR = Path(__file__).parent / "midis" / "Lipsync learn songs" / "milo_xbox_files"

# Mouth visemes (exclude facial expressions)
MOUTH_VISEMES = {
    'Bump_hi', 'Bump_lo', 'Cage_hi', 'Cage_lo', 'Church_hi', 'Church_lo',
    'Earth_hi', 'Earth_lo', 'Eat_hi', 'Eat_lo', 'Fave_hi', 'Fave_lo',
    'If_hi', 'If_lo', 'New_hi', 'New_lo', 'Oat_hi', 'Oat_lo', 'Ox_hi',
    'Ox_lo', 'Roar_hi', 'Roar_lo', 'Size_hi', 'Size_lo', 'Though_hi',
    'Though_lo', 'Told_hi', 'Told_lo', 'Wet_hi', 'Wet_lo'
}

def analyze_mouth_transitions(path: Path) -> dict:
    """Analyze how mouth visemes change frame-to-frame."""
    try:
        data = milo.parse_song_lipsync(path.read_bytes())
    except Exception as e:
        return {"error": str(e)}
    
    frames = data["frames"]
    n_frames = data["n_frames"]
    
    # Extract mouth state per frame (total mouth openness)
    mouth_states = []
    for fr in range(n_frames):
        state = frames.get(fr, {})
        mouth_total = sum(w for vis, w in state.items() if vis in MOUTH_VISEMES)
        mouth_states.append(mouth_total)
    
    # Analyze transitions
    transitions = []
    for i in range(1, len(mouth_states)):
        diff = abs(mouth_states[i] - mouth_states[i-1])
        transitions.append(diff)
    
    # Find "closed" frames (mouth_total == 0)
    closed_frames = [i for i, total in enumerate(mouth_states) if total == 0]
    
    # Analyze how mouth opens/closes
    opens = []
    closes = []
    for i in range(1, len(mouth_states)):
        diff = mouth_states[i] - mouth_states[i-1]
        if diff > 0:
            opens.append(diff)
        elif diff < 0:
            closes.append(abs(diff))
    
    # Find sustained notes (long runs of similar mouth state)
    sustained = []
    run_start = 0
    for i in range(1, len(mouth_states)):
        if abs(mouth_states[i] - mouth_states[run_start]) > 20:
            if i - run_start > 10:  # > 10 frames = sustained
                sustained.append(i - run_start)
            run_start = i
    
    return {
        "path": path.name,
        "n_frames": n_frames,
        "n_closed": len(closed_frames),
        "pct_closed": 100 * len(closed_frames) / n_frames,
        "avg_mouth": sum(mouth_states) / len(mouth_states),
        "max_mouth": max(mouth_states),
        "avg_transition": sum(transitions) / len(transitions) if transitions else 0,
        "max_transition": max(transitions) if transitions else 0,
        "avg_open": sum(opens) / len(opens) if opens else 0,
        "avg_close": sum(closes) / len(closes) if closes else 0,
        "n_sustained": len(sustained),
        "avg_sustained": sum(sustained) / len(sustained) if sustained else 0,
        "mouth_states_sample": mouth_states[:100],  # First 100 frames
    }

def main():
    milos = sorted(Path(MIDI_DIR).rglob("*.milo_xbox"))
    print(f"Analyzing mouth transitions in {len(milos)} milos...\n")
    
    all_stats = []
    for m in milos:
        stats = analyze_mouth_transitions(m)
        if "error" not in stats:
            all_stats.append(stats)
    
    if not all_stats:
        print("No valid milos found!")
        return
    
    # Aggregate
    total_frames = sum(s["n_frames"] for s in all_stats)
    total_closed = sum(s["n_closed"] for s in all_stats)
    
    print("=" * 80)
    print("MOUTH TRANSITION ANALYSIS")
    print("=" * 80)
    print(f"Total frames: {total_frames}")
    print(f"Closed frames (mouth_total=0): {total_closed} ({100*total_closed/total_frames:.2f}%)")
    print(f"Open frames: {total_frames - total_closed} ({100*(total_frames-total_closed)/total_frames:.2f}%)")
    print()
    
    avg_mouth = sum(s["avg_mouth"] for s in all_stats) / len(all_stats)
    max_mouth = max(s["max_mouth"] for s in all_stats)
    print(f"Avg mouth openness: {avg_mouth:.1f}")
    print(f"Max mouth openness: {max_mouth}")
    print()
    
    avg_trans = sum(s["avg_transition"] for s in all_stats) / len(all_stats)
    max_trans = max(s["max_transition"] for s in all_stats)
    print(f"Avg transition size: {avg_trans:.1f}")
    print(f"Max transition size: {max_trans}")
    print()
    
    avg_open = sum(s["avg_open"] for s in all_stats) / len(all_stats)
    avg_close = sum(s["avg_close"] for s in all_stats) / len(all_stats)
    print(f"Avg open step: {avg_open:.1f}")
    print(f"Avg close step: {avg_close:.1f}")
    print()
    
    # Sustained notes
    all_sustained = [s["n_sustained"] for s in all_stats]
    avg_sustained_len = sum(s["avg_sustained"] for s in all_stats if s["n_sustained"] > 0) / len([s for s in all_stats if s["n_sustained"] > 0]) if any(s["n_sustained"] > 0 for s in all_stats) else 0
    print(f"Sustained notes (>10 frames): {sum(all_sustained)} total")
    print(f"Avg sustained length: {avg_sustained_len:.1f} frames")
    print()
    
    # Sample trajectory
    print("=" * 80)
    print("SAMPLE MOUTH TRAJECTORY (first 100 frames of first milo)")
    print("=" * 80)
    sample = all_stats[0]["mouth_states_sample"]
    for i in range(0, len(sample), 10):
        chunk = sample[i:i+10]
        print(f"Frames {i:3d}-{i+9:3d}: {chunk}")
    
    # Compare with our code
    print()
    print("=" * 80)
    print("COMPARISON WITH OUR CODE")
    print("=" * 80)
    print(f"Official: {100*total_closed/total_frames:.2f}% closed frames")
    print(f"Our code: We have gaps between syllables (probably 10-20% closed)")
    print()
    print(f"Official: Avg transition = {avg_trans:.1f} per frame")
    print(f"Our code: _TRANSITION_S = 0.22s = 6.6 frames per transition")
    print()
    print(f"Official: Mouth opens/closes in ~{avg_open:.0f}/{avg_close:.0f} steps")
    print(f"Our code: attack/release = 0.22s * 0.4 = 0.088s = 2.6 frames")

if __name__ == "__main__":
    main()

"""Analyze official lipsync patterns from reference .milo_xbox files."""
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from downcharter import milo

MIDI_DIR = Path(__file__).parent / "midis" / "Lipsync learn songs" / "milo_xbox_files"

def analyze_milo(path: Path) -> dict:
    """Extract stats from a single .milo_xbox file."""
    try:
        data = milo.parse_song_lipsync(path.read_bytes())
    except Exception as e:
        return {"error": str(e), "path": path.name}
    
    visemes = data["visemes"]
    frames = data["frames"]
    n_frames = data["n_frames"]
    
    # Count viseme usage
    viseme_counts = defaultdict(int)
    viseme_weights = defaultdict(list)
    for fr, state in frames.items():
        for vis, w in state.items():
            viseme_counts[vis] += 1
            viseme_weights[vis].append(w)
    
    # Analyze transitions
    transitions = []
    prev_state = {}
    for fr in range(n_frames):
        cur_state = frames.get(fr, {})
        if cur_state != prev_state:
            transitions.append(fr)
        prev_state = cur_state
    
    # Analyze gaps (frames with no visemes)
    gaps = []
    gap_start = None
    for fr in range(n_frames):
        if not frames.get(fr):
            if gap_start is None:
                gap_start = fr
        else:
            if gap_start is not None:
                gaps.append(fr - gap_start)
                gap_start = None
    if gap_start is not None:
        gaps.append(n_frames - gap_start)
    
    # Analyze consecutive frames with same state
    runs = []
    run_start = 0
    prev = frames.get(0, {})
    for fr in range(1, n_frames):
        cur = frames.get(fr, {})
        if cur != prev:
            runs.append(fr - run_start)
            run_start = fr
            prev = cur
    runs.append(n_frames - run_start)
    
    return {
        "path": path.name,
        "n_visemes": len(visemes),
        "n_frames": n_frames,
        "n_active_frames": len(frames),
        "n_transitions": len(transitions),
        "n_gaps": len(gaps),
        "avg_gap": sum(gaps) / len(gaps) if gaps else 0,
        "max_gap": max(gaps) if gaps else 0,
        "avg_run": sum(runs) / len(runs) if runs else 0,
        "max_run": max(runs) if runs else 0,
        "viseme_counts": dict(viseme_counts),
        "viseme_weights": {k: (sum(v)/len(v), min(v), max(v)) for k, v in viseme_weights.items()},
    }

def main():
    milos = sorted(Path(MIDI_DIR).rglob("*.milo_xbox"))
    print(f"Analyzing {len(milos)} official milo files...\n")
    
    all_stats = []
    for m in milos:
        stats = analyze_milo(m)
        if "error" not in stats:
            all_stats.append(stats)
    
    if not all_stats:
        print("No valid milos found!")
        return
    
    # Aggregate stats
    total_frames = sum(s["n_frames"] for s in all_stats)
    total_active = sum(s["n_active_frames"] for s in all_stats)
    total_transitions = sum(s["n_transitions"] for s in all_stats)
    
    print("=" * 80)
    print("AGGREGATE STATS")
    print("=" * 80)
    print(f"Total milos analyzed: {len(all_stats)}")
    print(f"Total frames: {total_frames}")
    print(f"Active frames (with visemes): {total_active} ({100*total_active/total_frames:.1f}%)")
    print(f"Empty frames (gaps): {total_frames - total_active} ({100*(total_frames-total_active)/total_frames:.1f}%)")
    print(f"Total transitions: {total_transitions}")
    print(f"Avg transitions per milo: {total_transitions/len(all_stats):.1f}")
    print(f"Avg frames per milo: {total_frames/len(all_stats):.0f}")
    print()
    
    # Gap analysis
    all_gaps = [s["avg_gap"] for s in all_stats if s["n_gaps"] > 0]
    if all_gaps:
        print("GAP ANALYSIS (frames with no visemes between syllables)")
        print(f"Milos with gaps: {len(all_gaps)}/{len(all_stats)}")
        print(f"Avg gap duration: {sum(all_gaps)/len(all_gaps):.2f} frames")
        print(f"Max gap seen: {max(s['max_gap'] for s in all_stats)} frames")
        print()
    
    # Run analysis (consecutive frames with same state)
    all_runs = [s["avg_run"] for s in all_stats]
    print("RUN ANALYSIS (consecutive frames with same viseme state)")
    print(f"Avg run length: {sum(all_runs)/len(all_runs):.2f} frames")
    print(f"Max run seen: {max(s['max_run'] for s in all_stats)} frames")
    print()
    
    # Viseme usage
    print("VISEME USAGE (top 15)")
    viseme_totals = defaultdict(int)
    for s in all_stats:
        for vis, cnt in s["viseme_counts"].items():
            viseme_totals[vis] += cnt
    top_visemes = sorted(viseme_totals.items(), key=lambda x: -x[1])[:15]
    for vis, cnt in top_visemes:
        print(f"  {vis:20s} {cnt:6d} frames ({100*cnt/total_active:.1f}%)")
    print()
    
    # Weight analysis
    print("VISEME WEIGHT RANGES (avg, min, max)")
    weight_data = defaultdict(list)
    for s in all_stats:
        for vis, (avg, mn, mx) in s["viseme_weights"].items():
            weight_data[vis].append((avg, mn, mx))
    for vis in sorted(weight_data.keys()):
        entries = weight_data[vis]
        avg_avg = sum(e[0] for e in entries) / len(entries)
        global_min = min(e[1] for e in entries)
        global_max = max(e[2] for e in entries)
        print(f"  {vis:20s} avg={avg_avg:6.2f}  min={global_min:3d}  max={global_max:3d}")
    print()
    
    # Transition frequency
    print("TRANSITION FREQUENCY")
    print(f"Avg frames between transitions: {total_frames/total_transitions:.2f}")
    print(f"Transitions per second (at 30fps): {30*total_transitions/total_frames:.2f}")
    print()
    
    # Sample detailed analysis of 3 milos
    print("=" * 80)
    print("DETAILED SAMPLES (3 milos)")
    print("=" * 80)
    for s in all_stats[:3]:
        print(f"\n{s['path']}")
        print(f"  Frames: {s['n_frames']} (active: {s['n_active_frames']})")
        print(f"  Transitions: {s['n_transitions']}")
        print(f"  Gaps: {s['n_gaps']} (avg: {s['avg_gap']:.2f}, max: {s['max_gap']})")
        print(f"  Runs: avg={s['avg_run']:.2f}, max={s['max_run']}")
        print(f"  Top visemes:")
        for vis, cnt in sorted(s['viseme_counts'].items(), key=lambda x: -x[1])[:5]:
            print(f"    {vis:20s} {cnt:5d}")

if __name__ == "__main__":
    main()

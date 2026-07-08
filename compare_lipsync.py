"""Compare our generated lipsync with official milos."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from downcharter import milo, lipsync

MIDI_DIR = Path(__file__).parent / "midis" / "Lipsync learn songs" / "milo_xbox_files"

def analyze_official(path: Path) -> dict:
    """Extract key stats from an official milo."""
    try:
        data = milo.parse_song_lipsync(path.read_bytes())
    except Exception as e:
        return {"error": str(e)}
    
    frames = data["frames"]
    n_frames = data["n_frames"]
    
    # Count always-on facial expressions
    squint_count = sum(1 for fr in range(n_frames) if "Squint" in frames.get(fr, {}))
    brow_down_count = sum(1 for fr in range(n_frames) if "Brow_down" in frames.get(fr, {}))
    blink_count = sum(1 for fr in range(n_frames) if "Blink" in frames.get(fr, {}))
    
    # Count mouth visemes
    mouth_count = sum(1 for fr in range(n_frames) 
                     if any(v for v in frames.get(fr, {}) if not v.startswith(("Blink", "Squint", "Brow"))))
    
    return {
        "path": path.name,
        "n_frames": n_frames,
        "squint_pct": 100 * squint_count / n_frames,
        "brow_down_pct": 100 * brow_down_count / n_frames,
        "blink_pct": 100 * blink_count / n_frames,
        "mouth_pct": 100 * mouth_count / n_frames,
    }

def generate_test_lipsync(song_len_s: float = 180.0) -> dict:
    """Generate a test lipsync and analyze it."""
    # Create some test spans (simulating a song with lyrics)
    spans = [
        (10.0, 10.5, "Hello", 0.8),
        (11.0, 11.3, "world", 0.9),
        (12.0, 12.8, "this", 0.7),
        (13.0, 13.5, "is", 0.85),
        (14.0, 14.7, "a", 0.75),
        (15.0, 15.6, "test", 0.9),
    ]
    
    # Mock vocal notes and phrase ends to trigger facial animation
    vocal_notes = [(10.0, 60), (11.0, 62), (12.0, 64), (13.0, 65), (14.0, 67), (15.0, 69)]
    phrase_ends = [11.3, 13.5, 15.6]
    
    # Generate keyframes
    kf = lipsync.lipsync_keyframes_from_spans(
        spans, lang="en", song_len_s=song_len_s, facial_seed=42,
        vocal_notes=vocal_notes, phrase_ends=phrase_ends
    )
    
    # Convert to dense frames
    frames, n_frames = lipsync.frames_from_spans(
        spans, song_len_s, lang="en", facial_seed=42,
        vocal_notes=vocal_notes, phrase_ends=phrase_ends
    )
    
    # Analyze
    squint_count = sum(1 for fr in range(n_frames) if "Squint" in frames.get(fr, {}))
    brow_down_count = sum(1 for fr in range(n_frames) if "Brow_down" in frames.get(fr, {}))
    blink_count = sum(1 for fr in range(n_frames) if "Blink" in frames.get(fr, {}))
    mouth_count = sum(1 for fr in range(n_frames) 
                     if any(v for v in frames.get(fr, {}) if not v.startswith(("Blink", "Squint", "Brow"))))
    
    return {
        "n_frames": n_frames,
        "squint_pct": 100 * squint_count / n_frames,
        "brow_down_pct": 100 * brow_down_count / n_frames,
        "blink_pct": 100 * blink_count / n_frames,
        "mouth_pct": 100 * mouth_count / n_frames,
    }

def main():
    print("=" * 80)
    print("COMPARISON: OFFICIAL vs OUR GENERATED LIPSYNC")
    print("=" * 80)
    print()
    
    # Analyze first 10 official milos
    milos = sorted(Path(MIDI_DIR).rglob("*.milo_xbox"))[:10]
    official_stats = []
    for m in milos:
        stats = analyze_official(m)
        if "error" not in stats:
            official_stats.append(stats)
    
    if not official_stats:
        print("No official milos found!")
        return
    
    # Average official stats
    avg_squint = sum(s["squint_pct"] for s in official_stats) / len(official_stats)
    avg_brow = sum(s["brow_down_pct"] for s in official_stats) / len(official_stats)
    avg_blink = sum(s["blink_pct"] for s in official_stats) / len(official_stats)
    avg_mouth = sum(s["mouth_pct"] for s in official_stats) / len(official_stats)
    
    print("OFFICIAL MILOS (avg of 10 songs):")
    print(f"  Squint:        {avg_squint:5.1f}%")
    print(f"  Brow_down:     {avg_brow:5.1f}%")
    print(f"  Blink:         {avg_blink:5.1f}%")
    print(f"  Mouth visemes: {avg_mouth:5.1f}%")
    print()
    
    # Generate and analyze our lipsync
    our_stats = generate_test_lipsync()
    
    print("OUR GENERATED LIPSYNC (test song):")
    print(f"  Squint:        {our_stats['squint_pct']:5.1f}%")
    print(f"  Brow_down:     {our_stats['brow_down_pct']:5.1f}%")
    print(f"  Blink:         {our_stats['blink_pct']:5.1f}%")
    print(f"  Mouth visemes: {our_stats['mouth_pct']:5.1f}%")
    print()
    
    print("COMPARISON:")
    squint_ok = "OK" if abs(avg_squint - our_stats['squint_pct']) < 10 else "FAIL"
    brow_ok = "OK" if abs(avg_brow - our_stats['brow_down_pct']) < 15 else "FAIL"
    blink_ok = "OK" if abs(avg_blink - our_stats['blink_pct']) < 20 else "FAIL"
    print(f"  Squint:        Official={avg_squint:5.1f}% vs Ours={our_stats['squint_pct']:5.1f}% [{squint_ok}]")
    print(f"  Brow_down:     Official={avg_brow:5.1f}% vs Ours={our_stats['brow_down_pct']:5.1f}% [{brow_ok}]")
    print(f"  Blink:         Official={avg_blink:5.1f}% vs Ours={our_stats['blink_pct']:5.1f}% [{blink_ok}]")

if __name__ == "__main__":
    main()

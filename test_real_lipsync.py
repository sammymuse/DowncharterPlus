"""Real-world lipsync test: extract vocal stem from .mogg, generate lipsync, compare with official."""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from downcharter import audio, lipsync, milo

MIDI_DIR = Path(__file__).parent / "midis" / "Lipsync learn songs"

def test_song(song_folder: str):
    """Test lipsync generation for a single song."""
    mogg_files = list((MIDI_DIR / "mogg_files" / song_folder).glob("*.mogg"))
    milo_files = list((MIDI_DIR / "milo_xbox_files" / song_folder).glob("*.milo_xbox"))
    
    if not mogg_files or not milo_files:
        print(f"  [SKIP] Missing mogg or milo for {song_folder}")
        return None
    
    mogg_path = mogg_files[0]
    milo_path = milo_files[0]
    
    print(f"\n{'='*80}")
    print(f"TESTING: {song_folder}")
    print(f"  Mogg: {mogg_path.name}")
    print(f"  Milo: {milo_path.name}")
    print(f"{'='*80}")
    
    # Step 1: Extract vocal stem from mogg
    print("\n[1/5] Extracting vocal stem from .mogg...")
    vocal_data = audio.load_vocal_from_mogg(str(mogg_path))
    if vocal_data is None:
        print("  [FAIL] Could not extract vocal stem (encrypted or missing channels)")
        return None
    
    mono, sr = vocal_data
    duration_s = len(mono) / sr
    print(f"  [OK] Extracted {duration_s:.1f}s of vocal audio @ {sr} Hz")
    
    # Step 2: Detect voice activity
    print("\n[2/5] Detecting voice activity...")
    va = audio.voice_activity([str(mogg_path)])
    if va is None:
        print("  [FAIL] No voice activity detected")
        return None
    
    env, hop_s, thr = va
    print(f"  [OK] Voice activity envelope: {len(env)} frames, threshold={thr:.4f}")
    
    # Step 3: Get active singing spans
    print("\n[3/5] Extracting singing spans...")
    spans = audio.voice_active_spans(va, min_dur=0.3, merge_gap=0.25)
    print(f"  [OK] Found {len(spans)} singing spans")
    if len(spans) == 0:
        print("  [FAIL] No singing spans detected")
        return None
    
    # Show first 5 spans
    for i, (start, end) in enumerate(spans[:5]):
        print(f"    Span {i+1}: {start:.2f}s - {end:.2f}s (dur={end-start:.2f}s)")
    if len(spans) > 5:
        print(f"    ... and {len(spans) - 5} more")
    
    # Step 4: Generate lipsync from spans
    print("\n[4/5] Generating lipsync from vocal spans...")
    # Convert spans to the format expected by lipsync functions
    # spans format: [(start_s, end_s, text, gain)]
    # We don't have text, so use dummy text
    lip_spans = []
    for start, end in spans:
        # Calculate gain from voice activity envelope
        gain = audio.syllable_gain(va, start, end)
        lip_spans.append((start, end, "LA", gain))  # "LA" as dummy syllable
    
    # Extract phrase_ends from spans (end of each span = phrase boundary)
    phrase_ends = [end for start, end in spans]
    
    # Generate keyframes
    kf = lipsync.lipsync_keyframes_from_spans(
        lip_spans, lang="en", song_len_s=duration_s, facial_seed=42,
        phrase_ends=phrase_ends
    )
    print(f"  [OK] Generated {len(kf)} keyframes")
    
    # Convert to dense frames
    frames, n_frames = lipsync.frames_from_spans(
        lip_spans, duration_s, lang="en", facial_seed=42,
        phrase_ends=phrase_ends
    )
    print(f"  [OK] Generated {n_frames} frames ({len(frames)} active)")
    
    # Step 5: Compare with official milo
    print("\n[5/5] Comparing with official .milo_xbox...")
    try:
        official_data = milo.parse_song_lipsync(milo_path.read_bytes())
    except Exception as e:
        print(f"  [FAIL] Could not parse official milo: {e}")
        return None
    
    official_frames = official_data["frames"]
    official_n_frames = official_data["n_frames"]
    
    print(f"  Official: {official_n_frames} frames ({len(official_frames)} active)")
    print(f"  Ours:     {n_frames} frames ({len(frames)} active)")
    
    # Compare frame counts
    frame_diff = abs(n_frames - official_n_frames)
    print(f"  Frame count difference: {frame_diff} ({100*frame_diff/max(n_frames, official_n_frames):.1f}%)")
    
    # Compare viseme usage
    print("\n  VISEME USAGE COMPARISON:")
    our_visemes = defaultdict(int)
    for fr, state in frames.items():
        for vis in state:
            our_visemes[vis] += 1
    
    off_visemes = defaultdict(int)
    for fr, state in official_frames.items():
        for vis in state:
            off_visemes[vis] += 1
    
    all_visemes = sorted(set(our_visemes) | set(off_visemes))
    print(f"    {'Viseme':<20} {'Official':>10} {'Ours':>10} {'Diff':>10}")
    print(f"    {'-'*52}")
    
    total_off = sum(off_visemes.values())
    total_our = sum(our_visemes.values())
    
    for vis in all_visemes[:15]:  # Top 15
        off_pct = 100 * off_visemes[vis] / total_off if total_off > 0 else 0
        our_pct = 100 * our_visemes[vis] / total_our if total_our > 0 else 0
        diff = abs(off_pct - our_pct)
        status = "OK" if diff < 10 else "WARN" if diff < 20 else "FAIL"
        print(f"    {vis:<20} {off_pct:9.1f}% {our_pct:9.1f}% {diff:9.1f}% [{status}]")
    
    # Compare weights for a sample of frames
    print("\n  WEIGHT COMPARISON (first 100 active frames):")
    sample_frames = sorted(frames.keys())[:100]
    weight_diffs = []
    
    for fr in sample_frames:
        if fr in official_frames:
            for vis in frames[fr]:
                if vis in official_frames[fr]:
                    our_w = frames[fr][vis]
                    off_w = official_frames[fr][vis]
                    weight_diffs.append(abs(our_w - off_w))
    
    if weight_diffs:
        avg_diff = sum(weight_diffs) / len(weight_diffs)
        max_diff = max(weight_diffs)
        print(f"    Avg weight difference: {avg_diff:.1f}")
        print(f"    Max weight difference: {max_diff}")
    
    # Summary
    print("\n  SUMMARY:")
    active_pct_our = 100 * len(frames) / n_frames if n_frames > 0 else 0
    active_pct_off = 100 * len(official_frames) / official_n_frames if official_n_frames > 0 else 0
    print(f"    Active frames: Official={active_pct_off:.1f}%, Ours={active_pct_our:.1f}%")
    
    return {
        "song": song_folder,
        "duration_s": duration_s,
        "n_spans": len(spans),
        "n_frames_ours": n_frames,
        "n_frames_official": official_n_frames,
        "active_pct_ours": active_pct_our,
        "active_pct_official": active_pct_off,
        "avg_weight_diff": avg_diff if weight_diffs else None,
    }

def main():
    print("=" * 80)
    print("REAL-WORLD LIPSYNC TEST WITH VOCAL STEMS")
    print("=" * 80)
    
    # Test a few songs
    test_songs = [
        "bohemianrhapsody",
        "smokeonthewater",
        "crazytrain",
        "rehab",
        "imagine",
    ]
    
    results = []
    for song in test_songs:
        result = test_song(song)
        if result:
            results.append(result)
    
    if results:
        print("\n" + "=" * 80)
        print("AGGREGATE RESULTS")
        print("=" * 80)
        print(f"{'Song':<25} {'Dur':>6} {'Spans':>6} {'Frames':>8} {'Active%':>8} {'WtDiff':>8}")
        print("-" * 80)
        
        for r in results:
            wt_diff = f"{r['avg_weight_diff']:.1f}" if r['avg_weight_diff'] else "N/A"
            print(f"{r['song']:<25} {r['duration_s']:5.1f}s {r['n_spans']:6d} "
                  f"{r['n_frames_ours']:8d} {r['active_pct_ours']:7.1f}% {wt_diff:>8}")

if __name__ == "__main__":
    main()

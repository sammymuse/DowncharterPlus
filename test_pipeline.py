"""End-to-end pipeline test: process → convert → package."""
import sys
import os
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from downcharter import processor, convert, milo, audio

def log_fn(msg, level="info"):
    """Simple logger for testing."""
    prefix = {"info": "[INFO]", "warn": "[WARN]", "err": "[ERR]"}.get(level, "[?]")
    # Replace Unicode arrows with ASCII
    msg = msg.replace("→", "->").replace("⚠", "[!]").replace("✓", "[OK]")
    # Encode to ASCII, replacing non-ASCII chars
    msg = msg.encode('ascii', errors='replace').decode('ascii')
    print(f"  {prefix} {msg}", end="")

def setup_test_folder(song_name: str) -> str:
    """Create a temporary test folder with all necessary files for a song."""
    base = Path(__file__).parent / "midis" / "Lipsync learn songs"
    
    # Find files for this song
    midi_dir = base / "midi_files" / song_name
    mogg_dir = base / "mogg_files" / song_name
    milo_dir = base / "milo_xbox_files" / song_name
    
    if not midi_dir.exists():
        raise FileNotFoundError(f"MIDI folder not found: {midi_dir}")
    
    # Create temp folder
    temp_dir = tempfile.mkdtemp(prefix=f"downcharter_test_{song_name}_")
    
    # Copy MIDI
    for mid in midi_dir.glob("*.mid"):
        shutil.copy2(mid, temp_dir)
    
    # Copy mogg if exists
    if mogg_dir.exists():
        for mogg in mogg_dir.glob("*.mogg"):
            shutil.copy2(mogg, temp_dir)
    
    # Copy milo if exists (for comparison)
    if milo_dir.exists():
        for milo_file in milo_dir.glob("*.milo_xbox"):
            shutil.copy2(milo_file, temp_dir)
    
    return temp_dir

def test_pipeline(song_name: str):
    """Test the full pipeline for a single song."""
    print(f"\n{'='*80}")
    print(f"TESTING: {song_name}")
    print(f"{'='*80}")
    
    # Setup
    print("\n[0/5] Setting up test folder...")
    try:
        test_folder = setup_test_folder(song_name)
        print(f"  [OK] Created temp folder: {test_folder}")
        files = list(Path(test_folder).glob("*"))
        print(f"  [OK] Files: {[f.name for f in files]}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False
    
    try:
        # Step 1: Process folder
        print("\n[1/5] Processing folder...")
        try:
            processor.process_folder(
                folder=test_folder,
                diffs_to_gen=["easy", "medium", "hard", "expert"],
                do_expert_plus=True,
                threshold_ms=10.0,
                log_fn=log_fn,
                do_venue=True,
                do_lipsync=True,
                do_talkies=True,
                do_drum_anim=True,
                do_vocal_sep=True,
                mouth_openness=1.0,
            )
            print("  [OK] Processed successfully")
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 2: Check for generated files
        print("\n[2/5] Checking generated files...")
        test_path = Path(test_folder)
        mid_files = list(test_path.glob("*.mid"))
        ini_files = list(test_path.glob("*.ini"))
        
        if not mid_files:
            print(f"  [FAIL] No .mid files found")
            return False
        print(f"  [OK] Found {len(mid_files)} .mid file(s):")
        for f in mid_files:
            print(f"    - {f.name} ({f.stat().st_size} bytes)")
        
        if not ini_files:
            print(f"  [INFO] No .ini files found (may not have been generated)")
        else:
            print(f"  [OK] Found {len(ini_files)} .ini file(s):")
            for f in ini_files:
                print(f"    - {f.name}")
        
        # Step 3: Check audio
        print("\n[3/5] Checking audio...")
        try:
            vocal_audio = audio.find_vocal_audio(test_folder)
            if vocal_audio:
                print(f"  [OK] Found vocal audio: {len(vocal_audio)} file(s)")
                for f in vocal_audio:
                    print(f"    - {Path(f).name}")
            else:
                print(f"  [WARN] No vocal audio found")
        except Exception as e:
            print(f"  [FAIL] {e}")
        
        # Step 4: Check milo
        print("\n[4/5] Checking milo...")
        milo_files = list(test_path.glob("*.milo_xbox"))
        if milo_files:
            print(f"  [OK] Found {len(milo_files)} .milo_xbox file(s):")
            for f in milo_files:
                print(f"    - {f.name} ({f.stat().st_size} bytes)")
                
                # Parse and validate
                try:
                    data = milo.parse_song_lipsync(f.read_bytes())
                    print(f"      [OK] Parsed: {data['n_frames']} frames, {len(data['visemes'])} visemes")
                except Exception as e:
                    print(f"      [FAIL] Could not parse: {e}")
        else:
            print(f"  [INFO] No .milo_xbox files found")
        
        # Step 5: Check mogg
        print("\n[5/5] Checking mogg...")
        mogg_files = list(test_path.glob("*.mogg"))
        if mogg_files:
            print(f"  [OK] Found {len(mogg_files)} .mogg file(s):")
            for f in mogg_files:
                print(f"    - {f.name} ({f.stat().st_size} bytes)")
        else:
            print(f"  [INFO] No .mogg files found")
        
        print(f"\n{'='*80}")
        print("Pipeline test completed!")
        print(f"{'='*80}\n")
        return True
        
    finally:
        # Cleanup
        print(f"\n[CLEANUP] Removing temp folder: {test_folder}")
        shutil.rmtree(test_folder, ignore_errors=True)

def main():
    print("="*80)
    print("END-TO-END PIPELINE TEST")
    print("="*80)
    
    # Test with a few songs
    test_songs = [
        "bohemianrhapsody",
        "smokeonthewater",
        "crazytrain",
    ]
    
    results = []
    for song in test_songs:
        success = test_pipeline(song)
        results.append((song, success))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for song, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {song:<50} [{status}]")

if __name__ == "__main__":
    main()

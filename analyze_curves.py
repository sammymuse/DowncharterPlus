"""Analyze official transition curves in detail."""
import sys
sys.path.insert(0, '.')
from downcharter import milo
from pathlib import Path

# Analyze transition curves in detail
milo_path = Path('midis/Lipsync learn songs/milo_xbox_files/bohemianrhapsody/Queen - Bohemian Rhapsody.milo_xbox')
data = milo.parse_song_lipsync(milo_path.read_bytes())
frames = data['frames']
n_frames = data['n_frames']

# Find smooth transitions (sequences where a viseme changes gradually)
transitions = []
for vis in ['If_lo', 'Ox_lo', 'Eat_lo']:
    weights = [frames.get(fr, {}).get(vis, 0) for fr in range(n_frames)]
    
    # Find transitions (where weight changes)
    i = 0
    while i < len(weights):
        if weights[i] > 0:
            # Found start of a viseme activation
            start = i
            peak = weights[i]
            # Find end
            while i < len(weights) and weights[i] > 0:
                peak = max(peak, weights[i])
                i += 1
            end = i
            
            # Only analyze transitions with at least 3 frames
            if end - start >= 3 and peak > 50:
                transition = weights[start:end]
                transitions.append({
                    'viseme': vis,
                    'length': len(transition),
                    'peak': peak,
                    'shape': transition
                })
        else:
            i += 1

print(f'Found {len(transitions)} transitions')
print()

# Analyze curve shapes
opening_curves = []
closing_curves = []

for t in transitions[:20]:  # First 20
    shape = t['shape']
    peak = t['peak']
    
    if len(shape) >= 3:
        # Opening: starts at 0, goes to peak
        if shape[0] < shape[-1]:
            opening_curves.append(shape)
        # Closing: starts at peak, goes to 0
        elif shape[0] > shape[-1]:
            closing_curves.append(shape)

print(f'Opening curves: {len(opening_curves)}')
print(f'Closing curves: {len(closing_curves)}')
print()

# Analyze opening curve shape
if opening_curves:
    print('OPENING CURVES (0 -> peak):')
    for curve in opening_curves[:5]:
        # Normalize to 0-1 range
        normalized = [w / max(curve) for w in curve]
        formatted = [f'{w:.2f}' for w in normalized]
        print(f'  Length {len(curve)}: {formatted}')
    print()

# Analyze closing curve shape
if closing_curves:
    print('CLOSING CURVES (peak -> 0):')
    for curve in closing_curves[:5]:
        # Normalize to 0-1 range
        normalized = [w / max(curve) for w in curve]
        formatted = [f'{w:.2f}' for w in normalized]
        print(f'  Length {len(curve)}: {formatted}')

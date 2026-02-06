"""Quick script to verify dataset setup before training."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
import os

print("=" * 70)
print("Dataset Verification")
print("=" * 70)

# Check directories
audio_dir = Path("dataset/audio_files")
info_dir = Path("dataset/info")
test_dir = Path("test_audio")

print(f"\n1. Directory Check:")
print(f"   Audio files dir: {audio_dir} - {'EXISTS' if audio_dir.exists() else 'MISSING'}")
print(f"   Transcripts dir: {info_dir} - {'EXISTS' if info_dir.exists() else 'MISSING'}")
print(f"   Test audio dir: {test_dir} - {'EXISTS' if test_dir.exists() else 'MISSING (optional)'}")

# Check audio files
if audio_dir.exists():
    audio_files = list(audio_dir.glob("*.wav"))
    print(f"\n2. Audio Files:")
    print(f"   Found {len(audio_files)} audio files in {audio_dir}")
    if len(audio_files) > 0:
        print(f"   Sample files: {[f.name for f in audio_files[:3]]}")
else:
    audio_files = []
    print(f"\n2. Audio Files: Directory not found")

# Check JSON files
if info_dir.exists():
    json_files = list(info_dir.glob("*.json"))
    print(f"\n3. Transcript Files:")
    print(f"   Found {len(json_files)} JSON files in {info_dir}")
    for json_file in json_files:
        print(f"   - {json_file.name}")
else:
    json_files = []
    print(f"\n3. Transcript Files: ✗ Directory not found")

# Check test files
if test_dir.exists():
    test_files = list(test_dir.glob("*.wav"))
    print(f"\n4. Test Files:")
    print(f"   Found {len(test_files)} test audio files")
    if len(test_files) > 0:
        print(f"   Sample files: {[f.name for f in test_files[:3]]}")
else:
    test_files = []
    print(f"\n4. Test Files: (optional - not required for training)")

# Summary
print(f"\n" + "=" * 70)
if len(audio_files) > 0 and len(json_files) > 0:
    print("Dataset ready for training!")
    print(f"  - {len(audio_files)} audio files ready")
    print(f"  - {len(json_files)} transcript files ready")
    if len(test_files) > 0:
        print(f"  - {len(test_files)} test files ready")
else:
    print("Dataset not ready:")
    if len(audio_files) == 0:
        print("  - No audio files found in dataset/audio_files/")
    if len(json_files) == 0:
        print("  - No JSON transcript files found in dataset/info/")
print("=" * 70)


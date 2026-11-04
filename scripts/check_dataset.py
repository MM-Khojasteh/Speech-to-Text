"""
Check dataset status and verify setup.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_dataset():
    """Check dataset status."""
    print("=" * 70)
    print("Dataset Status Check")
    print("=" * 70)
    
    # Check audio files directory
    audio_dir = Path("dataset/audio_files")
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.wav"))
        print(f"\nAudio Files Directory: {audio_dir}")
        print(f"  Exists: Yes")
        print(f"  WAV files found: {len(audio_files)}")
        if len(audio_files) > 0:
            print(f"  Sample files:")
            for f in audio_files[:5]:
                print(f"    - {f.name}")
    else:
        print(f"\nAudio Files Directory: {audio_dir}")
        print(f"  Exists: No")
    
    # Check transcript JSON files
    print(f"\nTranscript Files:")

    transcript_files = sorted(Path("dataset/info").glob("*.json"))
    if not transcript_files:
        print("  No JSON transcript files found in dataset/info/")
    for transcript_file in transcript_files:
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"  {transcript_file.name}: {len(data)} entries")
            if len(data) > 0:
                sample = data[0]
                print(f"    Sample keys: {list(sample.keys())}")
                for key in ('recording_file', 'audio_file_path'):
                    if key in sample:
                        print(f"    Sample {key}: {sample[key]}")
        except Exception as e:
            print(f"  {transcript_file.name}: Error reading - {e}")
    
    # Check test audio
    test_dir = Path("test_audio")
    if test_dir.exists():
        test_files = list(test_dir.glob("*.wav"))
        print(f"\nTest Audio Directory: {test_dir}")
        print(f"  Exists: Yes")
        print(f"  WAV files found: {len(test_files)}")
    else:
        print(f"\nTest Audio Directory: {test_dir}")
        print(f"  Exists: No")
    
    print("\n" + "=" * 70)
    
    # Try to load dataset
    try:
        from src.data.custom_dataset import CustomAudioDataset
        print("\nAttempting to load dataset...")
        dataset = CustomAudioDataset(
            audio_files_dir="dataset/audio_files",
            transcripts_dir="dataset/info",
            sample_rate=16000
        )
        print(f"✓ Dataset loaded successfully!")
        print(f"  Total samples: {len(dataset)}")
    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        print("\nTo fix:")
        print("  1. Ensure audio files (.wav) are in dataset/audio_files/")
        print("  2. Ensure JSON transcript files are in dataset/info/")
        print("  3. Verify filenames in JSON match audio filenames exactly")
    
    print("=" * 70)

if __name__ == "__main__":
    check_dataset()


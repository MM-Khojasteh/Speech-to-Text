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
    
    # Check JSON files
    json1 = Path("dataset/info/postgres_laravel_calls.json")
    json2 = Path("dataset/info/postgres_public_call_transcripts.json")
    
    print(f"\nTranscript Files:")
    
    if json1.exists():
        try:
            with open(json1, 'r', encoding='utf-8') as f:
                data1 = json.load(f)
            print(f"  postgres_laravel_calls.json: {len(data1)} entries")
            if len(data1) > 0:
                sample = data1[0]
                print(f"    Sample keys: {list(sample.keys())}")
                if 'recording_file' in sample:
                    print(f"    Sample recording_file: {sample['recording_file']}")
        except Exception as e:
            print(f"  postgres_laravel_calls.json: Error reading - {e}")
    else:
        print(f"  postgres_laravel_calls.json: Not found")
    
    if json2.exists():
        try:
            with open(json2, 'r', encoding='utf-8') as f:
                data2 = json.load(f)
            print(f"  postgres_public_call_transcripts.json: {len(data2)} entries")
            if len(data2) > 0:
                sample = data2[0]
                print(f"    Sample keys: {list(sample.keys())}")
                if 'audio_file_path' in sample:
                    print(f"    Sample audio_file_path: {sample['audio_file_path']}")
        except Exception as e:
            print(f"  postgres_public_call_transcripts.json: Error reading - {e}")
    else:
        print(f"  postgres_public_call_transcripts.json: Not found")
    
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


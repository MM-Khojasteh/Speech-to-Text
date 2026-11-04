import os
from pathlib import Path

# Work from the repository root regardless of where the script is invoked
os.chdir(Path(__file__).resolve().parent.parent)
import json

audio_dir = Path('dataset/audio_files')
transcript_files = sorted(Path('dataset/info').glob('*.json'))

print(f"Working directory: {os.getcwd()}")
print(f"Audio dir exists: {audio_dir.exists()}")
if audio_dir.exists():
    files = list(audio_dir.glob('*.wav'))
    print(f"WAV files: {len(files)}")
    if files:
        print(f"First file: {files[0].name}")

print(f"\nTranscript files found: {len(transcript_files)}")
for transcript_file in transcript_files:
    data = json.loads(transcript_file.read_text(encoding='utf-8'))
    print(f"{transcript_file.name}: {len(data)} entries")
    if data:
        for key in ('recording_file', 'audio_file_path'):
            if key in data[0]:
                print(f"  Sample {key}: {data[0][key]}")

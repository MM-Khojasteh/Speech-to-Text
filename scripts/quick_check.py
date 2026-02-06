import os
os.chdir(r'D:\Winter 2026\Speech-to-Text')
from pathlib import Path
import json

audio_dir = Path('dataset/audio_files')
json1 = Path('dataset/info/postgres_laravel_calls.json')
json2 = Path('dataset/info/postgres_public_call_transcripts.json')

print(f"Working directory: {os.getcwd()}")
print(f"Audio dir exists: {audio_dir.exists()}")
if audio_dir.exists():
    files = list(audio_dir.glob('*.wav'))
    print(f"WAV files: {len(files)}")
    if files:
        print(f"First file: {files[0].name}")

print(f"\nJSON1 exists: {json1.exists()}")
if json1.exists():
    data1 = json.load(open(json1, 'r', encoding='utf-8'))
    print(f"JSON1 entries: {len(data1)}")
    if data1:
        print(f"Sample recording_file: {data1[0].get('recording_file', 'N/A')}")

print(f"\nJSON2 exists: {json2.exists()}")
if json2.exists():
    data2 = json.load(open(json2, 'r', encoding='utf-8'))
    print(f"JSON2 entries: {len(data2)}")
    if data2:
        print(f"Sample audio_file_path: {data2[0].get('audio_file_path', 'N/A')}")


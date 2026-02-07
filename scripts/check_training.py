"""Quick script to check training status."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathlib import Path
import os
import time

print("=" * 70)
print("Training Status Check")
print("=" * 70)

# Check for checkpoints
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if checkpoints:
        print(f"\nCheckpoints found: {len(checkpoints)}")
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"Latest checkpoint: {latest.name}")
        print(f"Modified: {time.ctime(latest.stat().st_mtime)}")
    else:
        print("\nNo checkpoints yet (training may be starting...)")
else:
    print("\nCheckpoints directory not created yet")

# Check for logs
log_dir = Path("logs")
if log_dir.exists():
    log_files = list(log_dir.rglob("*"))
    if log_files:
        print(f"\nLog files found: {len(log_files)}")
        print("TensorBoard logs are being written")
    else:
        print("\nLog directory exists but no files yet")
else:
    print("\nLogs directory not created yet")

# Check dataset
audio_dir = Path("dataset/audio_files")
if audio_dir.exists():
    audio_files = list(audio_dir.glob("*.wav"))
    print(f"\nDataset: {len(audio_files)} audio files ready")
else:
    print("\nDataset: Audio directory not found")

print("\n" + "=" * 70)
print("To monitor training in real-time:")
print("1. Check console output where training was started")
print("2. Run: tensorboard --logdir logs")
print("3. Open http://localhost:6006 in browser")
print("=" * 70)




# Dataset Guide: Custom Dataset Setup

This guide explains how to set up and use custom datasets with the speech recognition system.

## Dataset Structure

### Required Directory Structure

```
project_root/
├── dataset/
│   ├── audio_files/          # Training and validation audio files
│   │   ├── file1.wav
│   │   ├── file2.wav
│   │   └── ...
│   └── info/                  # Transcript JSON files
│       ├── postgres_laravel_calls.json
│       └── postgres_public_call_transcripts.json
└── test_audio/                # Test set audio files (separate)
    ├── test1.wav
    ├── test2.wav
    └── ...
```

## Dataset Split

### Train/Validation Split (80/20)
- **Source**: All `.wav` files in `dataset/audio_files/`
- **Split**: Automatically split 80% for training, 20% for validation
- **Random Seed**: Fixed seed (42) for reproducibility
- **No Overlap**: Train and validation sets are completely separate

### Test Set
- **Source**: All `.wav` files in `test_audio/` directory
- **Separate**: Completely independent from train/validation sets
- **Usage**: Only used for final evaluation, never during training
- **No Overlap**: Guaranteed no overlap with train/val sets

## Transcript JSON Format

The system supports two JSON file formats:

### Format 1: postgres_laravel_calls.json

```json
[
  {
    "uniqueid": "1234567890.1234567",
    "duration": 120,
    "recording_file": "file1.wav",
    "transcript": "سلام دنیا",
    "voip_recording_url": "https://..."
  },
  {
    "uniqueid": "1234567891.1234568",
    "recording_file": "file2.wav",
    "transcript": "Hello world",
    ...
  }
]
```

**Key Field**: `recording_file` - must match audio filename

### Format 2: postgres_public_call_transcripts.json

```json
[
  {
    "uniqueid": "1234567890.1234567",
    "audio_duration_seconds": 120.5,
    "audio_file_path": "file1.wav",
    "transcript": "سلام دنیا"
  },
  {
    "uniqueid": "1234567891.1234568",
    "audio_file_path": "file2.wav",
    "transcript": "Hello world",
    ...
  }
]
```

**Key Field**: `audio_file_path` - must match audio filename

## Matching Process

1. **Scan Audio Files**: System scans `dataset/audio_files/` for all `.wav` files
2. **Load Transcripts**: Loads both JSON files from `dataset/info/`
3. **Match by Filename**: For each audio file:
   - First searches in `postgres_laravel_calls.json` using `recording_file` field
   - If not found, searches in `postgres_public_call_transcripts.json` using `audio_file_path` field
   - Extracts `transcript` field
4. **Filter**: Only audio files with matching transcripts are included
5. **Split**: Matched files are split 80/20 for train/validation

## Persian/Farsi Support

The system fully supports Persian characters:

- **Persian Alphabet**: ا, ب, پ, ت, ث, ج, چ, ح, خ, د, ذ, ر, ز, ژ, س, ش, ص, ض, ط, ظ, ع, غ, ف, ق, ک, گ, ل, م, ن, و, ه, ی
- **Persian Numbers**: ۰, ۱, ۲, ۳, ۴, ۵, ۶, ۷, ۸, ۹
- **Mixed Content**: Supports both English and Persian in the same transcript
- **Vocabulary Size**: ~97 characters (English + Persian + numbers + punctuation)

Enable in config:
```yaml
data:
  include_persian: true
```

## Configuration

Update `configs/default_config.yaml`:

```yaml
data:
  dataset_type: "custom"              # Use custom dataset
  audio_files_dir: "dataset/audio_files"  # Training/validation files
  transcripts_dir: "dataset/info"         # JSON transcript files
  test_audio_dir: "test_audio"            # Test set directory
  train_split: 0.8                       # 80% train, 20% validation
  include_persian: true                  # Enable Persian support
```

## Usage

### Training

```bash
python scripts/train.py --config configs/default_config.yaml
```

This will:
- Load audio files from `dataset/audio_files/`
- Match with transcripts from JSON files
- Split 80/20 for train/validation
- Optionally load test set from `test_audio/` (if configured)

### Evaluation

Evaluate on test set:
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/default_config.yaml --split test
```

Evaluate on validation set:
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/default_config.yaml --split val
```

## Troubleshooting

### No Matching Transcripts

**Problem**: "No matching audio-transcript pairs found"

**Solutions**:
1. Check that JSON files are in `dataset/info/` directory
2. Verify filenames in JSON match audio filenames exactly
3. Ensure JSON files have `transcript` field
4. Check that audio files are `.wav` format

### Test Set Not Loading

**Problem**: Test loader is None

**Solutions**:
1. Verify `test_audio_dir` is set in config
2. Check that `test_audio/` directory exists
3. Ensure there are `.wav` files in `test_audio/`
4. Verify test audio files have matching transcripts in JSON files

### Empty Transcripts

**Problem**: Some audio files have empty transcripts

**Solutions**:
1. System automatically skips empty transcripts
2. Check JSON files for empty `transcript` fields
3. Ensure transcripts are not just whitespace

### Persian Characters Not Working

**Problem**: Persian text not encoding correctly

**Solutions**:
1. Ensure `include_persian: true` in config
2. Check that transcripts contain valid Persian characters
3. Verify JSON files are saved with UTF-8 encoding

## Best Practices

1. **Organize Files**: Keep train/val and test sets completely separate
2. **Consistent Naming**: Use consistent filename patterns
3. **UTF-8 Encoding**: Save JSON files with UTF-8 encoding for Persian support
4. **Quality Control**: Verify all audio files have matching transcripts
5. **Test Set**: Keep test set completely separate - never use it during training
6. **Backup**: Keep backups of your dataset structure

## Example Workflow

1. **Prepare Audio Files**:
   ```bash
   # Place training/validation files
   cp audio/*.wav dataset/audio_files/
   
   # Place test files (separate)
   cp test_audio/*.wav test_audio/
   ```

2. **Prepare Transcripts**:
   ```bash
   # Place JSON files
   cp transcripts/*.json dataset/info/
   ```

3. **Verify Matching**:
   ```bash
   # Run test script to verify
   python scripts/test_custom_dataset.py
   ```

4. **Train Model**:
   ```bash
   python scripts/train.py --config configs/default_config.yaml
   ```

5. **Evaluate**:
   ```bash
   python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/default_config.yaml --split test
   ```

## Data Statistics

After loading, the system will display:
- Number of matched audio-transcript pairs
- Train/validation split sizes
- Test set size (if available)
- Vocabulary size (including Persian if enabled)

Example output:
```
Matched 1000 audio files with transcripts
Dataset split: 800 train, 200 validation
Test dataset: 100 samples from test_audio
Vocabulary size: 97 (includes blank token)
```


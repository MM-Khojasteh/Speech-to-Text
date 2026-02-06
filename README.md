# Enhanced Speech-to-Text PyTorch Project

A production-ready, modular implementation of speech recognition using PyTorch with advanced features including attention mechanisms, mixed precision training, and comprehensive evaluation metrics.

## Features

### Model Architecture
- **Enhanced CNN Feature Extraction**: Residual CNN blocks with optional squeeze-and-excitation
- **Multi-Head Self-Attention**: Attention mechanisms for better sequence modeling
- **Flexible Backbone**: Choose between Bidirectional GRU or Transformer encoder
- **Deep Architecture**: Configurable depth for both CNN and RNN/Transformer layers

### Training Features
- **Automatic Mixed Precision (AMP)**: Faster training with reduced memory usage
- **Gradient Accumulation**: Support for effective larger batch sizes
- **Advanced Optimizers**: AdamW with proper weight decay
- **Learning Rate Scheduling**: OneCycle, Cosine, or Step schedulers
- **Model Checkpointing**: Save best model and periodic checkpoints
- **Early Stopping**: Prevent overfitting
- **TensorBoard Logging**: Track metrics, losses, and learning rates

### Data Processing
- **SpecAugment**: Time and frequency masking for data augmentation
- **Efficient Batching**: Dynamic padding and collation
- **Multi-worker Loading**: Fast data loading with pin memory

### Evaluation
- **Comprehensive Metrics**: WER, CER, SER, Word/Character Accuracy, BLEU, ROUGE-L, Precision/Recall/F1
- **Multiple Evaluation Levels**: Word-level and character-level metrics
- **Greedy Decoding**: Fast inference
- **Beam Search Decoding**: More accurate but slower (optional)

### Dataset Support
- **LibriSpeech**: Standard English speech dataset
- **Custom Datasets**: Support for custom audio files with JSON transcripts
- **Persian/Farsi Support**: Full support for Persian characters and mixed English/Persian content
- **Three-Way Split**: Train (80%) / Validation (20%) / Test (separate directory)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Speech-to-Text
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have CUDA-compatible PyTorch installed for GPU training:
```bash
# Visit https://pytorch.org/get-started/locally/ for installation instructions
```

## Project Structure

```
Speech-to-Text/
├── src/
│   ├── models/          # Model architectures
│   ├── data/            # Data loading and processing (custom dataset support)
│   ├── training/        # Training and validation
│   ├── metrics/         # Comprehensive ASR metrics
│   ├── inference/       # Decoding functions
│   └── utils/           # Configuration and logging
├── configs/             # Configuration files
├── scripts/             # Training, evaluation, inference scripts
├── dataset/             # Custom dataset directory
│   ├── audio_files/     # Training/validation audio files
│   └── info/            # Transcript JSON files
├── test_audio/          # Test set audio files
├── checkpoints/          # Saved model checkpoints
├── logs/                # TensorBoard logs
└── requirements.txt      # Python dependencies
```

## Quick Start

### 1. Prepare Data

#### Option A: Using LibriSpeech Dataset

Download the LibriSpeech dataset and update the paths in `configs/default_config.yaml`:

```yaml
data:
  dataset_type: "librispeech"
  train_root: "/path/to/librispeech"
  test_root: "/path/to/librispeech"
  train_url: "train-clean-100"
  test_url: "test-clean"
```

#### Option B: Using Custom Dataset

Organize your audio files and transcripts:

```
dataset/
├── audio_files/          # Training/validation audio files (.wav)
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
├── info/                 # Transcript JSON files
│   ├── postgres_laravel_calls.json
│   └── postgres_public_call_transcripts.json
test_audio/              # Test audio files (.wav)
├── test1.wav
├── test2.wav
└── ...
```

Update `configs/default_config.yaml`:

```yaml
data:
  dataset_type: "custom"
  audio_files_dir: "dataset/audio_files"
  transcripts_dir: "dataset/info"
  test_audio_dir: "test_audio"
  train_split: 0.8  # 80% train, 20% validation
  include_persian: true  # Enable Persian character support
```

### 2. Configure Training

Edit `configs/default_config.yaml` to adjust hyperparameters:

```yaml
model:
  n_cnn_layers: 3
  n_rnn_layers: 5
  rnn_dim: 512
  use_attention: true
  use_transformer: false

training:
  batch_size: 10
  epochs: 10
  learning_rate: 0.0005
  use_amp: true
  gradient_accumulation_steps: 1
```

### 3. Train Model

```bash
python scripts/train.py --config configs/default_config.yaml
```

To resume from a checkpoint:
```bash
python scripts/train.py --config configs/default_config.yaml --resume checkpoints/checkpoint_epoch_5.pth
```

### 4. Evaluate Model

Evaluate on test set:
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/default_config.yaml --split test
```

Evaluate on validation set:
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/default_config.yaml --split val
```

Evaluate on training set:
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/default_config.yaml --split train
```

### 5. Run Inference

```bash
python scripts/inference.py --audio path/to/audio.wav --checkpoint checkpoints/best_model.pth --config configs/default_config.yaml
```

## Configuration

The configuration file (`configs/default_config.yaml`) contains all hyperparameters:

### Model Configuration
- `n_cnn_layers`: Number of residual CNN layers
- `n_rnn_layers`: Number of RNN/Transformer layers
- `rnn_dim`: Hidden dimension for RNN/Transformer
- `use_attention`: Enable multi-head self-attention
- `use_transformer`: Use Transformer instead of RNN
- `use_se`: Enable squeeze-and-excitation in CNN blocks

### Training Configuration
- `batch_size`: Batch size
- `epochs`: Number of training epochs
- `learning_rate`: Initial learning rate
- `use_amp`: Enable Automatic Mixed Precision
- `gradient_accumulation_steps`: Steps for gradient accumulation
- `early_stopping_patience`: Early stopping patience (None to disable)

### Data Configuration
- `dataset_type`: "librispeech" or "custom"
- `train_root`, `test_root`: LibriSpeech dataset root directories (for librispeech type)
- `audio_files_dir`: Custom dataset audio files directory (for custom type)
- `transcripts_dir`: Custom dataset transcripts JSON directory (for custom type)
- `test_audio_dir`: Separate test set directory (for custom type)
- `train_split`: Train/validation split ratio (default: 0.8 for 80/20)
- `include_persian`: Enable Persian character support (default: true)
- `sample_rate`: Audio sample rate (default: 16000)
- `n_mels`: Number of mel filter banks (default: 128)
- `freq_mask_param`, `time_mask_param`: SpecAugment parameters

## Model Architecture

The model follows this architecture:

```
Audio Waveform
    ↓
MelSpectrogram
    ↓
CNN Feature Extraction (Residual Blocks)
    ↓
Multi-Head Self-Attention (Optional)
    ↓
Bidirectional RNN / Transformer Encoder
    ↓
Linear Classifier
    ↓
CTC Output
```

### Key Components

1. **CNN Feature Extraction**: 
   - Initial 2D convolution
   - Multiple residual CNN blocks with layer normalization
   - Optional squeeze-and-excitation for channel attention

2. **Attention Layer**:
   - Multi-head self-attention
   - Helps model focus on relevant parts of the sequence

3. **Sequence Modeling**:
   - Bidirectional GRU layers (default)
   - Or Transformer encoder layers (optional)

4. **Output**:
   - Linear classifier for CTC decoding

## Training Tips

1. **Start Small**: Begin with a smaller model and dataset subset to verify setup
2. **Monitor Metrics**: Use TensorBoard to track all metrics (WER, CER, BLEU, F1, etc.)
3. **Adjust Learning Rate**: If loss doesn't decrease, try lower learning rates
4. **Use AMP**: Enable AMP for faster training and lower memory usage
5. **Gradient Accumulation**: Use if you want larger effective batch sizes
6. **Early Stopping**: Set patience to prevent overfitting
7. **Dataset Split**: Ensure train/val/test sets are properly separated (no overlap)
8. **Persian Support**: If using Persian text, ensure `include_persian: true` in config

## Monitoring Training

View TensorBoard logs:
```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006` in your browser.

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Error Rates
- **WER (Word Error Rate)**: Percentage of words incorrectly predicted (lower is better)
- **CER (Character Error Rate)**: Percentage of characters incorrectly predicted (lower is better)
- **SER (Sentence Error Rate)**: Percentage of sentences with any error (lower is better)

### Accuracy Metrics
- **Word Accuracy**: Percentage of words correctly recognized (higher is better)
- **Character Accuracy**: Percentage of characters correctly recognized (higher is better)
- **Exact Match Rate**: Percentage of perfect sentence matches (higher is better)

### N-gram Metrics
- **BLEU Score**: N-gram overlap between reference and hypothesis (higher is better)
- **ROUGE-L Score**: Longest common subsequence similarity (higher is better)

### Precision/Recall/F1
- **Word-level**: Precision, Recall, and F1 at word level
- **Character-level**: Precision, Recall, and F1 at character level

All metrics are automatically calculated during validation and evaluation. See `METRICS_DOCUMENTATION.md` for detailed information.

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Enable gradient accumulation
- Use AMP (already enabled by default)
- Reduce model size (rnn_dim, n_rnn_layers)

### Poor Performance
- Train for more epochs
- Increase model capacity
- Adjust learning rate
- Use beam search decoding for inference

### Data Loading Issues
- Check dataset paths in config
- Ensure audio files are accessible
- Verify sample rate matches config

## Citation

If you use this code, please cite:

```bibtex
@software{speech_recognition_2024,
  title={Enhanced Speech-to-Text PyTorch Project},
  author={Your Name},
  year={2024}
}
```

## License

[Specify your license here]

## Documentation

- **README.md**: This file - main project documentation
- **TRAINING_GUIDE.md**: Detailed training instructions and tips
- **DATASET_GUIDE.md**: Custom dataset setup and usage guide
- **METRICS_DOCUMENTATION.md**: Comprehensive evaluation metrics documentation
- **ARCHITECTURE_SUMMARY.md**: Model architecture details

## Acknowledgments

- LibriSpeech dataset
- DeepSpeech 2 architecture inspiration
- PyTorch and torchaudio teams


# Training Status

## Training Started

The model training has been initiated with your custom dataset.

## What's Happening

1. **Dataset Loading**: 
   - Loading audio files from `dataset/audio_files/`
   - Matching with transcripts from `dataset/info/`
   - Splitting 80/20 for train/validation
   - Loading test set from `test_audio/` (if available)

2. **Model Initialization**:
   - Creating model with enhanced architecture
   - Vocabulary size determined from TextTransform (includes Persian)
   - Moving model to GPU

3. **Training Process**:
   - Training with batch size 4
   - Gradient accumulation: 4 steps (effective batch size 16)
   - Automatic Mixed Precision (AMP) enabled
   - Validation every epoch
   - Saving checkpoints every 5 epochs

## Monitor Training

### TensorBoard
Open a new terminal and run:
```bash
tensorboard --logdir logs
```
Then open `http://localhost:6006` in your browser to see:
- Training/validation loss
- All evaluation metrics (WER, CER, BLEU, F1, etc.)
- Learning rate schedule
- Model performance over time

### Check Logs
Training output is also printed to console. You can check:
- Console output for real-time progress
- `logs/` directory for TensorBoard logs
- `checkpoints/` directory for saved models

## Expected Output

You should see output like:
```
Using GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Vocabulary size: 97 (includes blank token)
Loading datasets...
Using custom dataset from dataset/audio_files
Matched X audio files with transcripts
Dataset split: Y train, Z validation
Test dataset: W samples from test_audio
Creating model...
Model parameters: 22,823,653
Train batches: ...
Validation batches: ...
Starting training...
Epoch 1/10
  Training...
  Validating...
  Loss: X.XXXXXX
  WER: X.XXXX
  ...
```

## Training Configuration

- **Epochs**: 10 (with early stopping after 3 epochs without improvement)
- **Batch Size**: 4
- **Effective Batch Size**: 16 (with gradient accumulation)
- **Learning Rate**: 0.0005
- **Optimizer**: AdamW
- **Scheduler**: OneCycleLR
- **Early Stopping**: Enabled (patience: 3)

## Checkpoints

- **Best Model**: `checkpoints/best_model.pth` (saved when validation loss improves)
- **Periodic Checkpoints**: `checkpoints/checkpoint_epoch_X.pth` (every 5 epochs)

## After Training

Once training completes (or you stop it), evaluate on test set:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/default_config.yaml --split test
```

## Tips

1. **Monitor GPU Memory**: Should stay under 4GB
2. **Watch Loss**: Should decrease steadily
3. **Check Metrics**: WER/CER should improve over epochs
4. **Early Stopping**: Will stop automatically if no improvement
5. **Resume Training**: If interrupted, resume with `--resume` flag

## Troubleshooting

If training fails:
1. Check that dataset directories exist and contain files
2. Verify JSON transcript files are properly formatted
3. Ensure GPU has enough memory (reduce batch_size if needed)
4. Check console output for specific error messages




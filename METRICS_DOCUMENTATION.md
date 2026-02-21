# Comprehensive ASR Evaluation Metrics

This document describes all the evaluation metrics available for the audio-to-transcript (ASR) model.

## Available Metrics

### 1. Error Rates

#### Word Error Rate (WER)
- **Formula**: `WER = (Substitutions + Insertions + Deletions) / Number of words in reference`
- **Range**: 0.0 (perfect) to 1.0+ (higher is worse)
- **Usage**: Standard ASR metric measuring word-level accuracy

#### Character Error Rate (CER)
- **Formula**: `CER = (Substitutions + Insertions + Deletions) / Number of characters in reference`
- **Range**: 0.0 (perfect) to 1.0+ (higher is worse)
- **Usage**: Measures character-level accuracy, useful for languages with complex scripts

#### Sentence Error Rate (SER)
- **Formula**: `SER = (Number of sentences with errors) / (Total sentences)`
- **Range**: 0.0 (perfect) to 1.0 (all sentences have errors)
- **Usage**: Measures how many sentences are completely correct

### 2. Accuracy Metrics

#### Word Accuracy
- **Formula**: `Word Accuracy = 1 - WER`
- **Range**: 0.0 to 1.0 (1.0 is perfect)
- **Usage**: Percentage of words correctly recognized

#### Character Accuracy
- **Formula**: `Character Accuracy = 1 - CER`
- **Range**: 0.0 to 1.0 (1.0 is perfect)
- **Usage**: Percentage of characters correctly recognized

#### Exact Match Rate (EMR)
- **Formula**: `EMR = (Perfect matches) / (Total sentences)`
- **Range**: 0.0 to 1.0 (1.0 is perfect)
- **Usage**: Percentage of sentences that are completely correct

### 3. N-gram Overlap Metrics

#### BLEU Score
- **Description**: Measures n-gram overlap between reference and hypothesis
- **Range**: 0.0 to 1.0 (1.0 is perfect)
- **Usage**: Common in machine translation, measures overall quality
- **Note**: Uses 1-4 gram precision with brevity penalty

#### ROUGE-L Score
- **Description**: Measures longest common subsequence (LCS) between reference and hypothesis
- **Range**: 0.0 to 1.0 (1.0 is perfect)
- **Usage**: Measures sentence-level similarity, useful for longer sequences
- **Note**: Can be calculated at word or character level

### 4. Precision, Recall, F1

#### Word-level Metrics
- **Precision**: `TP / (TP + FP)` - Percentage of predicted words that are correct
- **Recall**: `TP / (TP + FN)` - Percentage of reference words that were predicted
- **F1**: Harmonic mean of precision and recall
- **Usage**: Measures word-level classification performance

#### Character-level Metrics
- **Precision**: `TP / (TP + FP)` - Percentage of predicted characters that are correct
- **Recall**: `TP / (TP + FN)` - Percentage of reference characters that were predicted
- **F1**: Harmonic mean of precision and recall
- **Usage**: Measures character-level classification performance

## Usage

### During Training

All metrics are automatically calculated during validation. They are:
- Displayed in the console
- Logged to TensorBoard
- Saved in checkpoint files

### During Evaluation

Run the evaluation script:
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config configs/default_config.yaml
```

This will display all metrics in a formatted output.

### Programmatic Usage

```python
from src.metrics.asr_metrics import calculate_all_metrics, calculate_batch_metrics

# Single pair
metrics = calculate_all_metrics(reference="hello world", hypothesis="hello world")
print(f"WER: {metrics['wer']:.4f}")
print(f"BLEU: {metrics['bleu']:.4f}")

# Batch
references = ["hello world", "good morning"]
hypotheses = ["hello world", "good evening"]
batch_metrics = calculate_batch_metrics(references, hypotheses)
print(f"Average WER: {batch_metrics['wer']:.4f}")
print(f"SER: {batch_metrics['ser']:.4f}")
```

## Metric Interpretation

### For Model Development
- **WER/CER**: Primary metrics for ASR - lower is better
- **Word/Character Accuracy**: Easier to interpret - higher is better
- **SER/EMR**: Useful for understanding sentence-level performance

### For Quality Assessment
- **BLEU**: Overall quality indicator - higher is better
- **ROUGE-L**: Captures semantic similarity - higher is better
- **F1 Scores**: Balanced view of precision and recall

### For Error Analysis
- **Precision**: How many predictions are correct (fewer false positives)
- **Recall**: How many references were captured (fewer false negatives)
- **F1**: Balanced metric combining both

## Metric Selection Guide

| Use Case | Recommended Metrics |
|----------|---------------------|
| General ASR evaluation | WER, CER, Word Accuracy |
| Sentence-level quality | SER, EMR, ROUGE-L |
| Overall quality | BLEU, ROUGE-L |
| Error analysis | Precision, Recall, F1 (word/char level) |
| Production monitoring | WER, Word Accuracy, EMR |

## Notes

- **Empty Predictions**: Models producing empty predictions are penalized with 100% WER/CER for those samples
- **Empty References**: Samples with empty references are excluded from metrics (data quality issue, not model issue)
- Metrics can be case-sensitive or case-insensitive (configurable)
- Persian and English characters are fully supported
- Metrics are calculated efficiently using optimized algorithms

### Empty Prediction Handling

During validation and evaluation, if the model produces an empty prediction for a non-empty reference:
- The sample is counted as **100% error** (WER=1.0, CER=1.0, all accuracies=0)
- This ensures metrics reflect actual model performance, not just "easy" samples
- Prevents overly optimistic metrics when model collapses to blank predictions


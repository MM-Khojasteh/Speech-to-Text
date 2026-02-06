"""
Test script for comprehensive ASR metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.asr_metrics import calculate_all_metrics, calculate_batch_metrics

def test_metrics():
    """Test all metrics with sample data."""
    print("=" * 70)
    print("Testing Comprehensive ASR Metrics")
    print("=" * 70)
    
    # Test case 1: Perfect match
    print("\n1. Perfect Match Test:")
    ref1 = "hello world"
    hyp1 = "hello world"
    metrics1 = calculate_all_metrics(ref1, hyp1)
    print(f"  Reference: {ref1}")
    print(f"  Hypothesis: {hyp1}")
    print(f"  WER: {metrics1['wer']:.4f} (should be 0.0)")
    print(f"  CER: {metrics1['cer']:.4f} (should be 0.0)")
    print(f"  Word Accuracy: {metrics1['word_accuracy']:.4f} (should be 1.0)")
    print(f"  BLEU: {metrics1['bleu']:.4f} (should be 1.0)")
    print(f"  ROUGE-L: {metrics1['rouge_l']:.4f} (should be 1.0)")
    
    # Test case 2: Partial match
    print("\n2. Partial Match Test:")
    ref2 = "hello world how are you"
    hyp2 = "hello world how you"
    metrics2 = calculate_all_metrics(ref2, hyp2)
    print(f"  Reference: {ref2}")
    print(f"  Hypothesis: {hyp2}")
    print(f"  WER: {metrics2['wer']:.4f}")
    print(f"  CER: {metrics2['cer']:.4f}")
    print(f"  Word Accuracy: {metrics2['word_accuracy']:.4f}")
    print(f"  Word F1: {metrics2['word_f1']:.4f}")
    print(f"  BLEU: {metrics2['bleu']:.4f}")
    
    # Test case 3: Batch metrics
    print("\n3. Batch Metrics Test:")
    refs = ["hello world", "good morning", "how are you"]
    hyps = ["hello world", "good evening", "how you"]
    batch_metrics = calculate_batch_metrics(refs, hyps)
    print(f"  References: {refs}")
    print(f"  Hypotheses: {hyps}")
    print(f"  Average WER: {batch_metrics['wer']:.4f}")
    print(f"  Average CER: {batch_metrics['cer']:.4f}")
    print(f"  SER: {batch_metrics['ser']:.4f}")
    print(f"  Exact Match Rate: {batch_metrics['exact_match_rate']:.4f}")
    print(f"  Average BLEU: {batch_metrics['bleu']:.4f}")
    print(f"  Average Word F1: {batch_metrics['word_f1']:.4f}")
    
    print("\n" + "=" * 70)
    print("All metrics calculated successfully!")
    print("=" * 70)

if __name__ == "__main__":
    test_metrics()


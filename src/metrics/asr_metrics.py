"""
ASR (Automatic Speech Recognition) metrics calculation.
Includes comprehensive metrics for evaluating audio-to-transcript models:
- WER, CER (error rates)
- Word/Character Accuracy
- Sentence Error Rate
- BLEU score
- ROUGE-L score
- Precision, Recall, F1 (word and character level)
- Exact Match Rate
"""

from typing import List, Union, Optional, Dict, Tuple
from collections import Counter
import math


def levenshtein_distance(ref: Union[str, List], hyp: Union[str, List]) -> int:
    """
    Calculate the Levenshtein distance between two sequences.
    
    This is the minimum number of single-character edits (insertions, deletions,
    or substitutions) required to change one sequence into another.
    
    Args:
        ref: Reference sequence (string or list)
        hyp: Hypothesis sequence (string or list)
        
    Returns:
        Levenshtein distance (integer)
    """
    m, n = len(ref), len(hyp)
    
    # Early returns for common cases
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m
    
    # Optimize: use the shorter sequence as the reference for space efficiency
    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m
    
    # Initialize previous row (for the shorter sequence)
    previous_row = list(range(n + 1))
    
    # Dynamic programming: fill the distance matrix row by row
    for i, c1 in enumerate(ref):
        current_row = [i + 1]  # First element is always i+1
        
        for j, c2 in enumerate(hyp):
            # Calculate costs for three operations:
            # 1. Insertion: previous_row[j + 1] + 1
            # 2. Deletion: current_row[j] + 1
            # 3. Substitution: previous_row[j] + (1 if c1 != c2 else 0)
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (1 if c1 != c2 else 0)
            
            current_row.append(min(insertions, deletions, substitutions))
        
        previous_row = current_row
    
    return previous_row[-1]


def normalize_text(
    text: str, 
    ignore_case: bool = False, 
    remove_space: bool = False
) -> str:
    """
    Normalize text based on the specified conditions.
    
    Args:
        text: Input text string
        ignore_case: If True, convert to lowercase
        remove_space: If True, remove all spaces
        
    Returns:
        Normalized text string
    """
    if ignore_case:
        text = text.lower()
    if remove_space:
        text = ''.join(text.split())
    return text


def calculate_errors(
    reference: Union[str, List],
    hypothesis: Union[str, List],
    ignore_case: bool = False,
    remove_space: bool = False,
    delimiter: Optional[str] = None
) -> tuple[float, int]:
    """
    Calculate edit distance and reference length based on the mode (word or char level).
    
    Args:
        reference: Reference sequence
        hypothesis: Hypothesis sequence
        ignore_case: If True, ignore case differences
        remove_space: If True, remove spaces (for character-level)
        delimiter: If provided, split sequences by this delimiter (for word-level)
        
    Returns:
        Tuple of (edit_distance, reference_length)
    """
    # Normalize inputs
    ref = normalize_text(reference, ignore_case, remove_space)
    hyp = normalize_text(hypothesis, ignore_case, remove_space)
    
    # Split by delimiter if provided (for word-level metrics)
    if delimiter:
        ref = ref.split(delimiter)
        hyp = hyp.split(delimiter)
    
    # Calculate edit distance
    edit_distance = levenshtein_distance(ref, hyp)
    ref_len = len(ref)
    
    return float(edit_distance), ref_len


def wer(
    reference: str,
    hypothesis: str,
    ignore_case: bool = False,
    delimiter: str = ' '
) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER = (Substitutions + Insertions + Deletions) / Number of words in reference
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        ignore_case: If True, ignore case differences
        delimiter: Delimiter for splitting words (default: space)
        
    Returns:
        Word Error Rate (float between 0 and 1, where 0 is perfect)
        
    Raises:
        ValueError: If reference has zero words
    """
    edit_distance, ref_len = calculate_errors(
        reference, hypothesis, ignore_case, False, delimiter
    )
    
    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")
    
    return edit_distance / ref_len


def cer(
    reference: str,
    hypothesis: str,
    ignore_case: bool = False,
    remove_space: bool = False
) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = (Substitutions + Insertions + Deletions) / Number of characters in reference
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        ignore_case: If True, ignore case differences
        remove_space: If True, remove spaces before calculation
        
    Returns:
        Character Error Rate (float between 0 and 1, where 0 is perfect)
        
    Raises:
        ValueError: If reference has zero length
    """
    edit_distance, ref_len = calculate_errors(
        reference, hypothesis, ignore_case, remove_space
    )
    
    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")
    
    return edit_distance / ref_len


def word_accuracy(
    reference: str,
    hypothesis: str,
    ignore_case: bool = False,
    delimiter: str = ' '
) -> float:
    """
    Calculate Word Accuracy.
    
    Word Accuracy = 1 - WER = (Correct Words) / (Total Words in Reference)
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        ignore_case: If True, ignore case differences
        delimiter: Delimiter for splitting words (default: space)
        
    Returns:
        Word Accuracy (float between 0 and 1, where 1 is perfect)
    """
    try:
        return 1.0 - wer(reference, hypothesis, ignore_case, delimiter)
    except ValueError:
        # If reference is empty, return 0
        return 0.0


def character_accuracy(
    reference: str,
    hypothesis: str,
    ignore_case: bool = False,
    remove_space: bool = False
) -> float:
    """
    Calculate Character Accuracy.
    
    Character Accuracy = 1 - CER = (Correct Characters) / (Total Characters in Reference)
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        ignore_case: If True, ignore case differences
        remove_space: If True, remove spaces before calculation
        
    Returns:
        Character Accuracy (float between 0 and 1, where 1 is perfect)
    """
    try:
        return 1.0 - cer(reference, hypothesis, ignore_case, remove_space)
    except ValueError:
        # If reference is empty, return 0
        return 0.0


def sentence_error_rate(
    references: List[str],
    hypotheses: List[str],
    ignore_case: bool = False
) -> float:
    """
    Calculate Sentence Error Rate (SER).
    
    SER = (Number of sentences with at least one error) / (Total number of sentences)
    
    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        ignore_case: If True, ignore case differences
        
    Returns:
        Sentence Error Rate (float between 0 and 1, where 0 is perfect)
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    
    if len(references) == 0:
        return 0.0
    
    errors = 0
    for ref, hyp in zip(references, hypotheses):
        ref_norm = normalize_text(ref, ignore_case, False)
        hyp_norm = normalize_text(hyp, ignore_case, False)
        if ref_norm != hyp_norm:
            errors += 1
    
    return errors / len(references)


def exact_match_rate(
    references: List[str],
    hypotheses: List[str],
    ignore_case: bool = False
) -> float:
    """
    Calculate Exact Match Rate (EMR).
    
    EMR = (Number of perfect matches) / (Total number of sentences)
    
    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        ignore_case: If True, ignore case differences
        
    Returns:
        Exact Match Rate (float between 0 and 1, where 1 is perfect)
    """
    return 1.0 - sentence_error_rate(references, hypotheses, ignore_case)


def _get_ngrams(text: Union[str, List], n: int) -> Counter:
    """Extract n-grams from text."""
    if isinstance(text, str):
        tokens = text.split()
    else:
        tokens = text
    
    if len(tokens) < n:
        return Counter()
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    
    return Counter(ngrams)


def bleu_score(
    reference: str,
    hypothesis: str,
    max_n: int = 4,
    ignore_case: bool = False
) -> float:
    """
    Calculate BLEU score (Bilingual Evaluation Understudy).
    
    BLEU measures n-gram overlap between reference and hypothesis.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        max_n: Maximum n-gram order (default: 4)
        ignore_case: If True, ignore case differences
        
    Returns:
        BLEU score (float between 0 and 1, where 1 is perfect)
    """
    ref_norm = normalize_text(reference, ignore_case, False)
    hyp_norm = normalize_text(hypothesis, ignore_case, False)
    
    ref_tokens = ref_norm.split()
    hyp_tokens = hyp_norm.split()
    
    if len(hyp_tokens) == 0:
        return 0.0
    
    # Brevity penalty
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    else:
        bp = 1.0
    
    # Calculate precision for each n-gram order
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        hyp_ngrams = _get_ngrams(hyp_tokens, n)
        
        if len(hyp_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count matches
        matches = sum((ref_ngrams & hyp_ngrams).values())
        precision = matches / len(hyp_ngrams)
        precisions.append(precision)
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    
    return bp * geometric_mean


def _longest_common_subsequence(ref: List, hyp: List) -> int:
    """Calculate length of longest common subsequence."""
    m, n = len(ref), len(hyp)
    if m == 0 or n == 0:
        return 0
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def rouge_l_score(
    reference: str,
    hypothesis: str,
    ignore_case: bool = False,
    level: str = 'word'
) -> float:
    """
    Calculate ROUGE-L score (Longest Common Subsequence).
    
    ROUGE-L measures the longest common subsequence between reference and hypothesis.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        ignore_case: If True, ignore case differences
        level: 'word' or 'character' level matching
        
    Returns:
        ROUGE-L score (float between 0 and 1, where 1 is perfect)
    """
    ref_norm = normalize_text(reference, ignore_case, False)
    hyp_norm = normalize_text(hypothesis, ignore_case, False)
    
    if level == 'word':
        ref_seq = ref_norm.split()
        hyp_seq = hyp_norm.split()
    else:  # character level
        ref_seq = list(ref_norm)
        hyp_seq = list(hyp_norm)
    
    if len(ref_seq) == 0 or len(hyp_seq) == 0:
        return 0.0
    
    lcs_length = _longest_common_subsequence(ref_seq, hyp_seq)
    
    # ROUGE-L = (LCS length) / (average of reference and hypothesis lengths)
    precision = lcs_length / len(hyp_seq) if len(hyp_seq) > 0 else 0.0
    recall = lcs_length / len(ref_seq) if len(ref_seq) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def precision_recall_f1(
    reference: str,
    hypothesis: str,
    level: str = 'word',
    ignore_case: bool = False,
    delimiter: str = ' '
) -> Dict[str, float]:
    """
    Calculate Precision, Recall, and F1 score.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        level: 'word' or 'character' level
        ignore_case: If True, ignore case differences
        delimiter: Delimiter for word splitting (if level='word')
        
    Returns:
        Dictionary with 'precision', 'recall', and 'f1' scores
    """
    ref_norm = normalize_text(reference, ignore_case, False)
    hyp_norm = normalize_text(hypothesis, ignore_case, False)
    
    if level == 'word':
        ref_tokens = set(ref_norm.split(delimiter))
        hyp_tokens = set(hyp_norm.split(delimiter))
    else:  # character level
        ref_tokens = set(ref_norm)
        hyp_tokens = set(hyp_norm)
    
    if len(hyp_tokens) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if len(ref_tokens) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # True positives: tokens in both reference and hypothesis
    tp = len(ref_tokens & hyp_tokens)
    
    # Precision: TP / (TP + FP) = TP / |hypothesis|
    precision = tp / len(hyp_tokens)
    
    # Recall: TP / (TP + FN) = TP / |reference|
    recall = tp / len(ref_tokens)
    
    # F1: harmonic mean of precision and recall
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_all_metrics(
    reference: str,
    hypothesis: str,
    ignore_case: bool = False
) -> Dict[str, float]:
    """
    Calculate all available metrics for a single reference-hypothesis pair.
    
    Args:
        reference: Reference text
        hypothesis: Hypothesis text
        ignore_case: If True, ignore case differences
        
    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {}
    
    # Error rates
    try:
        metrics['wer'] = wer(reference, hypothesis, ignore_case)
    except ValueError:
        metrics['wer'] = float('inf')
    
    try:
        metrics['cer'] = cer(reference, hypothesis, ignore_case)
    except ValueError:
        metrics['cer'] = float('inf')
    
    # Accuracies
    metrics['word_accuracy'] = word_accuracy(reference, hypothesis, ignore_case)
    metrics['character_accuracy'] = character_accuracy(reference, hypothesis, ignore_case)
    
    # BLEU and ROUGE
    metrics['bleu'] = bleu_score(reference, hypothesis, max_n=4, ignore_case=ignore_case)
    metrics['rouge_l'] = rouge_l_score(reference, hypothesis, ignore_case=ignore_case, level='word')
    
    # Precision, Recall, F1
    word_prf = precision_recall_f1(reference, hypothesis, level='word', ignore_case=ignore_case)
    char_prf = precision_recall_f1(reference, hypothesis, level='character', ignore_case=ignore_case)
    
    metrics['word_precision'] = word_prf['precision']
    metrics['word_recall'] = word_prf['recall']
    metrics['word_f1'] = word_prf['f1']
    
    metrics['char_precision'] = char_prf['precision']
    metrics['char_recall'] = char_prf['recall']
    metrics['char_f1'] = char_prf['f1']
    
    return metrics


def calculate_batch_metrics(
    references: List[str],
    hypotheses: List[str],
    ignore_case: bool = False
) -> Dict[str, float]:
    """
    Calculate average metrics for a batch of reference-hypothesis pairs.
    
    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        ignore_case: If True, ignore case differences
        
    Returns:
        Dictionary containing average metrics
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    
    if len(references) == 0:
        return {}
    
    # Calculate individual metrics for each pair
    all_metrics = []
    for ref, hyp in zip(references, hypotheses):
        metrics = calculate_all_metrics(ref, hyp, ignore_case)
        all_metrics.append(metrics)
    
    # Calculate averages
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not math.isinf(m[key])]
        if values:
            avg_metrics[key] = sum(values) / len(values)
        else:
            avg_metrics[key] = float('inf')
    
    # Add sentence-level metrics
    avg_metrics['ser'] = sentence_error_rate(references, hypotheses, ignore_case)
    avg_metrics['exact_match_rate'] = exact_match_rate(references, hypotheses, ignore_case)
    
    return avg_metrics


"""
ASR (Automatic Speech Recognition) metrics calculation.
Includes Word Error Rate (WER) and Character Error Rate (CER) using Levenshtein distance.
"""

from typing import List, Union, Optional


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


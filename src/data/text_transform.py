"""
Text transformation utilities for converting between text and integer sequences.
Supports both English and Persian (Farsi) characters.
"""

import string
from typing import List, Union


class TextTransform:
    """
    Maps characters to integers and vice versa for CTC-based speech recognition.
    
    Character set includes:
    - English lowercase letters (a-z)
    - English numbers (0-9)
    - Persian alphabet (ا-ی)
    - Persian numbers (۰-۹)
    - Common punctuation
    - Space character (mapped as '<SPACE>')
    - Blank token for CTC
    """
    
    def __init__(self, include_persian: bool = True):
        """
        Initialize the character-to-index and index-to-character mappings.
        
        Args:
            include_persian: If True, include Persian characters (default: True)
        """
        # Persian alphabet (32 letters)
        persian_alphabet = [
            'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ',
            'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م',
            'ن', 'و', 'ه', 'ی'
        ]
        
        # Persian numbers (۰-۹)
        persian_numbers = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']
        
        # English numbers (0-9)
        english_numbers = list(string.digits)
        
        # Common punctuation
        punctuation = ["'", '"', '،', '؛', '؟', '!', '.', ',', ':', ';', '-', '_', '(', ')', '[', ']', '{', '}']
        
        # Special tokens
        special_tokens = ['<SPACE>']
        
        # Build character list
        all_chars = []
        
        # Add special tokens first
        all_chars.extend(special_tokens)
        
        # Add English lowercase letters
        all_chars.extend(list(string.ascii_lowercase))
        
        # Add English numbers
        all_chars.extend(english_numbers)
        
        # Add Persian characters if enabled
        if include_persian:
            all_chars.extend(persian_alphabet)
            all_chars.extend(persian_numbers)
        
        # Add punctuation
        all_chars.extend(punctuation)
        
        # Generate char_map: character -> index
        self.char_map = {char: i for i, char in enumerate(all_chars)}
        # Generate index_map: index -> character
        self.index_map = {index: char for char, index in self.char_map.items()}
        # Map space placeholder to actual space character
        self.index_map[self.char_map['<SPACE>']] = ' '
        
        # Blank token for CTC (separate class, not part of char_map)
        # In CTC, blank is typically the last class index
        # n_class = len(all_chars) + 1 (chars + blank)
        # blank_id should be len(all_chars) which is the last index in the model output
        self.blank_id = len(all_chars)
    
    def text_to_int(self, text: str) -> List[int]:
        """
        Convert text to an integer sequence using the character map.
        
        Args:
            text: Input text string (can contain English, Persian, or mixed)
            
        Returns:
            List of integers representing the text
        """
        # Don't lowercase Persian text as it may affect meaning
        # Only lowercase English letters
        result = []
        for c in text:
            if c in self.char_map:
                result.append(self.char_map[c])
            elif c == ' ':
                # Map space to <SPACE> token
                result.append(self.char_map['<SPACE>'])
            elif c.lower() in self.char_map and c.isupper():
                # Handle uppercase English letters
                result.append(self.char_map[c.lower()])
            else:
                # Unknown character - map to space
                result.append(self.char_map['<SPACE>'])
        return result
    
    def int_to_text(self, labels: Union[List[int], List[List[int]]]) -> str:
        """
        Convert integer labels to a text sequence using the character map.
        
        Args:
            labels: Integer sequence or list of sequences
            
        Returns:
            Text string (or list of strings if input is list of sequences)
        """
        # Handle empty input
        if not labels:
            return ''
        
        if isinstance(labels[0], list):
            # Handle batch of sequences
            return [self.int_to_text(seq) for seq in labels]
        
        # Single sequence
        text = ''.join(
            self.index_map.get(i, '') 
            for i in labels
        ).replace('<SPACE>', ' ')
        return text
    
    def int_to_text_remove_pad(self, int_sequence: List[int], pad_value: int = 0) -> str:
        """
        Convert integer sequence to text, removing trailing padding zeros.
        
        Args:
            int_sequence: Integer sequence that may contain padding
            pad_value: Value used for padding (default: 0)
            
        Returns:
            Text string without padding
        """
        from itertools import dropwhile
        
        # Reverse, remove padding from the end (now start), and reverse back
        text_sequence = list(
            dropwhile(lambda x: x == pad_value, reversed(int_sequence))
        )[::-1]
        
        # Convert the non-padded sequence to text
        text = ''.join(
            self.index_map.get(i, '') 
            for i in text_sequence
        ).replace('<SPACE>', ' ')
        return text
    
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size (number of unique characters + blank).
        
        Returns:
            Vocabulary size (characters + blank token)
        """
        # Vocabulary includes all characters plus blank token
        return len(self.char_map) + 1


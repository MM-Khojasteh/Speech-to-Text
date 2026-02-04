"""Model architectures for speech recognition."""

from .speech_recognition import SpeechRecognitionModel
from .rnn_blocks import ChunkedMultiHeadSelfAttention, MultiHeadSelfAttention

__all__ = ['SpeechRecognitionModel', 'ChunkedMultiHeadSelfAttention', 'MultiHeadSelfAttention']


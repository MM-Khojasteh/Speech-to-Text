"""
Audio transformation and augmentation utilities for speech recognition.
"""

import torch
import torch.nn as nn
import torchaudio


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.
    
    Applies frequency masking and time masking to spectrograms.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 30,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ):
        """
        Initialize SpecAugment.
        
        Args:
            freq_mask_param: Maximum width of frequency masks
            time_mask_param: Maximum width of time masks
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
        """
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param)
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram tensor of shape (freq, time) or (batch, freq, time)
            
        Returns:
            Augmented spectrogram
        """
        # Handle batch dimension
        if spectrogram.dim() == 3:
            # Batch processing
            for _ in range(self.num_freq_masks):
                spectrogram = self.freq_masking(spectrogram)
            for _ in range(self.num_time_masks):
                spectrogram = self.time_masking(spectrogram)
        else:
            # Single sample
            for _ in range(self.num_freq_masks):
                spectrogram = self.freq_masking(spectrogram)
            for _ in range(self.num_time_masks):
                spectrogram = self.time_masking(spectrogram)
        
        return spectrogram


def get_train_transforms(
    sample_rate: int = 16000,
    n_mels: int = 128,
    freq_mask_param: int = 30,
    time_mask_param: int = 100,
    num_freq_masks: int = 2,
    num_time_masks: int = 2
) -> nn.Sequential:
    """
    Get training audio transforms with augmentation.
    
    Args:
        sample_rate: Audio sample rate
        n_mels: Number of mel filter banks
        freq_mask_param: Maximum width of frequency masks
        time_mask_param: Maximum width of time masks
        num_freq_masks: Number of frequency masks
        num_time_masks: Number of time masks
        
    Returns:
        Sequential transform pipeline
    """
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160,
            f_min=0.0,
            f_max=sample_rate / 2.0
        ),
        SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            num_freq_masks=num_freq_masks,
            num_time_masks=num_time_masks
        )
    )


def get_valid_transforms(
    sample_rate: int = 16000,
    n_mels: int = 128
) -> torchaudio.transforms.MelSpectrogram:
    """
    Get validation audio transforms (no augmentation).
    
    Args:
        sample_rate: Audio sample rate
        n_mels: Number of mel filter banks
        
    Returns:
        MelSpectrogram transform
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=400,
        hop_length=160,
        f_min=0.0,
        f_max=sample_rate / 2.0
    )


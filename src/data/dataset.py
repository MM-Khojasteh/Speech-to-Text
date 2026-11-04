"""
Dataset loading and collation functions for speech recognition.
"""

import torch
import torch.nn as nn
import torch.utils.data as data
from typing import List, Tuple, Optional
import random

from .text_transform import TextTransform
from .transforms import get_train_transforms, get_valid_transforms
from .custom_dataset import CustomAudioDataset


class CollateFn:
    """
    Picklable collate function class for Windows multiprocessing compatibility.

    Transforms are cached in __init__ to avoid recreating them on every batch,
    which was a significant CPU bottleneck.
    """
    def __init__(
        self,
        data_type: str,
        text_transform: TextTransform,
        sample_rate: int,
        n_mels: int,
        freq_mask_param: int = 30,
        time_mask_param: int = 100,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        stride: int = 2
    ):
        self.data_type = data_type
        self.text_transform = text_transform
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.stride = stride

        # Cache transforms to avoid recreating them on every batch
        if data_type == 'train':
            self.audio_transforms = get_train_transforms(
                sample_rate=sample_rate,
                n_mels=n_mels,
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
                num_freq_masks=num_freq_masks,
                num_time_masks=num_time_masks
            )
        else:
            self.audio_transforms = get_valid_transforms(
                sample_rate=sample_rate,
                n_mels=n_mels
            )

    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
        """Call the data processing function with cached transforms."""
        return data_processing_with_transforms(
            batch,
            self.data_type,
            self.text_transform,
            self.audio_transforms,
            self.stride
        )


def data_processing_with_transforms(
    batch: List[Tuple],
    data_type: str = 'train',
    text_transform: Optional[TextTransform] = None,
    audio_transforms = None,
    stride: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Process a batch of data for training or validation using pre-built transforms.

    This is the optimized version that accepts cached transforms to avoid
    recreating them on every batch call.

    Supports both LibriSpeech format and custom dataset format:
    - LibriSpeech: (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
    - Custom: (waveform, sample_rate, transcript, audio_filename)

    Args:
        batch: List of tuples (waveform, sample_rate, utterance/transcript, ...)
        data_type: 'train' or 'valid'
        text_transform: TextTransform instance for encoding text
        audio_transforms: Pre-built audio transforms (cached)
        stride: CNN stride for calculating CTC input lengths (default: 2)

    Returns:
        Tuple of (spectrograms, labels, input_lengths, label_lengths)
        - spectrograms: Padded batch of spectrograms (batch, 1, freq, time)
        - labels: Padded batch of label sequences (batch, max_label_len)
        - input_lengths: List of actual sequence lengths for CTC
        - label_lengths: List of actual label lengths for CTC
    """
    if data_type not in ['train', 'valid']:
        raise ValueError("data_type should be either 'train' or 'valid'.")

    if text_transform is None:
        text_transform = TextTransform()
    
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    
    for item in batch:
        # Handle both LibriSpeech format (6 items) and custom format (4 items)
        if len(item) >= 6:
            # LibriSpeech format: (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
            waveform, _, utterance, _, _, _ = item
        elif len(item) >= 3:
            # Custom format: (waveform, sample_rate, transcript, audio_filename)
            waveform, _, utterance = item[:3]
        else:
            raise ValueError(f"Unexpected batch item format with {len(item)} elements")
        
        # Skip empty transcripts
        if not utterance or not utterance.strip():
            continue
        
        # Apply audio transforms
        spec = audio_transforms(waveform)
        
        # Handle shape: (1, freq, time) -> (freq, time) -> (time, freq)
        if spec.dim() == 3:
            spec = spec.squeeze(0)
        spec = spec.transpose(0, 1)  # (time, freq)
        
        spectrograms.append(spec)
        
        # Encode text to integers (don't lowercase Persian text)
        # TextTransform handles both English and Persian
        label = torch.tensor(
            text_transform.text_to_int(utterance),
            dtype=torch.long
        )
        labels.append(label)
        
        # Calculate lengths for CTC
        # Input length is reduced by stride in CNN (configurable, typically 2)
        input_lengths.append(spec.shape[0] // stride)
        label_lengths.append(len(label))
    
    if len(spectrograms) == 0:
        # Return None to indicate empty batch - trainer will skip it
        # This can happen with multiprocessing on Windows
        return None
    
    # Pad sequences to same length
    spectrograms = nn.utils.rnn.pad_sequence(
        spectrograms, batch_first=True
    ).unsqueeze(1).transpose(2, 3)  # (batch, 1, freq, time)
    
    labels = nn.utils.rnn.pad_sequence(
        labels, batch_first=True
    )
    
    return spectrograms, labels, input_lengths, label_lengths


def data_processing(
    batch: List[Tuple],
    data_type: str = 'train',
    text_transform: Optional[TextTransform] = None,
    sample_rate: int = 16000,
    n_mels: int = 128,
    freq_mask_param: int = 30,
    time_mask_param: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Process a batch of data for training or validation.

    DEPRECATED: This function recreates transforms on every call, which is slow.
    Use CollateFn class with data_processing_with_transforms instead.

    Kept for backward compatibility.
    """
    # Create transforms (inefficient - recreated every call)
    if data_type == 'train':
        audio_transforms = get_train_transforms(
            sample_rate=sample_rate,
            n_mels=n_mels,
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param
        )
    else:
        audio_transforms = get_valid_transforms(
            sample_rate=sample_rate,
            n_mels=n_mels
        )

    return data_processing_with_transforms(
        batch=batch,
        data_type=data_type,
        text_transform=text_transform,
        audio_transforms=audio_transforms
    )


class LibriSpeechDataset:
    """
    Wrapper for LibriSpeech dataset with proper data loading.
    """
    
    def __init__(
        self,
        root_path: str,
        url: str = "train-clean-100",
        download: bool = False
    ):
        """
        Initialize LibriSpeech dataset.
        
        Args:
            root_path: Root directory for dataset
            url: Dataset subset URL (e.g., "train-clean-100", "test-clean")
            download: Whether to download the dataset
        """
        import torchaudio
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root_path, url=url, download=download
        )
    
    def __getitem__(self, idx: int):
        """Get item from dataset."""
        return self.dataset[idx]
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)


def get_data_loaders(
    train_root: str,
    test_root: str,
    train_url: str = "train-clean-100",
    test_url: str = "test-clean",
    batch_size: int = 10,
    num_workers: int = 4,
    pin_memory: bool = True,
    sample_rate: int = 16000,
    n_mels: int = 128,
    freq_mask_param: int = 30,
    time_mask_param: int = 100
) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Create train and test data loaders.
    
    Args:
        train_root: Root directory for training dataset
        test_root: Root directory for test dataset
        train_url: Training dataset subset URL
        test_url: Test dataset subset URL
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        sample_rate: Audio sample rate
        n_mels: Number of mel filter banks
        freq_mask_param: Frequency masking parameter
        time_mask_param: Time masking parameter
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create text transform (shared between train and test)
    text_transform = TextTransform()
    
    # Create datasets
    train_dataset = LibriSpeechDataset(train_root, train_url, download=False)
    test_dataset = LibriSpeechDataset(test_root, test_url, download=False)
    
    # Create collate functions
    def train_collate_fn(batch):
        return data_processing(
            batch, 'train', text_transform, sample_rate, n_mels,
            freq_mask_param, time_mask_param
        )
    
    def valid_collate_fn(batch):
        return data_processing(
            batch, 'valid', text_transform, sample_rate, n_mels
        )
    
    # Create data loaders
    train_loader = data.DataLoader(
        dataset=train_dataset.dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = data.DataLoader(
        dataset=test_dataset.dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=valid_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def get_custom_data_loaders(
    audio_files_dir: str,
    transcripts_dir: str,
    batch_size: int = 10,
    num_workers: int = 4,
    pin_memory: bool = True,
    sample_rate: int = 16000,
    n_mels: int = 128,
    freq_mask_param: int = 30,
    time_mask_param: int = 100,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
    stride: int = 2,
    train_split: float = 0.8,
    seed: int = 42,
    include_persian: bool = True,
    test_audio_dir: Optional[str] = None
) -> Tuple[data.DataLoader, data.DataLoader, Optional[data.DataLoader], TextTransform]:
    """
    Create train, validation, and test data loaders for custom dataset.

    Args:
        audio_files_dir: Directory containing audio files (.wav) for train/val
        transcripts_dir: Directory containing JSON transcript files
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        sample_rate: Audio sample rate
        n_mels: Number of mel filter banks
        freq_mask_param: Frequency masking parameter
        time_mask_param: Time masking parameter
        num_freq_masks: Number of frequency masks for SpecAugment
        num_time_masks: Number of time masks for SpecAugment
        stride: CNN stride for CTC input length calculation (default: 2)
        train_split: Fraction of data to use for training (default: 0.8)
        seed: Random seed for train/val split
        include_persian: Whether to include Persian characters in TextTransform
        test_audio_dir: Optional directory containing test audio files (.wav)

    Returns:
        Tuple of (train_loader, val_loader, test_loader, text_transform)
        test_loader will be None if test_audio_dir is not provided or empty
    """
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create text transform (shared between train and val)
    text_transform = TextTransform(include_persian=include_persian)
    
    # Create full dataset
    full_dataset = CustomAudioDataset(
        audio_files_dir=audio_files_dir,
        transcripts_dir=transcripts_dir,
        sample_rate=sample_rate
    )
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    # Split dataset
    train_indices, val_indices = torch.utils.data.random_split(
        range(dataset_size),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Create picklable collate functions (for Windows multiprocessing)
    train_collate_fn = CollateFn(
        data_type='train',
        text_transform=text_transform,
        sample_rate=sample_rate,
        n_mels=n_mels,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        num_freq_masks=num_freq_masks,
        num_time_masks=num_time_masks,
        stride=stride
    )

    valid_collate_fn = CollateFn(
        data_type='valid',
        text_transform=text_transform,
        sample_rate=sample_rate,
        n_mels=n_mels,
        stride=stride
    )
    
    # Create data loaders
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=valid_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Dataset split: {train_size} train, {val_size} validation")
    
    # Create test dataset from test_audio_dir if provided
    test_loader = None
    if test_audio_dir:
        from pathlib import Path
        test_audio_path = Path(test_audio_dir)
        
        if test_audio_path.exists() and test_audio_path.is_dir():
            # Check if there are any audio files (both .wav and .WAV)
            test_audio_files = list(test_audio_path.glob('*.wav')) + list(test_audio_path.glob('*.WAV'))
            
            if len(test_audio_files) > 0:
                try:
                    # Check if test.json exists in test_audio directory
                    test_json_path = test_audio_path / 'test.json'
                    if test_json_path.exists():
                        # Use test.json from test_audio directory
                        test_dataset = CustomAudioDataset(
                            audio_files_dir=test_audio_dir,
                            transcripts_dir=test_audio_dir,  # Look for JSON in test_audio dir
                            sample_rate=sample_rate,
                            json_files=['test.json']  # Use test.json specifically
                        )
                    else:
                        # Fallback to using transcripts from main transcripts_dir
                        test_dataset = CustomAudioDataset(
                            audio_files_dir=test_audio_dir,
                            transcripts_dir=transcripts_dir,
                            sample_rate=sample_rate
                        )
                    
                    # Create test collate function (no augmentation) - picklable
                    test_collate_fn = CollateFn(
                        data_type='valid',
                        text_transform=text_transform,
                        sample_rate=sample_rate,
                        n_mels=n_mels,
                        stride=stride
                    )
                    
                    # Create test data loader
                    test_loader = data.DataLoader(
                        dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=test_collate_fn,
                        num_workers=num_workers,
                        pin_memory=pin_memory
                    )
                    
                    print(f"Test dataset: {len(test_dataset)} samples from {test_audio_dir}")
                except Exception as e:
                    print(f"Warning: Could not create test dataset from {test_audio_dir}: {e}")
                    print("Continuing without test set...")
            else:
                print(f"Warning: No audio files found in {test_audio_dir}")
        else:
            print(f"Warning: Test audio directory {test_audio_dir} does not exist")
    
    return train_loader, val_loader, test_loader, text_transform


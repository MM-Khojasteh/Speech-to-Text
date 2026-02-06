"""
Test script for custom dataset integration.
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.text_transform import TextTransform
from src.data.custom_dataset import CustomAudioDataset
from src.data.dataset import get_custom_data_loaders

def test_text_transform():
    """Test Persian text transformation."""
    print("=" * 60)
    print("Testing TextTransform with Persian support")
    print("=" * 60)
    
    tt = TextTransform(include_persian=True)
    vocab_size = tt.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    print(f"Blank token ID: {tt.blank_id}")
    
    # Test Persian text
    test_texts = [
        "سلام دنیا",
        "Hello world",
        "سلام Hello 123",
        "۰۱۲۳۴۵۶۷۸۹"
    ]
    
    for text in test_texts:
        encoded = tt.text_to_int(text)
        decoded = tt.int_to_text(encoded)
        try:
            print(f"\nOriginal: {text}")
            print(f"Encoded length: {len(encoded)}")
            print(f"Decoded: {decoded}")
            print(f"Match: {text == decoded}")
        except UnicodeEncodeError:
            # Handle encoding issues in Windows console
            print(f"\nOriginal: (Persian/Unicode text)")
            print(f"Encoded length: {len(encoded)}")
            print(f"Decoded: (Persian/Unicode text)")
            print(f"Match: {text == decoded}")
    
    print("\n" + "=" * 60)

def test_custom_dataset():
    """Test custom dataset loading."""
    print("=" * 60)
    print("Testing CustomAudioDataset")
    print("=" * 60)
    
    try:
        dataset = CustomAudioDataset(
            audio_files_dir="dataset/audio_files",
            transcripts_dir="dataset/info",
            sample_rate=16000
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test loading first item
            waveform, sample_rate, transcript, filename = dataset[0]
            print(f"\nFirst sample:")
            print(f"  Filename: {filename}")
            print(f"  Waveform shape: {waveform.shape}")
            print(f"  Sample rate: {sample_rate}")
            print(f"  Transcript length: {len(transcript)}")
            try:
                print(f"  Transcript preview: {transcript[:100]}...")
            except UnicodeEncodeError:
                print(f"  Transcript preview: (Persian/Unicode text, length: {len(transcript)})")
        
        print("\n" + "=" * 60)
        return True
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loaders():
    """Test data loaders with train/val split."""
    print("=" * 60)
    print("Testing Custom Data Loaders")
    print("=" * 60)
    
    try:
        train_loader, val_loader, test_loader, text_transform = get_custom_data_loaders(
            audio_files_dir="dataset/audio_files",
            transcripts_dir="dataset/info",
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            pin_memory=False,
            sample_rate=16000,
            n_mels=128,
            freq_mask_param=30,
            time_mask_param=100,
            train_split=0.8,
            seed=42,
            include_persian=True,
            test_audio_dir="test_audio"
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        if test_loader is not None:
            print(f"Test batches: {len(test_loader)}")
        else:
            print("Test batches: 0 (test_audio directory not found or empty)")
        print(f"Vocabulary size: {text_transform.get_vocab_size()}")
        
        # Test loading a batch
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            spectrograms, labels, input_lengths, label_lengths = batch
            print(f"\nBatch shapes:")
            print(f"  Spectrograms: {spectrograms.shape}")
            print(f"  Labels: {labels.shape}")
            print(f"  Input lengths: {input_lengths}")
            print(f"  Label lengths: {label_lengths}")
        
        print("\n" + "=" * 60)
        return True
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Custom Dataset Integration Test")
    print("=" * 60 + "\n")
    
    # Test 1: Text Transform
    test_text_transform()
    
    # Test 2: Custom Dataset
    dataset_ok = test_custom_dataset()
    
    # Test 3: Data Loaders (only if dataset loaded successfully)
    if dataset_ok:
        loaders_ok = test_data_loaders()
        
        if loaders_ok:
            print("\n" + "=" * 60)
            print("All tests passed!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("Data loader test failed")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Dataset test failed - skipping data loader test")
        print("=" * 60)


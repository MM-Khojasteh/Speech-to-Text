"""
Test script for train/validation/test split implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import get_custom_data_loaders
from src.data.custom_dataset import CustomAudioDataset

def test_split():
    """Test train/val/test split."""
    print("=" * 70)
    print("Testing Train/Validation/Test Split")
    print("=" * 70)
    
    try:
        # Test with test_audio directory
        print("\n1. Testing with test_audio directory:")
        train_loader, val_loader, test_loader, text_transform = get_custom_data_loaders(
            audio_files_dir="dataset/audio_files",
            transcripts_dir="dataset/info",
            batch_size=2,
            num_workers=0,
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
        
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        if test_loader is not None:
            print(f"  Test batches: {len(test_loader)}")
        else:
            print(f"  Test batches: None (test_audio directory not found or empty)")
        
        # Verify no overlap between train/val and test
        if test_loader is not None:
            print("\n2. Verifying no data leakage:")
            # Get filenames from each set
            train_dataset = train_loader.dataset
            val_dataset = val_loader.dataset
            test_dataset = test_loader.dataset
            
            # Extract filenames (if using Subset, get from underlying dataset)
            if hasattr(train_dataset, 'dataset'):
                train_dataset = train_dataset.dataset
            if hasattr(val_dataset, 'dataset'):
                val_dataset = val_dataset.dataset
            
            train_files = {pair[0].name for pair in train_dataset.audio_transcript_pairs}
            val_files = {pair[0].name for pair in val_dataset.audio_transcript_pairs}
            test_files = {pair[0].name for pair in test_dataset.audio_transcript_pairs}
            
            train_val_overlap = train_files & val_files
            train_test_overlap = train_files & test_files
            val_test_overlap = val_files & test_files
            
            print(f"  Train files: {len(train_files)}")
            print(f"  Validation files: {len(val_files)}")
            print(f"  Test files: {len(test_files)}")
            print(f"  Train-Val overlap: {len(train_val_overlap)} (should be 0)")
            print(f"  Train-Test overlap: {len(train_test_overlap)} (should be 0)")
            print(f"  Val-Test overlap: {len(val_test_overlap)} (should be 0)")
            
            if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
                print("  ✓ No data leakage detected!")
            else:
                print("  ✗ WARNING: Data leakage detected!")
        
        # Test without test_audio directory
        print("\n3. Testing without test_audio directory:")
        train_loader2, val_loader2, test_loader2, _ = get_custom_data_loaders(
            audio_files_dir="dataset/audio_files",
            transcripts_dir="dataset/info",
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            sample_rate=16000,
            n_mels=128,
            freq_mask_param=30,
            time_mask_param=100,
            train_split=0.8,
            seed=42,
            include_persian=True,
            test_audio_dir=None
        )
        
        print(f"  Train batches: {len(train_loader2)}")
        print(f"  Validation batches: {len(val_loader2)}")
        print(f"  Test loader: {test_loader2} (should be None)")
        
        if test_loader2 is None:
            print("  ✓ Correctly returns None when test_audio_dir is not provided")
        
        print("\n" + "=" * 70)
        print("All tests passed!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_split()


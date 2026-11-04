"""
Custom dataset loader for audio files with JSON transcripts.
Supports matching audio files with transcripts from multiple JSON sources.
"""

import json
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import torch
import torch.utils.data as data

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

# Ensure at least one audio library is available
if not SOUNDFILE_AVAILABLE and not TORCHAUDIO_AVAILABLE:
    raise ImportError("Neither soundfile nor torchaudio is available. Please install one of them.")


class CustomAudioDataset(data.Dataset):
    """
    Custom dataset that loads audio files and matches them with transcripts from JSON files.

    Transcript format is auto-detected from the entry keys: entries carrying
    a 'recording_file' key use that field as the audio filename, entries
    carrying an 'audio_file_path' key use that one; the text is always read
    from the 'transcript' field.
    """
    
    def __init__(
        self,
        audio_files_dir: str,
        transcripts_dir: str,
        sample_rate: int = 16000,
        json_files: Optional[List[str]] = None
    ):
        """
        Initialize custom audio dataset.
        
        Args:
            audio_files_dir: Directory containing audio files (.wav)
            transcripts_dir: Directory containing JSON transcript files
            sample_rate: Target sample rate for audio (default: 16000)
            json_files: List of JSON filenames to load. If None, loads every
                *.json file in transcripts_dir.
        """
        self.audio_files_dir = Path(audio_files_dir)
        self.transcripts_dir = Path(transcripts_dir)
        self.sample_rate = sample_rate

        # By default, load every JSON transcript file in the directory
        if json_files is None:
            json_files = sorted(p.name for p in self.transcripts_dir.glob('*.json'))
            if not json_files:
                raise ValueError(
                    f"No JSON transcript files found in {self.transcripts_dir}"
                )
        
        # Load transcripts from JSON files
        self.transcripts = self._load_transcripts(json_files)
        
        # Get all audio files (both .wav and .WAV)
        audio_files = list(self.audio_files_dir.glob('*.wav')) + list(self.audio_files_dir.glob('*.WAV'))
        
        # Match audio files with transcripts
        self.audio_transcript_pairs = self._match_audio_with_transcripts(audio_files)
        
        # Filter out pairs with empty transcripts
        self.audio_transcript_pairs = [
            (audio_path, transcript) 
            for audio_path, transcript in self.audio_transcript_pairs
            if transcript and transcript.strip()
        ]
        
        if len(self.audio_transcript_pairs) == 0:
            raise ValueError(
                f"No matching audio-transcript pairs found. "
                f"Found {len(audio_files)} audio files and {len(self.transcripts)} transcripts."
            )
    
    def _load_transcripts(self, json_files: List[str]) -> Dict[str, str]:
        """
        Load transcripts from JSON files.
        
        Args:
            json_files: List of JSON filenames to load
            
        Returns:
            Dictionary mapping audio filename to transcript text
        """
        transcripts = {}
        
        for json_file in json_files:
            json_path = self.transcripts_dir / json_file
            
            # Try to resolve absolute path
            if not json_path.exists():
                json_path = Path(json_path).resolve()
                if not json_path.exists():
                    print(f"Warning: JSON file not found: {self.transcripts_dir / json_file}")
                    print(f"  Resolved path: {json_path}")
                    print(f"  Transcripts dir exists: {self.transcripts_dir.exists()}")
                    continue
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Auto-detect the filename field from the first entry's keys
                first_entry = data[0] if data else {}
                if 'recording_file' in first_entry:
                    filename_field = 'recording_file'
                elif 'audio_file_path' in first_entry:
                    filename_field = 'audio_file_path'
                else:
                    print(f"Warning: Could not determine filename field in {json_file}")
                    continue
                
                # Extract transcripts
                for item in data:
                    if filename_field in item and 'transcript' in item:
                        audio_filename = item[filename_field]
                        transcript = item['transcript']
                        
                        # Normalize filename (handle path separators)
                        audio_filename = os.path.basename(audio_filename)
                        
                        if transcript and transcript.strip():
                            # If multiple transcripts for same file, keep the longest one
                            if audio_filename not in transcripts or len(transcript) > len(transcripts[audio_filename]):
                                transcripts[audio_filename] = transcript.strip()
                
                print(f"Loaded {len([k for k in transcripts.keys() if k in [item.get(filename_field, '') for item in data]])} transcripts from {json_file}")
            
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        return transcripts
    
    def _match_audio_with_transcripts(
        self,
        audio_files: List[Path]
    ) -> List[Tuple[Path, str]]:
        """
        Match audio files with their corresponding transcripts.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of tuples (audio_path, transcript)
        """
        pairs = []
        matched_count = 0
        unmatched_count = 0
        
        for audio_file in audio_files:
            audio_filename = audio_file.name
            
            if audio_filename in self.transcripts:
                pairs.append((audio_file, self.transcripts[audio_filename]))
                matched_count += 1
            else:
                unmatched_count += 1
        
        print(f"Matched {matched_count} audio files with transcripts")
        if unmatched_count > 0:
            print(f"Warning: {unmatched_count} audio files have no matching transcript")
        
        return pairs
    
    def _load_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and resample if necessary.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        if SOUNDFILE_AVAILABLE:
            waveform, sr = sf.read(str(audio_path))
            waveform = torch.from_numpy(waveform).float()
        elif TORCHAUDIO_AVAILABLE:
            waveform, sr = torchaudio.load(str(audio_path))
            waveform = waveform.squeeze(0)  # Remove channel dimension if mono
        else:
            raise RuntimeError("No audio loading library available")
        
        # Resample if necessary
        if sr != self.sample_rate:
            if TORCHAUDIO_AVAILABLE:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            else:
                # Simple resampling using interpolation (less accurate)
                import torch.nn.functional as F
                ratio = self.sample_rate / sr
                new_length = int(waveform.shape[0] * ratio)
                waveform = F.interpolate(
                    waveform.unsqueeze(0).unsqueeze(0),
                    size=new_length,
                    mode='linear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
        
        return waveform, self.sample_rate
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str]:
        """
        Get a single audio-transcript pair.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (waveform, sample_rate, transcript, audio_filename)
        """
        audio_path, transcript = self.audio_transcript_pairs[idx]
        
        try:
            waveform, sample_rate = self._load_audio(audio_path)
            # Ensure transcript is not empty (shouldn't happen due to filtering, but double-check)
            if not transcript or not transcript.strip():
                # Return a minimal valid item that will be filtered out in collate
                return torch.zeros(1), self.sample_rate, "", audio_path.name
            return waveform, sample_rate, transcript, audio_path.name
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return a minimal item that will be filtered out in collate
            return torch.zeros(1), self.sample_rate, "", audio_path.name
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.audio_transcript_pairs)
    
    def get_transcript(self, idx: int) -> str:
        """Get transcript text for a given index."""
        return self.audio_transcript_pairs[idx][1]


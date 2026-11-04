# Speech-to-Text (PyTorch)

CTC-based speech recognition supporting **English and Persian (Farsi)**, designed to train on low-VRAM (6 GB) GPUs.

## Highlights

- **Tiered adaptive CTC blank penalty** — counters blank-token collapse via an escalating penalty schedule (configurable tier matrix)
- **Memory-efficient modeling** — chunked multi-head self-attention, squeeze-and-excitation CNN blocks, AMP, gradient accumulation
- **Bilingual vocabulary** — shared English/Persian character set with mixed-script transcripts
- **SpecAugment** — time/frequency masking augmentation
- **Training toolbox** — LR warmup, OneCycle/Cosine/Step schedulers, early stopping, checkpointing, TensorBoard
- **Full evaluation suite** — WER, CER, SER, accuracy, BLEU, ROUGE-L, precision/recall/F1

## Architecture

```
Waveform → MelSpectrogram → Residual CNN (+SE) → Chunked MHSA → BiGRU → Linear → CTC
```

## Project Structure

```
├── src/
│   ├── models/      # CNN/RNN/attention blocks, full model
│   ├── data/        # datasets, transforms, bilingual text transform
│   ├── training/    # trainer, validator, checkpointing, CTC + blank-penalty loss
│   ├── metrics/     # ASR metrics (WER/CER/BLEU/...)
│   ├── inference/   # greedy & beam-search decoders
│   └── utils/       # config + environment layers
├── scripts/         # train / evaluate / inference / verification scripts
└── configs/         # example configuration (tuned matrix lives in .env)
```

## Quickstart

1. **Install**

   ```bash
   pip install -r requirements.txt
   # CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/
   ```

2. **Configure** (required — fails fast without it)

   ```bash
   cp .env.example .env                                              # then fill in the STT_* variables
   cp configs/default_config.example.yaml configs/default_config.yaml
   ```

   Tuned hyperparameters, the blank-penalty tier matrix, and logic toggles are injected from environment variables; the YAML defines only structure and dataset paths. No extra dependency needed — `src/utils/env.py` parses `.env` and reports every missing variable.

3. **Prepare data** (custom dataset)

   ```
   dataset/audio_files/*.wav    # training/validation audio
   dataset/info/*.json          # transcripts: {"recording_file": "...", "transcript": "..."}
   dataset/test_audio/*.wav     # held-out test set
   ```

   LibriSpeech is also supported (`STT_DATA_DATASET_TYPE=librispeech`).

4. **Train / evaluate / infer**

   ```bash
   python scripts/train.py --config configs/default_config.yaml
   python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --split test
   python scripts/inference.py --audio sample.wav --checkpoint checkpoints/best_model.pth
   ```

## Metrics

Validation and evaluation report WER, CER, SER, word/character accuracy, exact match, BLEU, ROUGE-L, and word/character precision/recall/F1.

## License

[Specify your license here]

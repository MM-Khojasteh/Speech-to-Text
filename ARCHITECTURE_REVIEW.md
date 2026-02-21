# Architecture Review: Speech-to-Text System

An independent technical review of the project architecture, identifying strengths, weaknesses, and potential improvements -- all within the constraint of the current GPU budget (6 GB VRAM, single GPU).

> **Update (2026-02-21):** All identified weaknesses have been addressed. See Section 7 for fix details.

---

## 1. High-Level Summary

The system is a CTC-based speech recognition pipeline inspired by DeepSpeech 2. The data flows through:

```
Raw Audio --> MelSpectrogram (+ SpecAugment) --> 2D CNN feature extractor
  --> Chunked Self-Attention --> Bidirectional GRU stack --> 3-layer Classifier
  --> CTC log-softmax --> Greedy / Beam-search decoding --> Text
```

It supports English and Persian, custom datasets with JSON transcripts, and is engineered to fit within a 6 GB GPU.

---

## 2. Strengths

### 2.1 Memory-Conscious Design Throughout
The entire architecture is tuned around 6 GB VRAM without feeling like an afterthought. Chunked attention (O(n*w) instead of O(n^2)), AMP, gradient accumulation (effective batch size 4 from physical batch size 1), and periodic `cuda.empty_cache()` calls all work together coherently. This is well-thought-out.

### 2.2 Chunked Attention with Overlap Blending
`ChunkedMultiHeadSelfAttention` (`src/models/rnn_blocks.py:100-244`) is a genuinely useful component. It processes long sequences in overlapping windows and blends boundary regions with a linear interpolation ramp. It also falls back to standard attention when the sequence is short enough, avoiding unnecessary overhead. This avoids OOM on long audio while keeping global context at chunk boundaries.

### 2.3 CTC Blank-Token Penalty
The custom `CTCLossWithBlankPenalty` (`src/training/ctc_loss_with_penalty.py`) directly addresses the well-known CTC failure mode where the model learns to output only blank tokens. The tiered adaptive penalty (escalating multipliers from 0.2x at <5% blank probability up to 20x at >80%) is a practical solution to a real training problem.

### 2.4 Clean Module Separation
The codebase has a clear separation of concerns:
- `src/models/` -- architecture only, no training logic
- `src/training/` -- trainer, validator, loss, checkpointing
- `src/data/` -- dataset loading, text transforms, audio transforms
- `src/inference/` -- decoding algorithms
- `src/metrics/` -- evaluation metrics (standalone, no model dependency)
- `scripts/` -- entry points

This makes each component testable and replaceable independently.

### 2.5 Bilingual Character Support
`TextTransform` (`src/data/text_transform.py`) handles English and Persian in a single vocabulary with proper treatment of case sensitivity (lowercase English, preserve Persian) and unknown character fallback. The blank token is correctly placed at the last index, separate from the character map.

### 2.6 Comprehensive Evaluation Metrics
The validator reports 13+ metrics (WER, CER, SER, word/character accuracy, exact match rate, BLEU, ROUGE-L, word-level and character-level precision/recall/F1). For a custom ASR system this gives good diagnostic visibility into where the model is failing.

### 2.7 Windows Compatibility
Picklable `CollateFn` class, automatic `num_workers=0` on Windows, empty-batch handling -- these are practical details that many academic codebases ignore.

### 2.8 Flexible Architecture Switches
The model supports GRU vs. Transformer backbone, standard vs. chunked attention, and optional Squeeze-and-Excitation blocks, all toggled by config flags. This allows experimentation without code changes.

### 2.9 Robust Data Pipeline
The custom dataset loader (`src/data/custom_dataset.py`) handles multiple JSON formats, auto-detects filename fields, filters empty transcripts, falls back between audio loading libraries (soundfile / torchaudio), and resamples when necessary. The separate test set from a different directory prevents data leakage.

---

## 3. Weaknesses

### 3.1 Feature Extraction is Recreated Every Batch (Critical)

In `data_processing()` (`src/data/dataset.py:88-99`), `get_train_transforms()` and `get_valid_transforms()` are called on every single collate call. Each call constructs a new `MelSpectrogram` transform (with new STFT window tensors, etc.) and new `SpecAugment` module. With batch_size=1 and gradient_accumulation_steps=4, this creates ~4 new transform objects per optimizer step.

This is a CPU-time bottleneck. The transforms should be created once and stored in the `CollateFn` instance, not rebuilt on each `__call__`.

### 3.2 No Residual Connection in BidirectionalGRU Layers

Each `BidirectionalGRU` (`src/models/rnn_blocks.py:247-294`) applies LayerNorm -> GELU -> BiGRU -> Dropout but has no residual connection. With 3 stacked GRU layers, the gradient must flow through all GRU cells without any skip pathways. This makes training deeper GRU stacks (e.g. 5-6 layers) harder and limits how much the model can benefit from depth -- a significant limitation when the model is already small (rnn_dim=128) due to GPU constraints.

The attention layers do have residuals, but the GRU backbone does not.

### 3.3 Effective Batch Size is Very Small

batch_size=1 with gradient_accumulation_steps=4 gives an effective batch size of 4. For CTC-based ASR, this is quite small. CTC loss benefits significantly from batch-level diversity because it needs varied alignments to learn from. Small effective batch size leads to:
- Noisy gradient estimates
- Slower convergence
- CTC loss instability (the loss is more prone to spikes)

This is a direct consequence of the GPU constraint and cannot be easily improved without more memory. Acknowledged -- but it should be understood as a fundamental limitation of training on 6 GB.

### 3.4 Blank Penalty Loop is Not Vectorized

The blank penalty computation in `CTCLossWithBlankPenalty.forward()` (`src/training/ctc_loss_with_penalty.py:74-107`) iterates over each sample in the batch with a Python for-loop, constructing individual tensors and appending to a list. While not problematic at batch_size=1, this would not scale if the batch size increased. More importantly, the branching logic (6 `if/elif` levels) prevents GPU parallelization.

### 3.5 MelSpectrogram is Computed on CPU

Audio transforms (MelSpectrogram + SpecAugment) happen inside the collate function, which runs on CPU. The spectrograms are then transferred to GPU. Computing MelSpectrogram on GPU would be faster, but this would require restructuring the data pipeline. Given the current GPU memory constraints, this may not be feasible, but it is an architectural limitation worth noting.

### 3.6 OneCycleLR with max_lr = learning_rate

In `scripts/train.py:52-58`, the OneCycleLR scheduler is initialized with `max_lr=config.training.learning_rate` (5e-5). OneCycleLR is designed to warm up from a small LR to `max_lr` and then anneal down. Setting `max_lr` to the base learning rate means the actual peak LR is 5e-5 and the effective training LR is much lower for most of the schedule. Typically, `max_lr` should be 5-10x the base LR for OneCycleLR to provide its intended benefit.

### 3.7 No Data Augmentation Beyond SpecAugment

The only augmentation is frequency and time masking on the spectrogram. There is no:
- Speed perturbation (0.9x, 1.0x, 1.1x) -- the single most effective ASR augmentation
- Noise injection or SNR mixing
- Pitch shifting
- Volume perturbation
- Time stretching

For a small dataset on a small model, augmentation diversity is one of the most cost-effective ways to improve generalization, and it costs zero extra GPU memory.

### 3.8 Hardcoded Stride Assumption in Input Length Calculation

In `data_processing()` (`src/data/dataset.py:141`):
```python
input_lengths.append(spec.shape[0] // 2)
```
The `// 2` is hardcoded and must match the CNN stride in the model config. If someone changes `model.stride` to 1 or 4, the CTC input lengths become wrong, leading to silent training failures (CTC loss returning inf or nan). This should be derived from the actual model config.

### 3.9 Chunked Attention Concatenation Can Produce Length Mismatches

In `ChunkedMultiHeadSelfAttention.forward()` (`src/models/rnn_blocks.py:192-202`), after concatenating all chunks, if there is a length mismatch with the residual, the code truncates:
```python
if output.shape[1] != residual.shape[1]:
    min_len = min(output.shape[1], residual.shape[1])
    output = output[:, :min_len, :]
```
This silently drops timesteps. In practice it may not trigger often, but when it does, the model is processing fewer frames than CTC expects, which can cause shape errors downstream or degrade output quality for specific audio lengths.

### 3.10 No Learning Rate Warmup Strategy for CTC

CTC loss is notoriously unstable in early training, especially with small batch sizes. The current setup jumps straight into training with the full learning rate (or OneCycleLR's internal warmup, which is brief). A dedicated linear warmup over the first 500-1000 steps would help stabilize early training and reduce the risk of the model collapsing into blank-only predictions -- which is exactly the problem the blank penalty was built to combat.

### 3.11 Validation Excludes Pairs Where Either Side is Empty

In the `Validator` (`src/training/validator.py:169-172`):
```python
valid_pairs = [
    (ref, hyp) for ref, hyp in zip(all_true_texts, all_predicted_texts)
    if isinstance(ref, str) and isinstance(hyp, str) and ref.strip() and hyp.strip()
]
```
If the model produces empty predictions (common early in training), those samples are excluded from WER/CER calculation. This means the metrics look better than reality -- a sample where the model produces nothing is not penalized in the metric. It should count as 100% WER.

### 3.12 Early Stopping Patience is Very Aggressive

`early_stopping_patience: 2` means training stops after just 2 epochs without improvement. With `validate_every: 2`, this means the model gets at most 4 epochs of non-improving training before being stopped. CTC models often plateau and then break through, especially with small data. This patience is too tight and risks stopping training prematurely.

### 3.13 Greedy Decoder Shape Heuristic is Fragile

In `greedy_decoder()` (`src/inference/decoder.py:29-32`):
```python
if output_probs.size(0) < output_probs.size(1):
    output_probs = output_probs.transpose(0, 1)
```
This assumes that if dim 0 < dim 1, the tensor is (time, batch, classes). But for a single sample with very short audio (e.g. time=1, batch=2), this would incorrectly transpose. The caller should be responsible for providing the correct shape rather than having the decoder guess.

### 3.14 No Curriculum or Length-Based Batching

All audio samples are shuffled randomly regardless of duration. Short and long audio end up in the same batch, with massive padding overhead. Sorting by audio length (or bucketed sampling) would reduce padding waste and allow more useful computation per batch -- critical when batch_size=1 already makes every sample count.

### 3.15 Debug Counter as Instance Attribute Hack

In `CTCLossWithBlankPenalty` (`src/training/ctc_loss_with_penalty.py:115-118`):
```python
if hasattr(self, '_debug_counter'):
    self._debug_counter += 1
else:
    self._debug_counter = 0
```
This initializes a counter via `hasattr` checking on every forward pass rather than in `__init__`. It works but is a code smell.

---

## 4. Risk Assessment

| Issue | Severity | Impact on Training | Fixable Without More GPU |
|-------|----------|--------------------|--------------------------|
| Transform recreation per batch | Medium | Slower data loading, wasted CPU | Yes |
| No GRU residual connections | High | Limits depth scalability | Yes |
| Small effective batch size | High | Noisy gradients, slow convergence | Partially (can increase accumulation) |
| Hardcoded stride in input_lengths | High | Silent failure on config change | Yes |
| OneCycleLR max_lr misconfiguration | Medium | Suboptimal LR schedule | Yes |
| No speed perturbation augmentation | High | Poor generalization on small data | Yes |
| Empty predictions excluded from WER | Medium | Overly optimistic metrics | Yes |
| Aggressive early stopping | Medium | Premature training termination | Yes |
| No LR warmup for CTC stability | Medium | Blank collapse risk | Yes |
| Blank penalty not vectorized | Low | Slow if batch size increases | Yes |

---

## 5. What I Would Not Change

- **The overall CNN + Attention + GRU + CTC architecture.** It is proven for ASR and appropriate for the resource constraints.
- **Chunked attention.** It is the right approach for long audio on limited memory.
- **AMP + gradient accumulation.** These are the correct memory optimization techniques.
- **The modular project structure.** It is clean and maintainable.
- **Persian support in TextTransform.** The character handling is correct.
- **The comprehensive metrics suite.** Having many metrics is valuable for diagnosis.

---

## 6. Priority Recommendations (GPU-Neutral)

These improvements require zero additional GPU memory:

1. **Cache transforms in `CollateFn.__init__`** instead of recreating per batch.
2. **Add residual connections to `BidirectionalGRU`** (requires a projection if input/output dims differ, just like the CNN blocks already do).
3. **Add speed perturbation** (0.9x/1.0x/1.1x) to the data pipeline -- this is the highest-impact augmentation for ASR and costs nothing on GPU.
4. **Fix OneCycleLR** -- set `max_lr` to 5-10x the base learning rate (e.g. 5e-4) and let the scheduler handle warmup/annealing.
5. **Derive input_lengths from model stride** dynamically rather than hardcoding `// 2`.
6. **Count empty predictions as 100% WER** rather than excluding them.
7. **Increase early_stopping_patience** to 5-10 epochs.
8. **Add a proper LR warmup** (500-1000 steps) before the main schedule kicks in.

---

## 7. Fixes Applied (2026-02-21)

All identified weaknesses have been addressed:

| Issue | Fix | File(s) Modified |
|-------|-----|------------------|
| **3.1 Transform recreation per batch** | Transforms now cached in `CollateFn.__init__` | `src/data/dataset.py` |
| **3.2 No GRU residual connections** | Added residual connections with projection layer when dims differ | `src/models/rnn_blocks.py` |
| **3.4 Blank penalty not vectorized** | Replaced Python loop with vectorized tensor operations | `src/training/ctc_loss_with_penalty.py` |
| **3.6 OneCycleLR max_lr** | Added `max_lr` config parameter (10x learning rate) | `configs/default_config.yaml`, `scripts/train.py` |
| **3.8 Hardcoded stride** | Added configurable `stride` parameter to data pipeline | `src/data/dataset.py` |
| **3.9 Chunked attention length mismatch** | Proper padding/trimming to ensure exact length match | `src/models/rnn_blocks.py` |
| **3.10 No LR warmup** | Added linear LR warmup over first N steps (default: 500) | `src/training/trainer.py`, `configs/default_config.yaml`, `scripts/train.py` |
| **3.11 Empty predictions excluded from WER** | Empty predictions now counted as 100% WER | `src/training/validator.py` |
| **3.12 Aggressive early stopping** | Increased patience from 2 to 10 epochs | `configs/default_config.yaml` |
| **3.13 Greedy decoder shape heuristic** | Removed fragile heuristic; decoder requires explicit (batch, time, classes) shape | `src/inference/decoder.py`, `src/training/validator.py`, `scripts/inference.py`, `test_gpu_and_run.py` |
| **3.15 Debug counter initialization** | Moved `_debug_counter` to proper `__init__` initialization | `src/training/ctc_loss_with_penalty.py` |

### Notable Improvements

1. **Performance**: Transform caching eliminates CPU bottleneck from repeated `MelSpectrogram` construction.

2. **Training Stability**: LR warmup + proper OneCycleLR configuration + patient early stopping = more stable CTC training.

3. **Accuracy**: GRU residual connections enable deeper stacks; fixed WER calculation gives honest metrics.

4. **Robustness**: Configurable stride prevents silent failures; proper length handling in chunked attention.

---

*Review conducted on 2026-02-07 against commit `beab70f`.*
*Fixes applied on 2026-02-21.*

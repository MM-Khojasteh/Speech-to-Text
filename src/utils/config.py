"""
Configuration management utilities for loading YAML config files.

Configuration is loaded in two stages:

1. The YAML file (``configs/default_config.yaml``, local only) provides the
   base structure and dataset paths.
2. :func:`apply_env_overrides` overlays the tuned configuration matrix from
   ``STT_*`` environment variables (loaded from the git-ignored ``.env``).

The tuned values (hyperparameters, blank-penalty tier matrix, logic toggles)
are REQUIRED environment variables with no code-level fallbacks: the project
intentionally fails fast until the matrix is provided. See ``.env.example``
for the full documented matrix.
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from .env import EnvSource, load_env_file, optional_env, to_bool


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    n_cnn_layers: int = 3
    n_rnn_layers: int = 5
    rnn_dim: int = 512
    n_class: int = 29
    n_feats: int = 128
    stride: int = 2
    dropout: float = 0.1
    use_attention: bool = True
    n_attention_heads: int = 8
    use_transformer: bool = False
    use_se: bool = False
    use_chunked_attention: bool = True
    attention_chunk_size: int = 512
    attention_overlap: int = 64


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 10
    epochs: int = 1
    learning_rate: float = 5e-4
    max_lr: Optional[float] = None  # Peak LR for OneCycleLR; injected from STT_TRAIN_MAX_LR
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    validate_every: int = 1
    save_every: int = 10
    early_stopping_patience: Optional[int] = None
    scheduler_type: str = "onecycle"  # "onecycle", "cosine", "step", None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    warmup_steps: int = 0  # Linear LR warmup steps; injected from STT_TRAIN_WARMUP_STEPS


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_type: str = "librispeech"  # "librispeech" or "custom"
    train_root: str = "/kaggle/input/librispeech-clean"
    test_root: str = "/kaggle/input/librispeech-clean"
    train_url: str = "train-clean-100"
    test_url: str = "test-clean"
    # Custom dataset paths
    audio_files_dir: str = "dataset/audio_files"
    transcripts_dir: str = "dataset/info"
    test_audio_dir: Optional[str] = None  # Separate test set directory
    train_split: float = 0.8
    include_persian: bool = True
    # Audio processing
    sample_rate: int = 16000
    n_mels: int = 128
    freq_mask_param: int = 30
    time_mask_param: int = 100
    num_freq_masks: int = 2
    num_time_masks: int = 2


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 42


def apply_env_overrides(config: Config) -> Config:
    """
    Overlay the tuned configuration matrix from environment variables.

    Every variable below is REQUIRED and has no fallback in code: the
    repository intentionally ships without working training values. The
    local git-ignored ``.env`` file (or an injected environment) provides
    them; ``.env.example`` documents the full matrix. If any variable is
    missing, a single :class:`MissingEnvironmentError` is raised listing
    everything that must be set.

    Optional infrastructure variables (paths, device, worker counts) have
    safe generic defaults and may be omitted.
    """
    load_env_file()
    env = EnvSource()

    # --- Model architecture (tuned; no code-level defaults) ---
    config.model.n_cnn_layers = env.require("STT_MODEL_N_CNN_LAYERS", int)
    config.model.n_rnn_layers = env.require("STT_MODEL_N_RNN_LAYERS", int)
    config.model.rnn_dim = env.require("STT_MODEL_RNN_DIM", int)
    config.model.n_feats = env.require("STT_MODEL_N_FEATS", int)
    config.model.dropout = env.require("STT_MODEL_DROPOUT", float)
    config.model.use_attention = env.require("STT_MODEL_USE_ATTENTION", to_bool)
    config.model.n_attention_heads = env.require("STT_MODEL_N_ATTENTION_HEADS", int)
    config.model.use_transformer = env.require("STT_MODEL_USE_TRANSFORMER", to_bool)
    config.model.use_se = env.require("STT_MODEL_USE_SE", to_bool)
    config.model.use_chunked_attention = env.require(
        "STT_MODEL_USE_CHUNKED_ATTENTION", to_bool
    )
    config.model.attention_chunk_size = env.require(
        "STT_MODEL_ATTENTION_CHUNK_SIZE", int
    )
    config.model.attention_overlap = env.require("STT_MODEL_ATTENTION_OVERLAP", int)

    # --- Training schedule (tuned; no code-level defaults) ---
    config.training.batch_size = env.require("STT_TRAIN_BATCH_SIZE", int)
    config.training.epochs = env.require("STT_TRAIN_EPOCHS", int)
    config.training.learning_rate = env.require("STT_TRAIN_LEARNING_RATE", float)
    config.training.max_lr = env.require("STT_TRAIN_MAX_LR", float)
    config.training.weight_decay = env.require("STT_TRAIN_WEIGHT_DECAY", float)
    config.training.gradient_accumulation_steps = env.require(
        "STT_TRAIN_GRADIENT_ACCUMULATION_STEPS", int
    )
    config.training.warmup_steps = env.require("STT_TRAIN_WARMUP_STEPS", int)
    config.training.scheduler_type = env.require("STT_TRAIN_SCHEDULER_TYPE", str)
    config.training.early_stopping_patience = env.require(
        "STT_TRAIN_EARLY_STOPPING_PATIENCE", int
    )
    config.training.validate_every = env.require("STT_TRAIN_VALIDATE_EVERY", int)
    config.training.save_every = env.require("STT_TRAIN_SAVE_EVERY", int)

    # --- Data pipeline (tuned; no code-level defaults) ---
    config.data.dataset_type = env.require("STT_DATA_DATASET_TYPE", str)
    config.data.include_persian = env.require("STT_DATA_INCLUDE_PERSIAN", to_bool)
    config.data.train_split = env.require("STT_DATA_TRAIN_SPLIT", float)
    config.data.n_mels = env.require("STT_DATA_N_MELS", int)
    config.data.freq_mask_param = env.require("STT_DATA_FREQ_MASK_PARAM", int)
    config.data.time_mask_param = env.require("STT_DATA_TIME_MASK_PARAM", int)
    config.data.num_freq_masks = env.require("STT_DATA_NUM_FREQ_MASKS", int)
    config.data.num_time_masks = env.require("STT_DATA_NUM_TIME_MASKS", int)

    # --- Optional infrastructure overrides (safe generic defaults) ---
    config.training.max_grad_norm = optional_env(
        "STT_TRAIN_MAX_GRAD_NORM", config.training.max_grad_norm, float
    )
    config.training.use_amp = optional_env(
        "STT_TRAIN_USE_AMP", config.training.use_amp, to_bool
    )
    config.training.num_workers = optional_env(
        "STT_TRAIN_NUM_WORKERS", config.training.num_workers, int
    )
    config.training.pin_memory = optional_env(
        "STT_TRAIN_PIN_MEMORY", config.training.pin_memory, to_bool
    )
    config.data.sample_rate = optional_env(
        "STT_DATA_SAMPLE_RATE", config.data.sample_rate, int
    )
    config.data.audio_files_dir = optional_env(
        "STT_DATA_AUDIO_FILES_DIR", config.data.audio_files_dir, str
    )
    config.data.transcripts_dir = optional_env(
        "STT_DATA_TRANSCRIPTS_DIR", config.data.transcripts_dir, str
    )
    config.data.test_audio_dir = optional_env(
        "STT_DATA_TEST_AUDIO_DIR", config.data.test_audio_dir, str
    )
    config.data.train_root = optional_env(
        "STT_DATA_TRAIN_ROOT", config.data.train_root, str
    )
    config.data.test_root = optional_env(
        "STT_DATA_TEST_ROOT", config.data.test_root, str
    )
    config.checkpoint_dir = optional_env(
        "STT_CHECKPOINT_DIR", config.checkpoint_dir, str
    )
    config.log_dir = optional_env("STT_LOG_DIR", config.log_dir, str)
    config.device = optional_env("STT_DEVICE", config.device, str)
    config.seed = optional_env("STT_SEED", config.seed, int)

    env.finalize()
    return config


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded settings
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Parse configuration
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    
    # Create main config
    config = Config(
        model=model_config,
        training=training_config,
        data=data_config,
        checkpoint_dir=config_dict.get('checkpoint_dir', 'checkpoints'),
        log_dir=config_dict.get('log_dir', 'logs'),
        device=config_dict.get('device', 'cuda'),
        seed=config_dict.get('seed', 42)
    )

    # Overlay the tuned configuration matrix from the environment (.env).
    # Raises MissingEnvironmentError listing every variable that must be set.
    config = apply_env_overrides(config)

    return config


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save
        config_path: Path to save YAML file
    """
    config_dict = {
        'model': {
            'n_cnn_layers': config.model.n_cnn_layers,
            'n_rnn_layers': config.model.n_rnn_layers,
            'rnn_dim': config.model.rnn_dim,
            'n_class': config.model.n_class,
            'n_feats': config.model.n_feats,
            'stride': config.model.stride,
            'dropout': config.model.dropout,
            'use_attention': config.model.use_attention,
            'n_attention_heads': config.model.n_attention_heads,
            'use_transformer': config.model.use_transformer,
            'use_se': config.model.use_se,
            'use_chunked_attention': config.model.use_chunked_attention,
            'attention_chunk_size': config.model.attention_chunk_size,
            'attention_overlap': config.model.attention_overlap
        },
        'training': {
            'batch_size': config.training.batch_size,
            'epochs': config.training.epochs,
            'learning_rate': config.training.learning_rate,
            'max_lr': config.training.max_lr,
            'weight_decay': config.training.weight_decay,
            'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
            'max_grad_norm': config.training.max_grad_norm,
            'use_amp': config.training.use_amp,
            'num_workers': config.training.num_workers,
            'pin_memory': config.training.pin_memory,
            'validate_every': config.training.validate_every,
            'save_every': config.training.save_every,
            'early_stopping_patience': config.training.early_stopping_patience,
            'scheduler_type': config.training.scheduler_type,
            'scheduler_params': config.training.scheduler_params,
            'warmup_steps': config.training.warmup_steps
        },
        'data': {
            'dataset_type': config.data.dataset_type,
            'train_root': config.data.train_root,
            'test_root': config.data.test_root,
            'train_url': config.data.train_url,
            'test_url': config.data.test_url,
            'audio_files_dir': config.data.audio_files_dir,
            'transcripts_dir': config.data.transcripts_dir,
            'test_audio_dir': config.data.test_audio_dir,
            'train_split': config.data.train_split,
            'include_persian': config.data.include_persian,
            'sample_rate': config.data.sample_rate,
            'n_mels': config.data.n_mels,
            'freq_mask_param': config.data.freq_mask_param,
            'time_mask_param': config.data.time_mask_param,
            'num_freq_masks': config.data.num_freq_masks,
            'num_time_masks': config.data.num_time_masks
        },
        'checkpoint_dir': config.checkpoint_dir,
        'log_dir': config.log_dir,
        'device': config.device,
        'seed': config.seed
    }
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


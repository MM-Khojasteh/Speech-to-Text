"""
Configuration management utilities for loading YAML config files.
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field


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


@dataclass
class DataConfig:
    """Data configuration."""
    train_root: str = "/kaggle/input/librispeech-clean"
    test_root: str = "/kaggle/input/librispeech-clean"
    train_url: str = "train-clean-100"
    test_url: str = "test-clean"
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
            'scheduler_params': config.training.scheduler_params
        },
        'data': {
            'train_root': config.data.train_root,
            'test_root': config.data.test_root,
            'train_url': config.data.train_url,
            'test_url': config.data.test_url,
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


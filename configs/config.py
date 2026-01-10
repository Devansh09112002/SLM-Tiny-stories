from dataclasses import dataclass, field
from typing import Dict

@dataclass
class DataConfig:
    name: str = "roneneldan/TinyStories"
    num_workers: int = 4
    batch_size: int = 16
    max_length: int = 512 # For Transformer-based models
    cache_dir: str = ".cache"

@dataclass
class ModelConfig:
    
    # Common params set dynamically
    
    vocab_size: int = 10000 
    pad_token_id: int = 0
    bos_token_id: int = 1  # Add BOS token ID for compressor model
    
    # Transformer / Contrastive Params
    
    transformer: Dict = field(default_factory=lambda: {
        "hidden_size": 256,
        "layers": 6,
        "heads": 4,
        "feedforward_dim": 1024
    })

    # Compressor Params
    
    compressor: Dict = field(default_factory=lambda: {
        "n_positions": 512,
        "d_model": 256,
        "n_head": 4,
        "encoder_layers": 4,
        "trunk_layers": 8,
        "decoder_layers": 4,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "compression_ratio": 2
    })
    
    # Linear Model Params
    
    linear: Dict = field(default_factory=lambda: {
        "context_length": 64,
        "embedding_dim": 256,
        "use_relu": True
    })

@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 4
    
    # Contrastive specific
    
    contrastive_alpha: float = 5.0

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

# Create a default config instance

config = Config()
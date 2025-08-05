#!/usr/bin/env python3
"""Configuration management for Amharic H-Net project."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration settings."""
    name: str = "amharic-hnet"
    vocab_size: int = 8000
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_length: int = 512
    model_path: str = "models/amharic_hnet.pt"
    tokenizer_path: str = "models/tokenizer/amharic_vocab.json"


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    cors_origins: list = field(default_factory=lambda: ["*"])
    rate_limit_requests_per_minute: int = 60
    max_request_size: int = 1024 * 1024  # 1MB
    request_timeout: int = 30
    enable_metrics: bool = True
    enable_auth: bool = False
    auth_token: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 5e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    seed: int = 42


@dataclass
class DataConfig:
    """Data processing configuration."""
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    training_data_dir: str = "data/training"
    min_text_length: int = 10
    max_text_length: int = 1000
    min_amharic_ratio: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1


@dataclass
class GenerationConfig:
    """Text generation configuration."""
    max_length: int = 200
    min_length: int = 10
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    early_stopping: bool = True


@dataclass
class ProjectConfig:
    """Main project configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Environment settings
    environment: str = "development"  # development, production, testing
    debug: bool = True
    log_level: str = "INFO"
    
    # Paths
    project_root: str = str(Path(__file__).parent)
    logs_dir: str = "logs"
    outputs_dir: str = "outputs"
    models_dir: str = "models"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ProjectConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Create nested dataclass instances
        config = cls()
        
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'api' in config_dict:
            config.api = APIConfig(**config_dict['api'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'generation' in config_dict:
            config.generation = GenerationConfig(**config_dict['generation'])
        
        # Set top-level attributes
        for key, value in config_dict.items():
            if key not in ['model', 'api', 'training', 'data', 'generation']:
                setattr(config, key, value)
        
        return config
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'environment': self.environment,
            'debug': self.debug,
            'log_level': self.log_level,
            'project_root': self.project_root,
            'logs_dir': self.logs_dir,
            'outputs_dir': self.outputs_dir,
            'models_dir': self.models_dir,
            'model': self.model.__dict__,
            'api': self.api.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'generation': self.generation.__dict__,
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def get_env_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        env_configs = {
            'development': {
                'debug': True,
                'log_level': 'DEBUG',
                'api.reload': True,
                'api.workers': 1,
            },
            'production': {
                'debug': False,
                'log_level': 'INFO',
                'api.reload': False,
                'api.workers': 4,
                'api.enable_auth': True,
            },
            'testing': {
                'debug': True,
                'log_level': 'DEBUG',
                'api.port': 8001,
                'training.num_epochs': 1,
            }
        }
        
        return env_configs.get(self.environment, {})
    
    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # API overrides
        if os.getenv('API_HOST'):
            self.api.host = os.getenv('API_HOST')
        if os.getenv('API_PORT'):
            self.api.port = int(os.getenv('API_PORT'))
        if os.getenv('API_WORKERS'):
            self.api.workers = int(os.getenv('API_WORKERS'))
        
        # Model overrides
        if os.getenv('MODEL_PATH'):
            self.model.model_path = os.getenv('MODEL_PATH')
        if os.getenv('TOKENIZER_PATH'):
            self.model.tokenizer_path = os.getenv('TOKENIZER_PATH')
        
        # Environment override
        if os.getenv('ENVIRONMENT'):
            self.environment = os.getenv('ENVIRONMENT')
        
        # Apply environment-specific configs
        env_config = self.get_env_config()
        for key, value in env_config.items():
            if '.' in key:
                # Handle nested attributes like 'api.reload'
                obj_name, attr_name = key.split('.', 1)
                obj = getattr(self, obj_name)
                setattr(obj, attr_name, value)
            else:
                setattr(self, key, value)


def get_config(config_path: Optional[str] = None) -> ProjectConfig:
    """Get project configuration."""
    if config_path is None:
        # Try to find config file
        possible_paths = [
            'config.yaml',
            'configs/production.yaml',
            'configs/base.yaml',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        config = ProjectConfig.from_yaml(config_path)
    else:
        config = ProjectConfig()
    
    # Apply environment overrides
    config.apply_env_overrides()
    
    return config


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print(f"Environment: {config.environment}")
    print(f"API Port: {config.api.port}")
    print(f"Model Path: {config.model.model_path}")
    
    # Save example config
    config.to_yaml('example_config.yaml')
    print("Example configuration saved to example_config.yaml")
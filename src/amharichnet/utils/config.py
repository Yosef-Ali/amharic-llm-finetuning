from pydantic import BaseModel
from typing import Optional

class DataConfig(BaseModel):
    train_path: str = "data/train.jsonl"
    val_path: Optional[str] = "data/val.jsonl"
    tokenizer: str = "default"
    batch_size: int = 16
    num_workers: int = 4

class ModelConfig(BaseModel):
    name: str = "hnet-compact"
    hidden_dim: int = 512
    num_layers: int = 12
    checkpoint: Optional[str] = None

class TrainConfig(BaseModel):
    seed: int = 1337
    epochs: int = 1
    lr: float = 5e-4
    weight_decay: float = 0.01
    precision: str = "fp16"
    device: str = "auto"
    output_dir: str = "outputs/run"

class Config(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

from pathlib import Path
import yaml
from amharichnet.cli import main

def test_training_scaffold(tmp_path):
    cfg = {
        'data': {'train_path': 'data/train.jsonl', 'val_path': 'data/val.jsonl', 'tokenizer': 'default', 'batch_size': 2, 'num_workers': 0},
        'model': {'name': 'hnet-compact', 'hidden_dim': 64, 'num_layers': 2, 'checkpoint': None},
        'train': {'seed': 123, 'epochs': 1, 'lr': 0.001, 'weight_decay': 0.0, 'precision': 'fp32', 'device': 'cpu', 'output_dir': str(tmp_path / 'out')},
    }
    cfg_path = tmp_path / 'cfg.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg))
    main(["train", "--config", str(cfg_path)])
    assert (tmp_path / 'out' / 'used_config.txt').exists()

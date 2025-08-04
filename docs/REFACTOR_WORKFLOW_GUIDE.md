# REFACTOR WORKFLOW GUIDE: Amharic H-Net Clean Implementation

This guide explains how to use the **refactored clean implementation** in `src/amharichnet/`: run training, resume from checkpoints, run tests, and extend the codebase. It assumes the new package layout and CLI-first approach implemented in August 2024.

## 🎯 What This Guide Covers

This is the **post-refactor workflow guide** for the clean `src/amharichnet/` implementation. For other aspects of the project, see:
- `docs/IMPLEMENTATION_GUIDE.md` - Original comprehensive implementation
- `docs/ENVIRONMENT_SETUP.md` - Environment configuration  
- `docs/PROJECT_STRUCTURE.md` - Overall project architecture

## 1) Project Layout (Post-Refactor)

```
Amharic-Hnet-Qwin/
├── configs/
│   └── base.yaml
├── src/
│   └── amharichnet/
│       ├── cli.py                 # Command-line interface
│       ├── utils/
│       │   ├── config.py          # Pydantic configuration
│       │   └── repro.py           # Deterministic seeding
│       ├── data/
│       │   ├── jsonl.py           # JSONL reading
│       │   ├── loader.py          # DataLoader creation
│       │   └── tokenizer.py       # Basic tokenizer
│       ├── models/
│       │   └── hnet.py            # TinyHNet model
│       └── train/
│           └── training_loop.py   # Training pipeline
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   └── test_cli.py
│   └── smoke/
│       └── test_training_scaffold.py
├── data/                          # Organized datasets
│   ├── raw/collected_articles/    # 962 original articles
│   ├── processed/                 # Cleaned data
│   └── training/                  # Training-ready data
├── legacy/                        # Archived implementations
└── outputs/                       # Created on first run
```

**Key Roles:**
- `configs/`: YAML configuration files (data, model, training)
- `src/amharichnet/`: Clean, installable package with CLI
- `outputs/`: Generated on first run; stores metrics and checkpoints
- `tests/`: Unit + smoke tests (CPU-friendly)
- `data/`: Organized datasets from reorganization
- `legacy/`: All previous implementations safely preserved

## 2) Installation Options

**Prerequisites:** Python 3.10+ environment

### Option A: Minimal Development Setup
```bash
python -m pip install --upgrade pip
pip install pyyaml pydantic pytest
# Optional for actual PyTorch operations:
pip install torch
```

### Option B: Editable Install (Recommended)
First, ensure `pyproject.toml` exists in root:
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "amharichnet"
version = "0.1.0"
description = "Clean Amharic H-Net implementation"
requires-python = ">=3.10"
dependencies = ["pydantic", "pyyaml"]

[tool.setuptools.packages.find]
where = ["src"]
```

Then install:
```bash
pip install -e .
```

## 3) Configuration

Edit `configs/base.yaml`:
```yaml
data:
  train_path: data/training/train.jsonl    # Use organized data
  val_path: data/training/val.jsonl
  tokenizer: default
  batch_size: 16
  num_workers: 4

model:
  name: hnet-compact
  hidden_dim: 512
  num_layers: 12
  checkpoint:  # Set for resume: outputs/run/checkpoints/ckpt.pt

train:
  seed: 1337
  epochs: 1
  lr: 0.0005
  weight_decay: 0.01
  precision: fp16
  device: auto
  output_dir: outputs/run
```

**Notes:**
- If data paths are missing, training falls back to synthetic data for testing
- `output_dir` will be created automatically
- Use actual data from `data/raw/collected_articles/` (962 articles available)

## 4) Running the CLI

### Without Editable Install:
```bash
# From repo root
PYTHONPATH=src python -m amharichnet.cli --help
PYTHONPATH=src python -m amharichnet.cli train --config configs/base.yaml
```

### With Editable Install:
```bash
python -m amharichnet.cli --help
python -m amharichnet.cli train --config configs/base.yaml
```

### Expected Output Structure:
```
outputs/run/
├── used_config.txt              # Config used for this run
├── metrics.json                 # Training metrics
└── checkpoints/
    ├── meta.json               # Checkpoint metadata
    └── ckpt.pt                 # Model checkpoint (if torch installed)
```

**Example `metrics.json`:**
```json
{
  "steps": 150,
  "final_loss": 2.456,
  "val_loss": 2.678
}
```

## 5) Resume Training

Set the checkpoint path in your config:
```yaml
model:
  checkpoint: outputs/run/checkpoints/ckpt.pt
```

Run training again:
```bash
PYTHONPATH=src python -m amharichnet.cli train --config configs/base.yaml
```

**Expected output:**
```
[RESUME] loaded outputs/run/checkpoints/ckpt.pt
```

**Tip:** Create `configs/resume.yaml` to keep base config clean:
```yaml
# configs/resume.yaml
# Inherit from base but specify checkpoint
data:
  train_path: data/training/train.jsonl
  val_path: data/training/val.jsonl
  batch_size: 16

model:
  name: hnet-compact
  hidden_dim: 512
  num_layers: 12
  checkpoint: outputs/run/checkpoints/ckpt.pt  # Resume from here

train:
  seed: 1337
  epochs: 5    # Continue for more epochs
  lr: 0.0001   # Lower learning rate for fine-tuning
```

## 6) Testing

### Unit Tests:
```bash
pytest -q tests/unit
```

### Smoke Tests:
```bash
pytest -q tests/smoke
```

### All Tests:
```bash
pytest -q tests/
```

### Heavy Tests (if added):
```bash
pytest -q --heavy
# or set environment variable:
RUN_HEAVY_TESTS=1 pytest -q tests/
```

## 7) Current Scaffold Capabilities

The clean implementation provides:

- ✅ **Deterministic seeding** (`src/amharichnet/utils/repro.py`)
- ✅ **Type-safe config parsing** via Pydantic (`src/amharichnet/utils/config.py`)
- ✅ **Data loading pipeline** (JSONL → token IDs) with simple tokenizer
- ✅ **DataLoader + toy model** (TinyHNet) for CPU-friendly training
- ✅ **Training loop** with step tracking and loss monitoring
- ✅ **Validation pass** on validation data with metrics
- ✅ **Checkpoint system** (saving and resuming)
- ✅ **Configuration-driven** workflow
- ✅ **CLI interface** for easy execution

## 8) Extension Roadmap

Replace placeholder components step-by-step:

### Phase 1: Real Tokenization (`src/amharichnet/data/`)
```python
# 1. Replace BasicCharTokenizer in tokenizer.py
# 2. Update loader.py collate_fn for proper batching
# 3. Generate input_ids, attention_mask, labels tensors
```

### Phase 2: H-Net Model (`src/amharichnet/models/`)
```python
# 1. Replace TinyHNet with real H-Net architecture
# 2. Implement proper attention mechanisms
# 3. Add Amharic-specific model components
```

### Phase 3: Advanced Training (`src/amharichnet/train/`)
```python
# 1. Implement proper loss computation (cross-entropy)
# 2. Add periodic checkpointing (every N steps)
# 3. Best checkpoint selection based on validation metrics
# 4. Integration with Weights & Biases or TensorBoard
```

### Phase 4: Evaluation & Inference
```python
# 1. Add evaluation CLI command:
#    python -m amharichnet.cli eval --config configs/eval.yaml
# 2. Add inference module and CLI:
#    python -m amharichnet.cli infer --prompt "ሰላም"
# 3. Optional: FastAPI server for deployment
```

### Phase 5: Advanced Configs
Create specialized configurations:
```
configs/
├── base.yaml                    # Basic training
├── train_small.yaml            # Small model config
├── train_large.yaml            # Large model config  
├── eval.yaml                   # Evaluation config
├── inference.yaml              # Inference config
└── production.yaml             # Production deployment
```

## 9) Data Integration

Leverage the organized data from refactor:

### Use Collected Articles:
```bash
# 962 articles available in data/raw/collected_articles/
ls data/raw/collected_articles/ | wc -l
# 962

# Process for training:
python scripts/prepare_training_data.py \
  --input data/raw/collected_articles/ \
  --output data/training/ \
  --train-split 0.8 \
  --val-split 0.1 \
  --test-split 0.1
```

### Update Config:
```yaml
data:
  train_path: data/training/train.jsonl
  val_path: data/training/val.jsonl
  test_path: data/training/test.jsonl
```

## 10) CI/CD Integration

The refactor includes `.github/workflows/ci.yml` for automated testing.

### Extend with Code Quality:
```yaml
# Add to .github/workflows/ci.yml
- name: Code Quality
  run: |
    pip install ruff black mypy
    ruff check .
    black --check .
    mypy src
```

### Add to `pyproject.toml`:
```toml
[tool.ruff]
target-version = "py310"
select = ["E", "F", "I"]

[tool.black]
target-version = ["py310"]
line-length = 88

[tool.mypy]
python_version = "3.10"
strict = true
```

## 11) Troubleshooting

### Common Issues:

**ModuleNotFoundError: amharichnet**
```bash
# Solution 1: Use PYTHONPATH
PYTHONPATH=src python -m amharichnet.cli

# Solution 2: Install package
pip install -e .
```

**RuntimeError: pyyaml is required**
```bash
pip install pyyaml pydantic
```

**No ckpt.pt generated**
```bash
# Install PyTorch for checkpoint saving
pip install torch
# Without torch, metrics are still saved but no .pt files
```

**Resume not loading**
```bash
# Check config path:
ls -la outputs/run/checkpoints/ckpt.pt

# Verify config:
grep -A5 "model:" configs/base.yaml
# Should show: checkpoint: outputs/run/checkpoints/ckpt.pt
```

**Legacy code conflicts**
```bash
# All legacy code is safely archived in legacy/
# To reference old implementations:
ls legacy/amharic-hnet/           # Original comprehensive implementation
ls legacy/old_training_scripts/   # Previous training approaches
ls legacy/experiments/            # Experimental utilities
```

## 12) Migration from Legacy Code

### If coming from legacy implementations:

1. **Your data is preserved**: `data/raw/collected_articles/` (962 articles)
2. **Legacy code accessible**: `legacy/amharic-hnet/` for reference
3. **Clean starting point**: `src/amharichnet/` for new development
4. **Configuration-driven**: Use YAML configs instead of hardcoded parameters

### Migration Strategy:
```bash
# 1. Start with clean implementation:
PYTHONPATH=src python -m amharichnet.cli train --config configs/base.yaml

# 2. Reference legacy for complex logic:
# Check legacy/amharic-hnet/ for specific implementations
# Copy relevant functions to src/amharichnet/ modules

# 3. Use organized data:
# Your 962 articles are in data/raw/collected_articles/
# Process them with the new pipeline

# 4. Extend systematically:
# Follow the extension roadmap above
# Keep configuration-driven approach
```

---

## ✅ Summary

This refactored implementation provides:
- **Clean architecture** with proper separation of concerns
- **CLI-first interface** for reproducible workflows  
- **Configuration-driven** training and evaluation
- **Resume capabilities** with automatic checkpoint management
- **Testing framework** for reliable development
- **Extension roadmap** for scaling to production
- **Data organization** with 962 articles ready for use
- **Legacy preservation** for reference and migration

Follow this guide to work with the **clean, maintainable implementation** while having access to all previous work for reference.

**Next Steps:**
1. Run your first training: `PYTHONPATH=src python -m amharichnet.cli train --config configs/base.yaml`  
2. Explore the data: `ls data/raw/collected_articles/`
3. Reference legacy implementations: `ls legacy/amharic-hnet/`
4. Extend systematically following the roadmap above

*This guide covers the refactored clean implementation. For comprehensive legacy documentation, see the other guides in the `docs/` directory.*
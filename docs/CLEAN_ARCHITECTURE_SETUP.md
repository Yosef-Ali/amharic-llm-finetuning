# CLEAN ARCHITECTURE SETUP GUIDE

Quick setup guide for the **clean implementation** in `src/amharichnet/` after the August 2024 refactor.

## 🎯 Purpose

This guide covers:
- Installing the clean `src/amharichnet/` package
- Running the CLI interface  
- Basic configuration and usage
- Quick verification

For comprehensive workflow details, see `docs/REFACTOR_WORKFLOW_GUIDE.md`.

## ⚡ Quick Setup

### 1. Prerequisites
```bash
# Ensure Python 3.10+
python --version
# Python 3.10.0+

# Clone and navigate (if not already done)
cd Amharic-Hnet-Qwin
```

### 2. Install Dependencies
```bash
# Minimal setup
pip install pyyaml pydantic pytest

# For full PyTorch support
pip install torch
```

### 3. Verify Installation
```bash
# Test CLI access
PYTHONPATH=src python -m amharichnet.cli --help

# Expected output:
# usage: cli.py [-h] {train} ...
# Amharic H-Net CLI
```

### 4. Run First Training
```bash
# Train with default config
PYTHONPATH=src python -m amharichnet.cli train --config configs/base.yaml

# Check outputs
ls outputs/run/
# expected: used_config.txt, metrics.json, checkpoints/
```

## 🔧 Installation Options

### Option A: Development Mode (Recommended)

Create `pyproject.toml` in project root:
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

[project.optional-dependencies]
dev = ["pytest", "torch"]
```

Install:
```bash
pip install -e .

# Now you can use without PYTHONPATH:
python -m amharichnet.cli --help
```

### Option B: Direct Usage
```bash
# Always use PYTHONPATH for commands:
PYTHONPATH=src python -m amharichnet.cli train --config configs/base.yaml
```

## ⚙️ Configuration

### Basic Config (`configs/base.yaml`)
```yaml
data:
  train_path: data/training/train.jsonl
  val_path: data/training/val.jsonl
  batch_size: 8
  num_workers: 2

model:
  name: "TinyHNet"
  hidden_dim: 256
  num_layers: 4
  checkpoint: null  # Set for resume

train:
  seed: 1337
  epochs: 2
  lr: 0.001
  output_dir: "outputs"
```

### Using Real Data
```yaml
data:
  # Use the 962 collected articles
  train_path: "data/raw/collected_articles/processed_train.jsonl"
  val_path: "data/raw/collected_articles/processed_val.jsonl"
```

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/unit/

# Smoke tests  
pytest tests/smoke/

# All tests
pytest tests/
```

### Verify Clean Architecture
```bash
# Check package structure
ls src/amharichnet/
# Expected: cli.py, data/, models/, train/, utils/

# Check CLI functionality
PYTHONPATH=src python -c "from amharichnet.cli import main; print('✅ CLI imports work')"

# Check training pipeline
PYTHONPATH=src python -c "from amharichnet.train.training_loop import run_training; print('✅ Training pipeline accessible')"
```

## 📊 Data Access

### Use Organized Data
```bash
# Check available data
ls data/raw/collected_articles/ | head -5
# article_0001.json, article_0002.json, ...

wc -l data/raw/collected_articles/*.json | tail -1
# Total articles available

# Check processed data
ls data/processed/
# Cleaned and processed articles
```

### Prepare Training Data
```bash
# If you need to process raw articles for training:
# (This is a placeholder - implement your data processing script)
python scripts/prepare_training_data.py \
  --input data/raw/collected_articles/ \
  --output data/training/ \
  --format jsonl
```

## 🔄 Resume Training

### Save Progress
Training automatically saves to `outputs/run/checkpoints/`

### Resume
```yaml
# In configs/resume.yaml
model:
  checkpoint: "outputs/run/checkpoints/ckpt.pt"

train:
  epochs: 10  # Continue for more epochs
```

```bash
PYTHONPATH=src python -m amharichnet.cli train --config configs/resume.yaml
```

## 🚨 Troubleshooting

### Common Issues

**Command not found: `python -m amharichnet.cli`**
```bash
# Solution: Use PYTHONPATH
PYTHONPATH=src python -m amharichnet.cli --help

# Or install package:
pip install -e .
```

**ImportError: No module named 'pydantic'**
```bash
pip install pydantic pyyaml
```

**No data files found**
```bash
# Check data organization:
ls data/raw/collected_articles/ | wc -l
# Should show 962 articles

# If missing, check if data is in legacy:
ls legacy/experiments/collected_articles/ 2>/dev/null || echo "Check legacy locations"
```

**Checkpoint not loading**
```bash
# Verify checkpoint exists:
ls -la outputs/run/checkpoints/

# Check config:
grep -A3 "model:" configs/base.yaml
```

### Package Structure Issues
```bash
# Verify clean structure:
find src/amharichnet -name "*.py" | head -10
# Should show organized module structure

# Check for conflicts:
python -c "import sys; print([p for p in sys.path if 'amharichnet' in p])"
```

## 📁 Directory Navigation

### Project Layout After Refactor
```bash
# Main implementation
src/amharichnet/           # ← Work here for new development

# Configuration
configs/                   # ← YAML configuration files

# Data (organized)
data/raw/collected_articles/    # ← 962 articles ready to use
data/processed/                 # ← Cleaned data
data/training/                  # ← Training-ready splits

# Legacy (reference only)
legacy/amharic-hnet/       # ← Original comprehensive implementation
legacy/old_training_scripts/   # ← Previous training approaches
legacy/experiments/            # ← Utilities and experiments

# Generated
outputs/                   # ← Created on first training run
```

### Key Commands
```bash
# Work with clean implementation
PYTHONPATH=src python -m amharichnet.cli train --config configs/base.yaml

# Reference legacy implementations
ls legacy/amharic-hnet/    # Comprehensive original code
cat legacy/amharic-hnet/README.md

# Check available data
ls data/raw/collected_articles/ | wc -l  # 962 articles
```

## ✅ Verification Checklist

After setup, verify:

- [ ] `PYTHONPATH=src python -m amharichnet.cli --help` works
- [ ] `configs/base.yaml` exists and is readable
- [ ] `pytest tests/unit/` passes
- [ ] Training creates `outputs/run/` directory
- [ ] Data is accessible in `data/raw/collected_articles/`
- [ ] Legacy code is preserved in `legacy/`

## 🚀 Next Steps

1. **Complete setup**: Follow this guide to get the CLI working
2. **First training run**: Use `configs/base.yaml` with synthetic data
3. **Explore data**: Check the 962 articles in `data/raw/collected_articles/`
4. **Read comprehensive guide**: See `docs/REFACTOR_WORKFLOW_GUIDE.md`
5. **Reference legacy**: Explore `legacy/amharic-hnet/` for implementations
6. **Extend systematically**: Follow the extension roadmap

---

**🎯 Goal**: Get the clean `src/amharichnet/` implementation running quickly

**📚 For detailed workflows**: See `docs/REFACTOR_WORKFLOW_GUIDE.md`

**🗄️ For legacy reference**: Explore `legacy/` directories
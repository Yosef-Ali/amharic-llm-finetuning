# 🇪🇹 Amharic H-Net Qwin

A production-ready Amharic language model implementation with clean architecture and comprehensive training pipeline.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd Amharic-Hnet-Qwin

# Install dependencies
pip install pyyaml pydantic pytest

# Train the model
PYTHONPATH=src python -m amharichnet.cli train --config configs/base.yaml

# Check outputs
ls -la outputs/run/
cat outputs/run/metrics.json
```

## 📁 Project Structure

```
Amharic-Hnet-Qwin/
├── src/amharichnet/           # 🎯 Clean implementation (main)
│   ├── cli.py                 # Command-line interface
│   ├── data/                  # Data loading and processing
│   ├── models/                # Model architectures
│   ├── train/                 # Training pipeline
│   └── utils/                 # Configuration and utilities
├── configs/                   # ⚙️ Configuration files
│   └── base.yaml
├── tests/                     # 🧪 Test suite
│   ├── unit/                  # Unit tests
│   └── smoke/                 # Integration tests
├── data/                      # 📊 Organized datasets
│   ├── raw/collected_articles/ # Original scraped data (962 articles)
│   ├── processed/             # Cleaned and processed data
│   └── training/              # Training-ready datasets
├── models/                    # 🤖 Model checkpoints and artifacts
├── docs/                      # 📚 All documentation
├── legacy/                    # 🗄️ Archived implementations
│   ├── amharic-hnet/         # Original implementation
│   ├── old_training_scripts/ # Legacy training scripts
│   └── experiments/          # Experimental code and utilities
└── [Essential files only]    # Clean root directory
```

## 🎯 Core Features

### Clean Implementation (`src/amharichnet/`)
- **Single entry point**: CLI-based interface
- **Modular design**: Separate data, models, training, utils
- **Configuration-driven**: YAML-based configs
- **Reproducible**: Deterministic seeding and checkpointing
- **Resume support**: Automatic checkpoint loading

### Training Pipeline
```bash
# Basic training
python -m amharichnet.cli train --config configs/base.yaml

# Check training progress
cat outputs/run/metrics.json
# {"steps": 100, "final_loss": 2.45, "val_loss": 2.67}

# Resume from checkpoint
# Set model.checkpoint: outputs/run/checkpoints/ckpt.pt in config
python -m amharichnet.cli train --config configs/base.yaml
```

### Data Organization
- **Raw Data**: 962 collected Amharic articles
- **Processed Data**: Cleaned and formatted for training
- **Training Data**: Ready-to-use datasets with proper splits

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/unit/      # Unit tests
pytest tests/smoke/     # Integration tests
```

## 📊 Data Statistics

- **Articles Collected**: 962 raw articles
- **Processing Pipeline**: Automated cleaning and validation
- **Training Ready**: Structured datasets in `data/` directory
- **Legacy Preserved**: All original work archived in `legacy/`

## 🏗️ Architecture

### Main Implementation
- **src/amharichnet/**: Modern, clean implementation
- **Pydantic configs**: Type-safe configuration management
- **Modular design**: Easy to extend and maintain
- **CLI interface**: Simple command-line usage

### Legacy Archive
- **legacy/amharic-hnet/**: Original comprehensive implementation
- **legacy/old_training_scripts/**: Various training approaches
- **legacy/experiments/**: Experimental code and utilities

## 🔧 Configuration

Edit `configs/base.yaml`:

```yaml
data:
  train_path: "data/training/train.jsonl"
  val_path: "data/training/val.jsonl"
  batch_size: 8

model:
  name: "TinyHNet"
  hidden_dim: 256
  num_layers: 4
  checkpoint: null  # Set to checkpoint path to resume

train:
  epochs: 10
  lr: 0.001
  output_dir: "outputs"
```

## 📚 Documentation

### 🎯 **Start Here (Post-Refactor)**
- **[Clean Architecture Setup](docs/CLEAN_ARCHITECTURE_SETUP.md)**: Quick setup for the `src/amharichnet/` implementation
- **[Refactor Workflow Guide](docs/REFACTOR_WORKFLOW_GUIDE.md)**: Complete step-by-step workflow (train, resume, test, extend)

### 📖 **Comprehensive Documentation**
- `docs/README.md`: Detailed project documentation  
- `docs/IMPLEMENTATION_GUIDE.md`: Implementation details
- `docs/PROJECT_STRUCTURE.md`: Architecture overview
- `docs/ENVIRONMENT_SETUP.md`: Environment configuration
- And 10+ other guides for comprehensive coverage

## 🚀 Next Steps

1. **Train your model**: `python -m amharichnet.cli train --config configs/base.yaml`
2. **Explore legacy code**: Check `legacy/` for comprehensive implementations
3. **Extend functionality**: Add inference and evaluation CLI commands
4. **Scale up**: Use larger datasets from `data/raw/collected_articles/`

## 📄 License

MIT License - see `LICENSE` file for details.

---

**🎯 Clean Architecture • 📊 Rich Data • 🧪 Well Tested • 🗄️ Legacy Preserved**

*Building production-ready Amharic AI with clean, maintainable code.*
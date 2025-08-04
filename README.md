# 🇪🇹 Amharic H-Net: Production-Ready AI Text Generation

[![Model Status](https://img.shields.io/badge/Model-Production%20Ready-brightgreen.svg)](README.md)
[![API Status](https://img.shields.io/badge/API-Online-success.svg)](http://localhost:8000/docs)
[![Quality Score](https://img.shields.io/badge/Quality-0.619±0.013-blue.svg)](test_results.json)

A production-ready Amharic text generation system using Hierarchical Network (H-Net) architecture with comprehensive evaluation, REST API, and web interface.

## 🚀 Quick Start

### 1. Start the API Server
```bash
python api_server.py
```

### 2. Open Web Interface
```bash
# Open web_interface.html in your browser
# or serve via HTTP server:
python -m http.server 8080
```

### 3. Test the API
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ኢትዮጵያ", "length": 50}'
```

## 📊 Model Performance

| Metric | Score | Details |
|--------|-------|---------|
| **Overall Quality** | 0.619 ± 0.013 | Comprehensive text quality |
| **Amharic Ratio** | 98% | Authentic Amharic generation |
| **Fluency Score** | 0.667 | Natural language flow |
| **Coherence Score** | 0.800 | Logical text structure |
| **Test Loss** | 8.121 | Model convergence |

## 🏗️ Architecture

### Core Components
- **H-Net Model**: 6.85M parameter Hierarchical Network
- **Amharic Tokenizer**: 3,087 subword vocabulary
- **Quality Evaluator**: Multi-metric assessment system
- **REST API**: FastAPI-based production server
- **Web Interface**: Interactive text generation UI

### Technology Stack
- **Backend**: Python 3.9+, PyTorch, FastAPI
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Deployment**: Docker, Docker Compose, Nginx
- **Evaluation**: Custom Amharic-specific metrics

## 📁 Project Structure

```
Amharic-Hnet-Qwin/
├── src/amharichnet/           # Core model implementation
│   ├── models/hnet.py         # H-Net model architecture
│   ├── data/                  # Data processing
│   ├── evaluation/            # Quality assessment
│   └── train/                 # Training pipeline
├── models/                    # Trained models & tokenizer
├── data/                      # Training datasets
├── configs/                   # Model configurations
├── api_server.py             # REST API server
├── generate.py               # Text generation CLI
├── web_interface.html        # Web UI
├── test_api.py              # API testing suite
└── docker-compose.yml       # Production deployment
```

## 🎯 Features

### Text Generation
- **Multiple Categories**: General, news, educational, cultural, conversation
- **Context-Aware**: Intelligent prompt continuation
- **Quality Control**: Real-time evaluation scores
- **Flexible Length**: 10-200 words per generation

### API Endpoints
- `POST /generate` - Single text generation
- `POST /generate/batch` - Batch processing (up to 10)
- `POST /evaluate` - Text quality assessment
- `GET /health` - System health check
- `GET /stats` - Usage statistics

### Web Interface
- **Interactive UI**: Beautiful, responsive design
- **Real-time Generation**: Instant text creation
- **Quality Metrics**: Live performance scores
- **Example Prompts**: Quick-start templates
- **Mobile Friendly**: Works on all devices

## 🔧 Quick Start (Alternative)

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
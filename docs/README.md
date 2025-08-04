# 🇪🇹 Amharic LLM Development Project

A comprehensive implementation of a Large Language Model for Amharic text generation, following a local-first development approach with free cloud resources.

## 🎯 Project Overview

This project implements an Amharic Language Model using:
- **Local Development**: Python virtual environment with Jupyter notebooks
- **Training**: Kaggle notebooks with GPU acceleration
- **Deployment**: Hugging Face Spaces for model serving
- **Cost**: $0/month until production readiness

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ installed
- Git installed
- Internet connection for package installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Amharic-Hnet-Qwin

# 🔒 SECURE SETUP (Recommended)
python secure_setup.py

# OR Manual setup:
cp .env.example .env
nano .env  # Add your API credentials
```

### 2. Verify Setup

```bash
# Test environment configuration
python amharic-hnet/test_env.py

# Check system status
python amharic-hnet/quick_start.py --status

# Run evaluation (works offline)
python amharic-hnet/quick_start.py --phase eval
```

### 3. API Credentials (For Full Features)

**Kaggle API** (for data download):
1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Generate new API token
3. Add to your `.env` file

**HuggingFace Token** (for model deployment):
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create new token
3. Add to your `.env` file

## 📁 Project Structure

```
Amharic-Hnet-Qwin/
├── 📄 IMPLEMENTATION_PLAN.md      # Detailed 14-week roadmap
├── 📄 amharic-llm-review.md       # Original project analysis
├── 📄 README.md                   # This file
├── 📄 setup_environment.py        # Automated setup script
├── 📄 quick_start.sh             # Quick start commands
├── 🐍 amharic_data_collector.py   # Web scraping for Amharic content
├── 📓 kaggle_amharic_trainer.ipynb # Kaggle training notebook
├── 🌐 huggingface_spaces_app.py   # Gradio deployment app
├── 📄 requirements_spaces.txt     # HF Spaces dependencies
├── 📂 data/
│   ├── 📂 raw/                    # Raw scraped data
│   ├── 📂 processed/              # Cleaned and tokenized data
│   └── 📂 collected/              # Final training datasets
├── 📂 models/
│   ├── 📂 checkpoints/            # Training checkpoints
│   └── 📂 final/                  # Production-ready models
├── 📂 notebooks/                  # Development notebooks
├── 📂 scripts/                    # Utility scripts
├── 📂 logs/                       # Training and evaluation logs
└── 📂 venv/                       # Python virtual environment
```

## 🔄 Development Workflow

### Phase 1: Data Collection (Week 1-2)

```bash
# 1. Collect Amharic web content
python amharic_data_collector.py

# 2. Review collected data
ls -la data/collected/
cat data/collected/amharic_texts_*.txt

# 3. Upload to Kaggle dataset
kaggle datasets create -p data/collected/
```

### Phase 2: Model Training (Week 3-6)

```bash
# 1. Upload notebook to Kaggle
# Copy kaggle_amharic_trainer.ipynb to Kaggle

# 2. Run training on Kaggle GPU
# - Enable GPU in Kaggle notebook
# - Run all cells
# - Save model checkpoints

# 3. Download trained model
kaggle datasets download -d your-username/amharic-llm-model
```

### Phase 3: Deployment (Week 7-8)

```bash
# 1. Test locally
python huggingface_spaces_app.py

# 2. Create Hugging Face Space
# - Upload huggingface_spaces_app.py
# - Upload requirements_spaces.txt
# - Upload trained model

# 3. Deploy and test
# Your model will be available at:
# https://huggingface.co/spaces/your-username/amharic-llm
```

## 🛠️ Core Components

### 1. Data Collector (`amharic_data_collector.py`)

**Features:**
- Web scraping for Amharic content
- Text quality filtering (Amharic character ratio > 70%)
- Automatic cleaning and preprocessing
- JSON and plain text output formats
- Statistics tracking

**Usage:**
```python
from amharic_data_collector import AmharicDataCollector

collector = AmharicDataCollector()
collector.collect_from_urls([
    "https://www.bbc.com/amharic",
    "https://www.voa.com/amharic"
])
```

### 2. Kaggle Trainer (`kaggle_amharic_trainer.ipynb`)

**Features:**
- GPT-2 based architecture for Amharic
- Custom Amharic tokenizer
- Mixed precision training
- Gradient accumulation
- Automatic checkpoint saving
- Model evaluation and testing

**Key Classes:**
- `AmharicDataProcessor`: Data loading and preprocessing
- `KaggleAmharicTrainer`: Training pipeline management

### 3. Deployment App (`huggingface_spaces_app.py`)

**Features:**
- Gradio web interface
- Real-time text generation
- Amharic-specific examples
- Model parameter controls
- Responsive design

## 📊 Performance Targets

### Data Collection Goals
- **Week 1**: 10K Amharic sentences
- **Week 2**: 50K Amharic sentences
- **Week 4**: 100K Amharic sentences
- **Week 8**: 500K Amharic sentences

### Model Performance Goals
- **Baseline**: Perplexity < 50
- **Target**: Perplexity < 30
- **Production**: Perplexity < 20
- **Inference**: < 2 seconds per generation

### Quality Metrics
- Amharic character accuracy > 95%
- Grammar coherence score > 80%
- Semantic relevance score > 75%

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Set custom paths
export AMHARIC_DATA_DIR="/path/to/data"
export AMHARIC_MODEL_DIR="/path/to/models"
export KAGGLE_CONFIG_DIR="~/.kaggle"
```

### Data Collector Settings

```python
# In amharic_data_collector.py
class AmharicDataCollector:
    def __init__(self):
        self.min_amharic_ratio = 0.7    # 70% Amharic characters
        self.min_text_length = 50       # Minimum text length
        self.max_text_length = 5000     # Maximum text length
        self.request_delay = 1.0        # Delay between requests
```

### Training Parameters

```python
# In kaggle_amharic_trainer.ipynb
training_args = {
    "learning_rate": 5e-5,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "warmup_steps": 500,
    "save_steps": 1000
}
```

## 🧪 Testing

### Unit Tests

```bash
# Test data collector
python -c "from amharic_data_collector import AmharicDataCollector; \
collector = AmharicDataCollector(); \
print('✅ Data collector working!' if collector.is_amharic_text('ሰላም ነው') else '❌ Failed')"

# Test model loading
python -c "from transformers import GPT2LMHeadModel; \
model = GPT2LMHeadModel.from_pretrained('gpt2'); \
print('✅ Model loading working!')"
```

### Integration Tests

```bash
# Test full pipeline
python amharic_data_collector.py --test
jupyter nbconvert --execute kaggle_amharic_trainer.ipynb
python huggingface_spaces_app.py --test
```

## 📈 Monitoring

### Training Metrics
- Loss curves (training/validation)
- Perplexity scores
- Learning rate schedules
- GPU utilization
- Memory usage

### Data Quality Metrics
- Amharic character ratio distribution
- Text length distribution
- Vocabulary coverage
- Duplicate detection

### Deployment Metrics
- Response time
- Generation quality
- User engagement
- Error rates

## 🚨 Troubleshooting

### Common Issues

**1. Virtual Environment Issues**
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

**2. Kaggle API Issues**
```bash
# Check Kaggle configuration
kaggle --version
kaggle datasets list --user your-username
```

**3. Memory Issues During Training**
```python
# Reduce batch size in training config
training_args["per_device_train_batch_size"] = 4
training_args["gradient_accumulation_steps"] = 8
```

**4. Amharic Text Detection Issues**
```python
# Debug text detection
from amharic_data_collector import AmharicDataCollector
collector = AmharicDataCollector()
collector.min_amharic_ratio = 0.5  # Lower threshold
collector.min_text_length = 20     # Shorter minimum
```

### Getting Help

1. Check the [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed guidance
2. Review Kaggle notebook outputs for training issues
3. Test components individually using the test commands above
4. Check Hugging Face Spaces logs for deployment issues

## 🎯 Next Steps

### Immediate Actions (This Week)
1. ✅ Run `python setup_environment.py`
2. ✅ Configure Kaggle API
3. ⏳ Collect initial Amharic data
4. ⏳ Test data collector with sample URLs

### Week 1 Deliverables
1. ⏳ 10K Amharic sentences collected
2. ⏳ Kaggle training notebook uploaded
3. ⏳ Initial model training started
4. ⏳ Local Gradio interface tested

### Week 2-4 Goals
1. Scale data collection to 100K sentences
2. Complete initial model training
3. Deploy to Hugging Face Spaces
4. Implement evaluation pipeline

## 📚 Resources

### Documentation
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Gradio Documentation](https://gradio.app/docs/)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)

### Amharic Language Resources
- [Amharic Wikipedia](https://am.wikipedia.org/)
- [BBC Amharic](https://www.bbc.com/amharic)
- [VOA Amharic](https://amharic.voanews.com/)
- [Ethiopian News Agency](https://www.ena.et/)

### Technical References
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Tokenization Best Practices](https://huggingface.co/docs/tokenizers/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

---

**🇪🇹 Building the future of Amharic AI, one token at a time!**

*For detailed implementation guidance, see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)*

This repository contains two main components:

1. **Amharic Article Collection System**: A comprehensive system for collecting and processing Amharic articles
2. **Amharic H-Net Model**: An improved transformer-based language model specifically designed for the Amharic language

# Part 1: Amharic Article Collection System

## Overview
A comprehensive system for collecting and processing Amharic articles using Playwright MCP (Model Context Protocol) as an alternative to traditional web scraping methods.

## Features

### 🎯 **Article Collection**
- **Target**: 1000 Amharic articles
- **Sources**: Wikipedia Amharic articles and category discovery
- **Method**: HTTP-based collection (no browser automation required)
- **Content Validation**: Automatic Amharic script detection and quality filtering

### 🔧 **Processing Pipeline**
- **Text Cleaning**: Removes non-Amharic characters, normalizes punctuation
- **Quality Validation**: Filters articles by length, content quality, and Amharic percentage
- **Sentence Extraction**: Splits articles into individual sentences
- **Format Support**: JSON, plain text, and sentence-only outputs

### 📊 **Results Summary**
- **Articles Collected**: 118 raw articles
- **Articles Processed**: 81 valid articles (68.6% success rate)
- **Total Characters**: 35,880 characters
- **Total Words**: 7,178 words
- **Total Sentences**: 373 sentences

## File Structure

```
Amharic-Hnet-Qwin/
├── collected_articles/          # Raw collected articles (118 files)
│   ├── article_0001.json
│   ├── ...
│   └── collection_summary.json
├── processed_articles/          # Cleaned and validated articles (81 files)
│   ├── amharic_corpus.json     # Complete corpus with metadata
│   ├── amharic_corpus.txt      # Plain text corpus for training
│   ├── amharic_sentences.txt   # Individual sentences
│   └── processed_article_*.json
├── simple_article_collector.js # Main collection script
├── article_processor.js        # Processing and cleaning script
├── mcp-server.js               # Playwright MCP server
└── package.json                # Project configuration
```

## Scripts

### Collection Scripts
- `npm run collect` - Run full collection (1000 articles target)
- `npm run collect:test` - Test collection (10 articles)
- `npm run process` - Process collected articles
- `npm run pipeline` - Run complete collection + processing pipeline

### MCP Server
- `npm run mcp:serve` - Start Playwright MCP server on port 3334

## Content Sources

### Primary Sources
- **Wikipedia Amharic**: Main encyclopedia articles
- **Category Discovery**: Automatic discovery of related articles
- **Content Expansion**: Section-based article variations

### Article Topics
- Geography: ኢትዮጵያ, አዲስ አበባ, ላሊበላ, ሐረር, ባሕር-ዳር
- Culture: አማርኛ, ኦሮሞ, ወላይታ, ሲዳማ
- Religion: የኢትዮጵያ ኦርቶዶክስ ተዋሕዶ ቤተ ክርስቲያን
- Science: ሳይንስ, አቡጊዳ
- History: ኩሽ, ኢትዮጵስት በዓለም ዙሪያ

## Data Quality

### Validation Criteria
- ✅ Minimum 200 characters content length
- ✅ At least 10% Amharic script characters
- ✅ Meaningful title (10+ characters)
- ✅ Content diversity (20+ unique characters)

### Processing Features
- **Text Normalization**: Standardized Amharic punctuation
- **Content Cleaning**: Removal of non-relevant characters
- **Sentence Segmentation**: Proper Ethiopian punctuation handling
- **Statistical Analysis**: Character counts, word counts, quality metrics

## Technical Architecture

### Collection Method
- **HTTP-based**: Uses native Node.js HTTP/HTTPS modules
- **No Browser Dependencies**: Avoids Chromium installation issues
- **Wikipedia API**: Leverages MediaWiki REST API
- **Rate Limited**: Respectful 1-2 second delays between requests

### Processing Pipeline
1. **Raw Collection** → JSON files with metadata
2. **Validation** → Quality filtering and Amharic detection
3. **Cleaning** → Text normalization and punctuation
4. **Segmentation** → Sentence-level extraction
5. **Export** → Multiple format outputs

# Part 2: Amharic H-Net Model

Amharic H-Net is an improved transformer-based language model specifically designed for the Amharic language. This project enhances the original H-Net model with advanced attention mechanisms, improved tokenization, and optimized training techniques to better capture the unique linguistic features of Amharic.

## Features

- **Improved Architecture**: Enhanced transformer architecture with specialized attention mechanisms for Amharic language patterns
- **Hybrid Tokenization**: Combined character-level and subword tokenization optimized for Amharic morphology
- **Advanced Training**: Techniques like mixed precision, gradient accumulation, and learning rate scheduling
- **Enhanced Text Generation**: Template-based generation and improved nucleus sampling with n-gram repetition detection
- **Comprehensive Preprocessing**: Specialized text cleaning and normalization for Amharic
- **Data Augmentation**: Techniques to generate additional training samples
- **Model Optimization**: Dynamic and static quantization, quantization-aware training, TorchScript export, and inference optimization for efficient deployment
- **Evaluation Framework**: Linguistic quality metrics and perplexity evaluation
- **Visualization Tools**: Training progress and model comparison visualization
- **Deployment Options**: REST API and web interface for text generation

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/Amharic-Hnet-Qwin.git
cd Amharic-Hnet-Qwin

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Convert an Original H-Net Model

```python
from amharic_hnet.convert_model import ModelConverter

# Initialize the converter
converter = ModelConverter(original_model_path="models/original/hnet_model")

# Convert the model
improved_model = converter.convert()

# Save the improved model
converter.save_model(improved_model, "models/improved/hnet_improved")
```

### Preprocess Amharic Text

```python
from amharic_preprocessor import AmharicTextPreprocessor

# Initialize the preprocessor
preprocessor = AmharicTextPreprocessor(
    remove_non_amharic=True,
    normalize_spaces=True,
    normalize_punctuation=True,
    remove_urls=True,
    remove_emails=True,
    remove_numbers=False,
    min_length=5
)

# Preprocess a single text
text = "ይህ የአማርኛ ጽሑፍ ነው።"
processed_text = preprocessor.preprocess(text)

# Preprocess a file
preprocessor.preprocess_file(
    input_file="data/raw/amharic_text.txt",
    output_file="data/processed/amharic_text_processed.txt"
)
```

### Generate Amharic Text

```python
from amharic_hnet.model import HNetTransformer
from amharic_hnet.hybrid_tokenizer import HybridAmharicTokenizer
from improved_generation import TextGenerator

# Load the model and tokenizer
model = HNetTransformer.from_pretrained("models/improved/hnet_improved")
model.eval()
tokenizer = HybridAmharicTokenizer.from_pretrained("models/improved/hnet_improved")

# Initialize the generator
generator = TextGenerator(model=model, tokenizer=tokenizer)

# Generate text with template-based generation and enhanced repetition penalty
prompt = "ኢትዮጵያ"
generated_text = generator.generate(
    prompt=prompt,
    max_length=100,
    temperature=0.7,
    top_p=0.95,  # Nucleus sampling
    repetition_penalty=1.2,
    repetition_window=50,
    use_enhanced_penalty=True  # Use enhanced n-gram repetition penalty
)

print(generated_text)
```

### Train the Model

```python
from amharic_hnet.model import HNetTransformer
from amharic_hnet.hybrid_tokenizer import HybridAmharicTokenizer
from improved_training import ImprovedTrainer

# Load the model and tokenizer
model = HNetTransformer.from_pretrained("models/improved/hnet_improved")
tokenizer = HybridAmharicTokenizer.from_pretrained("models/improved/hnet_improved")

# Initialize the trainer
trainer = ImprovedTrainer(
    model=model,
    tokenizer=tokenizer,
    train_file="data/processed/train.txt",
    val_file="data/processed/val.txt",
    output_dir="models/improved/hnet_improved_trained",
    batch_size=16,
    learning_rate=5e-5,
    num_epochs=3,
    use_mixed_precision=True,
    gradient_accumulation_steps=4,
    warmup_steps=500
)

# Train the model
trainer.train()
```

### Evaluate the Model

```python
from evaluate_model import ModelEvaluator

# Initialize the evaluator
evaluator = ModelEvaluator(
    model_path="models/improved/hnet_improved",
    test_file="data/processed/test.txt",
    prompts_file="amharic-hnet/test_prompts.txt"
)

# Evaluate the model
results = evaluator.evaluate()
print(results)
```

### Deploy the Model

```bash
# Deploy as a web service
python amharic-hnet/deploy_model.py \
    --model_path models/improved/hnet_improved \
    --model_type pytorch \
    --device cpu \
    --host 0.0.0.0 \
    --port 5000 \
    --create_templates
```

## Project Structure

For a detailed overview of the project structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Running Tests

```bash
cd amharic-hnet
./run_tests.sh
```

For more options, see the [test suite documentation](amharic-hnet/test_suite_README.md).

## Model Optimization

The project includes comprehensive model optimization capabilities:

```bash
# Apply dynamic quantization
python amharic-hnet/model_optimization.py --model models/improved/hnet_improved/model.pt \
    --tokenizer models/improved/hnet_improved/tokenizer.json --dynamic-quant

# Apply static quantization with custom calibration size
python amharic-hnet/model_optimization.py --static-quant --calibration-size 200

# Apply quantization-aware training
python amharic-hnet/model_optimization.py --qat --train-data data/processed/train.txt \
    --qat-epochs 5 --qat-lr 1e-5

# Apply all optimizations and benchmark
python amharic-hnet/model_optimization.py --all --benchmark --benchmark-generation
```

## Benchmarking

```bash
python amharic-hnet/benchmark.py \
    --model_path models/improved/hnet_improved \
    --batch_sizes 1 2 4 8 \
    --sequence_lengths 32 64 128 256 \
    --output_file results/benchmark_results.json \
    --plot_results
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The original H-Net model developers
- The Amharic language community
- Contributors to the PyTorch and Transformers libraries
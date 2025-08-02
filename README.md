# Amharic H-Net: Improved Transformer Model for Amharic Language

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

## Project Overview

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
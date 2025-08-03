# Improved Amharic H-Net Model

This repository contains an improved implementation of the H-Net model for Amharic language generation. The enhancements focus on model architecture, training methodology, text generation quality, and practical implementation aspects.

## Overview

The improved Amharic H-Net model builds upon the original implementation with the following key enhancements:

1. **Model Architecture**
   - Integration of Transformer decoder blocks
   - Enhanced multi-head attention mechanisms
   - Hybrid architecture combining LSTM and Transformer components
   - Improved positional encoding

2. **Training Methodology**
   - Advanced optimization techniques (AdamW, cosine annealing)
   - Gradient accumulation for effective larger batch sizes
   - Mixed precision training for faster computation
   - Data augmentation techniques specific to Amharic

3. **Text Generation**
   - Nucleus sampling for more diverse outputs
   - Enhanced repetition penalty for coherent text
   - Template-based generation for structured content
   - Improved handling of Amharic-specific linguistic features

4. **Practical Implementation**
   - Hybrid tokenizer with subword and character-level capabilities
   - Model quantization for reduced size and faster inference
   - Comprehensive evaluation metrics for Amharic text quality
   - Optimized deployment options

## Quick Setup

### 1. Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your credentials
nano .env
```

### 2. Required Credentials

Add the following to your `.env` file:

```bash
# Hugging Face Token (get from: https://huggingface.co/settings/tokens)
HUGGINGFACE_TOKEN=your_token_here

# Kaggle Credentials (get from: https://www.kaggle.com/account)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### 3. Test Configuration

```bash
# Test environment setup
python test_env.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/amharic-hnet.git
cd amharic-hnet

# Install dependencies
pip install -r requirements.txt
```

## Environment Configuration

The `env_config.py` module automatically searches for `.env` files in:

1. Current directory (`./amharic-hnet/.env`)
2. Parent directory (`./.env`)
3. Home directory (`~/.env`)

### Supported Environment Variables

#### Credentials
- `HUGGINGFACE_TOKEN` - Hugging Face API token
- `KAGGLE_USERNAME` - Kaggle username
- `KAGGLE_KEY` - Kaggle API key

#### Training Parameters
- `BATCH_SIZE` - Training batch size (default: 16)
- `LEARNING_RATE` - Learning rate (default: 5e-5)
- `NUM_EPOCHS` - Number of training epochs (default: 10)
- `DEVICE` - Training device (default: auto-detect)

#### Paths
- `MODEL_DIR` - Model storage directory (default: `./models`)
- `DATA_DIR` - Data storage directory (default: `./data`)
- `RESULTS_DIR` - Results output directory (default: `../results`)
- `LOG_DIR` - Log files directory (default: `./logs`)

## Usage

### Training a New Model

```bash
python improved_training.py \
    --data_dir /path/to/data \
    --output_dir /path/to/save/model \
    --train_tokenizer \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 5e-5 \
    --use_mixed_precision \
    --gradient_accumulation_steps 4
```

### Generating Text

```bash
python improved_generation.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --prompt "ኢትዮጵያ" \
    --max_length 100 \
    --temperature 0.7 \
    --top_p 0.95 \
    --repetition_penalty 1.5 \
    --use_template
```

### Evaluating Model Performance

```bash
python evaluate_model.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --test_data_dir /path/to/test/data \
    --test_prompts_path /path/to/test/prompts.txt \
    --output_dir ./evaluation_results
```

### Comparing with Original Model

```bash
python compare_models.py \
    --original_model_path /path/to/original/model \
    --improved_model_path /path/to/improved/model \
    --original_tokenizer_path /path/to/original/tokenizer \
    --improved_tokenizer_path /path/to/improved/tokenizer \
    --test_prompts_path /path/to/test/prompts.txt \
    --output_dir ./comparison_results
```

### Optimizing Model for Deployment

```bash
python model_optimization.py \
    --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \
    --output_dir ./optimized_models \
    --quantization_type dynamic \
    --export_format torchscript
```

## Components

### Hybrid Tokenizer

The `HybridAmharicTokenizer` combines subword tokenization (BPE) with character-level fallback for handling Amharic's morphological complexity. This approach provides better coverage of the language while maintaining efficiency.

```python
from hybrid_tokenizer import HybridAmharicTokenizer

# Train a new tokenizer
tokenizer = HybridAmharicTokenizer()
tokenizer.train_bpe_tokenizer(files=["file1.txt", "file2.txt"], vocab_size=10000)
tokenizer.save("./tokenizer")

# Load an existing tokenizer
tokenizer = HybridAmharicTokenizer.from_pretrained("./tokenizer")

# Tokenize text
tokens = tokenizer.tokenize("ኢትዮጵያ ውብ ሀገር ናት።")
token_ids = tokenizer.encode("ኢትዮጵያ ውብ ሀገር ናት።")
```

### Improved Model Architecture

The `HNetTransformer` combines the strengths of the original H-Net architecture with Transformer components for enhanced performance.

```python
from improved_model import HNetTransformer

# Create a new model
model = HNetTransformer(
    vocab_size=10000,
    d_model=512,
    n_layers=12,
    n_heads=8,
    d_ff=2048,
    dropout=0.1
)

# Load a pretrained model
model = HNetTransformer.from_pretrained("./model")
```

### Advanced Text Generation

The `ImprovedAmharicGenerator` provides enhanced text generation capabilities with nucleus sampling and template-based generation.

```python
from improved_generation import ImprovedAmharicGenerator

generator = ImprovedAmharicGenerator(
    model=model,
    tokenizer=tokenizer
)

# Generate text
generated_text = generator.generate(
    prompt="ኢትዮጵያ",
    max_length=100,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.5,
    use_template=True
)
```

### Linguistic Quality Evaluation

The `AmharicLinguisticEvaluator` assesses the quality of generated Amharic text based on grammar, coherence, repetition, and cultural relevance.

```python
from linguistic_quality_metrics import AmharicLinguisticEvaluator

evaluator = AmharicLinguisticEvaluator()

# Evaluate text
scores = evaluator.evaluate_text("ኢትዮጵያ ውብ ሀገር ናት። የተለያዩ ባህሎች፣ ቋንቋዎች እና ሃይማኖቶች አሏት።")
print(scores)
```

## Model Optimization

The `ModelOptimizer` provides tools for optimizing models for deployment, including quantization and export to different formats.

```python
from model_optimization import ModelOptimizer

optimizer = ModelOptimizer(model, tokenizer)

# Apply dynamic quantization
quantized_model = optimizer.apply_dynamic_quantization()

# Export to TorchScript
torchscript_model = optimizer.export_to_torchscript()
```

## Experiment Runner

The `ExperimentRunner` facilitates running experiments with different model configurations and comparing results.

```python
from run_experiments import ExperimentRunner, DEFAULT_EXPERIMENTS

runner = ExperimentRunner(
    data_dir="/path/to/data",
    output_dir="/path/to/output"
)

# Run experiments
results = runner.run_experiments(DEFAULT_EXPERIMENTS)

# Generate report
runner.generate_report(results, "experiment_report.md")
```

## Directory Structure

```
amharic-hnet/
├── improved_model.py         # Enhanced model architecture
├── hybrid_tokenizer.py       # Hybrid tokenization approach
├── improved_training.py      # Advanced training methodology
├── improved_generation.py    # Enhanced text generation
├── linguistic_quality_metrics.py  # Evaluation metrics
├── model_optimization.py     # Model optimization tools
├── evaluate_model.py         # Model evaluation script
├── compare_models.py         # Model comparison script
├── run_experiments.py        # Experiment runner
├── amharic_templates.txt     # Templates for generation
└── requirements.txt          # Dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Original H-Net model implementation
- Amharic language resources and datasets
- PyTorch and Hugging Face Transformers library
# Amharic H-Net Project Structure

This document provides an overview of the project structure for the Improved Amharic H-Net model. It explains the purpose of each file and directory in the project.

## Directory Structure

```
Amharic-Hnet-Qwin/
├── .github/
│   └── workflows/
│       └── test.yml              # GitHub Actions workflow for CI/CD
├── amharic-hnet/
│   ├── amharic_hnet/
│   │   ├── __init__.py           # Package initialization
│   │   ├── model.py              # HNetTransformer model implementation
│   │   ├── hybrid_tokenizer.py    # HybridAmharicTokenizer implementation
│   │   └── generator.py          # ImprovedAmharicGenerator implementation
│   ├── amharic_preprocessor.py   # Amharic text preprocessing utilities
│   ├── benchmark.py              # Performance benchmarking script
│   ├── compare_models.py         # Model comparison utilities
│   ├── convert_model.py          # Model conversion utilities
│   ├── data_augmentation.py      # Data augmentation techniques
│   ├── deploy_model.py           # Model deployment utilities
│   ├── evaluate_model.py         # Model evaluation utilities
│   ├── improved_training.py      # Improved training implementation
│   ├── optimize_model.py         # Model optimization utilities
│   ├── preprocess_data.py        # Data preprocessing utilities
│   ├── run_pipeline.sh           # Pipeline execution script
│   ├── run_tests.sh              # Test execution script
│   ├── test_prompts.txt          # Test prompts for evaluation
│   ├── test_suite.py             # Comprehensive test suite
│   ├── test_suite_README.md      # Documentation for test suite
│   └── visualize_results.py      # Result visualization utilities
├── data/
│   ├── raw/                      # Raw data files
│   ├── processed/                # Processed data files
│   └── augmented/                # Augmented data files
├── models/
│   ├── original/                 # Original H-Net model files
│   ├── improved/                 # Improved H-Net model files
│   └── optimized/                # Optimized model files
├── results/
│   ├── evaluation/               # Evaluation results
│   ├── comparison/               # Comparison results
│   └── visualization/            # Visualization results
├── LICENSE                       # Project license
├── PROJECT_STRUCTURE.md          # This file
├── README.md                     # Project documentation
└── requirements.txt              # Project dependencies
```

## Core Components

### Model Architecture

- **amharic_hnet/model.py**: Defines the `HNetTransformer` model architecture, which is an improved version of the original H-Net model. It includes enhancements such as attention mechanisms, positional embeddings, and layer normalization.

- **amharic_hnet/hybrid_tokenizer.py**: Implements the `HybridAmharicTokenizer`, which combines character-level and subword tokenization for Amharic text. It handles Amharic-specific characters and morphology.

- **amharic_hnet/generator.py**: Provides the `ImprovedAmharicGenerator` class for generating Amharic text using the improved model. It includes various decoding strategies and post-processing techniques.

### Data Processing

- **amharic_preprocessor.py**: Contains the `AmharicTextPreprocessor` class for cleaning and normalizing Amharic text data. It handles tasks such as removing non-Amharic characters, normalizing spaces and punctuation, and filtering by length.

- **data_augmentation.py**: Implements the `AmharicDataAugmenter` class for generating additional Amharic text samples. It includes techniques such as character swapping, word dropout, word swapping, and synonym replacement.

- **preprocess_data.py**: Provides utilities for preprocessing Amharic text data for training language models. It includes functions for cleaning, normalizing, and preparing data.

### Training and Evaluation

- **improved_training.py**: Implements the `ImprovedTrainer` class for training the Amharic H-Net model. It includes enhancements such as mixed precision training, gradient accumulation, and learning rate scheduling.

- **evaluate_model.py**: Contains the `ModelEvaluator` class for evaluating the model's performance using linguistic quality metrics and perplexity. It generates text using test prompts and compares it with reference texts.

- **compare_models.py**: Implements the `ModelComparator` class for comparing the original H-Net model with the improved version. It evaluates linguistic quality, model size, inference speed, and optimization.

### Optimization and Deployment

- **optimize_model.py**: Provides the `ModelOptimizer` class for optimizing the model for deployment. It includes techniques such as ONNX conversion, quantization, pruning, and benchmarking.

- **deploy_model.py**: Implements utilities for deploying the model as a web service. It includes a REST API and a simple web interface for generating Amharic text.

- **convert_model.py**: Contains the `ModelConverter` class for converting original H-Net models to the improved format. It handles loading original models, extracting configurations, and transferring weights.

### Testing and Benchmarking

- **test_suite.py**: Provides a comprehensive test suite for all components of the Amharic H-Net model. It includes tests for preprocessing, data augmentation, model architecture, training, generation, evaluation, optimization, and visualization.

- **benchmark.py**: Implements the `ModelBenchmark` class for evaluating the model's performance in terms of inference speed, memory usage, and throughput under various conditions.

### Visualization and Analysis

- **visualize_results.py**: Contains the `ResultVisualizer` class for visualizing the model's performance and training progress. It includes functions for plotting training loss, learning rate, evaluation metrics, model comparisons, and perplexity.

### Scripts and Utilities

- **run_pipeline.sh**: A shell script for automating the entire Amharic H-Net pipeline, including training, optimization, evaluation, and generation.

- **run_tests.sh**: A shell script for running the test suite with various options, such as verbose mode, fail-fast mode, and specific test classes or methods.

## Continuous Integration

- **.github/workflows/test.yml**: A GitHub Actions workflow for continuous integration and deployment. It runs tests, linting, and code formatting checks on push and pull requests.

## Documentation

- **README.md**: The main project documentation, which includes an overview of the project, installation instructions, usage examples, and references.

- **test_suite_README.md**: Documentation for the test suite, including information on how to run tests and what they cover.

- **PROJECT_STRUCTURE.md**: This file, which provides an overview of the project structure.

## Dependencies

- **requirements.txt**: Lists the project dependencies, including PyTorch, Transformers, SentencePiece, NumPy, Pandas, Matplotlib, and other libraries.

## Data and Models

- **data/**: Contains raw, processed, and augmented data files for training and evaluation.

- **models/**: Stores original, improved, and optimized model files.

- **results/**: Contains evaluation results, comparison results, and visualization outputs.

## License

- **LICENSE**: The project license, which specifies the terms under which the code can be used, modified, and distributed.
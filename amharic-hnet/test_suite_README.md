# Amharic H-Net Test Suite

This document provides information about the comprehensive test suite for the Amharic H-Net model. The test suite is designed to verify the functionality of all components of the model, including preprocessing, data augmentation, model architecture, training, generation, evaluation, optimization, and visualization.

## Overview

The test suite is organized into several test classes, each focusing on a specific component of the Amharic H-Net model:

- `TestAmharicPreprocessor`: Tests the Amharic text preprocessor functionality
- `TestDataAugmentation`: Tests the data augmentation techniques
- `TestAmharicDataset`: Tests the dataset loading and processing
- `TestHNetTransformer`: Tests the model architecture and basic operations
- `TestImprovedTrainer`: Tests the training functionality
- `TestAmharicGenerator`: Tests the text generation capabilities
- `TestModelEvaluator`: Tests the model evaluation metrics
- `TestModelComparator`: Tests the model comparison functionality
- `TestModelOptimizer`: Tests the model optimization techniques
- `TestResultVisualizer`: Tests the visualization of results

## Running the Tests

### Prerequisites

Before running the tests, make sure you have installed all the required dependencies as specified in the `requirements.txt` file.

### Running All Tests

To run all tests in the test suite, use the following command:

```bash
python test_suite.py
```

### Running Specific Test Classes

To run tests for a specific component, use the following command:

```bash
python -m unittest test_suite.TestAmharicPreprocessor
```

Replace `TestAmharicPreprocessor` with the name of the test class you want to run.

### Running Specific Test Methods

To run a specific test method, use the following command:

```bash
python -m unittest test_suite.TestAmharicPreprocessor.test_remove_non_amharic
```

Replace `TestAmharicPreprocessor.test_remove_non_amharic` with the name of the test class and method you want to run.

## Test Coverage

The test suite covers the following aspects of the Amharic H-Net model:

### Text Preprocessing

- Removing non-Amharic characters
- Normalizing spaces and punctuation
- Removing URLs, emails, and numbers
- Filtering text by length

### Data Augmentation

- Character swapping
- Word dropout
- Word swapping
- Synonym replacement

### Dataset

- Dataset loading and processing
- Sliding window implementation
- Data augmentation integration

### Model Architecture

- Model initialization
- Forward pass
- Text generation
- Model saving and loading

### Training

- Training loop
- Evaluation during training
- Checkpoint saving

### Text Generation

- Single text generation
- Batch text generation
- Generation parameters (temperature, top-k, top-p, etc.)

### Evaluation

- Perplexity calculation
- Linguistic quality metrics (grammar, coherence, repetition, cultural relevance)
- Overall score calculation

### Model Comparison

- Comparing linguistic quality metrics
- Comparing model size
- Comparing inference time

### Optimization

- Optimizing for inference
- Converting to ONNX format

### Visualization

- Plotting training loss
- Plotting learning rate
- Plotting evaluation metrics
- Plotting model comparison
- Plotting inference time and model size
- Plotting perplexity

## Adding New Tests

To add new tests to the test suite, follow these steps:

1. Identify the component you want to test
2. Add a new test method to the appropriate test class or create a new test class if needed
3. Implement the test logic following the unittest framework guidelines
4. Run the tests to verify your implementation

## Troubleshooting

If you encounter issues when running the tests, consider the following:

- Make sure all dependencies are installed
- Check if the required files and directories exist
- Verify that the model and tokenizer are properly initialized
- Check if the test data is properly formatted

## Contributing

Contributions to the test suite are welcome. If you find a bug or want to add a new test, please submit a pull request or open an issue.
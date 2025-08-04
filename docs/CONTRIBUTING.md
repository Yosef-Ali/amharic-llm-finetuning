# Contributing to Amharic H-Net

Thank you for your interest in contributing to the Amharic H-Net project! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Guidelines](#development-guidelines)
  - [Code Style](#code-style)
  - [Testing](#testing)
  - [Documentation](#documentation)
- [Project Structure](#project-structure)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
   ```bash
   git clone https://github.com/your-username/Amharic-Hnet-Qwin.git
   cd Amharic-Hnet-Qwin
   ```
3. Install the development dependencies
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. Create a branch for your feature or bugfix
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue tracker to see if the problem has already been reported. If it has and the issue is still open, add a comment to the existing issue instead of opening a new one.

When creating a bug report, include as many details as possible:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- System information (OS, Python version, etc.)
- Any relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear and descriptive title
- A detailed description of the proposed enhancement
- An explanation of why this enhancement would be useful
- Any relevant examples or references

### Pull Requests

1. Update your fork to the latest upstream changes
   ```bash
   git fetch upstream
   git merge upstream/main
   ```
2. Make your changes in your feature branch
3. Run the tests to ensure your changes don't break existing functionality
   ```bash
   cd amharic-hnet
   ./run_tests.sh
   ```
4. Add or update tests for your changes
5. Ensure your code follows the project's code style
   ```bash
   black .
   isort .
   flake8
   ```
6. Commit your changes with a descriptive commit message
   ```bash
   git commit -m "Add feature: your feature description"
   ```
7. Push your branch to GitHub
   ```bash
   git push origin feature/your-feature-name
   ```
8. Create a pull request from your branch to the main repository

## Development Guidelines

### Code Style

This project follows the [Black](https://black.readthedocs.io/en/stable/) code style. We also use [isort](https://pycqa.github.io/isort/) for import sorting and [flake8](https://flake8.pycqa.org/en/latest/) for linting.

To ensure your code follows the project's style, run:

```bash
black .
isort .
flake8
```

### Testing

All new features and bug fixes should include tests. We use the `unittest` framework for testing.

To run the tests, use the provided script:

```bash
cd amharic-hnet
./run_tests.sh
```

For more options, see the [test suite documentation](amharic-hnet/test_suite_README.md).

### Documentation

All new features should include documentation. We use Markdown for documentation.

When adding or updating documentation:

- Use clear and concise language
- Include examples where appropriate
- Update the README.md if necessary
- Add docstrings to new functions and classes

## Project Structure

For an overview of the project structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Areas for Contribution

Here are some areas where contributions are particularly welcome:

### Model Improvements

- Enhancing the attention mechanisms
- Improving the tokenization for Amharic
- Implementing more efficient training techniques
- Adding support for more Amharic dialects

### Data Processing

- Adding more data augmentation techniques
- Improving the preprocessing for Amharic text
- Creating better evaluation datasets

### Optimization

- Implementing more efficient inference techniques
- Reducing the model size without sacrificing quality
- Improving deployment options

### Documentation and Examples

- Adding more examples of using the model
- Improving the documentation
- Creating tutorials for specific use cases

## Questions?

If you have any questions about contributing, please open an issue or contact the project maintainers.

Thank you for contributing to Amharic H-Net!
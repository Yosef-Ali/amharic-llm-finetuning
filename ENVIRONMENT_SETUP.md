# Environment Variables Setup Guide

This guide explains how to properly configure environment variables for credentials and settings in the Amharic LLM project.

## 🔐 Quick Setup

### 1. Create Your Environment File

```bash
# Copy the example file
cp .env.example .env

# Edit with your credentials
nano .env  # or use your preferred editor
```

### 2. Set Your Credentials

Edit the `.env` file and replace the placeholder values with your actual credentials:

```bash
# Hugging Face Token
HUGGINGFACE_TOKEN=hf_your_actual_token_here
HF_TOKEN=hf_your_actual_token_here

# Kaggle Credentials
KAGGLE_USERNAME=your_actual_username
KAGGLE_KEY=your_actual_api_key
```

## 📍 Where to Save Environment Variables

### Option 1: `.env` File (Recommended)

**Location**: `/Users/mekdesyared/Amharic-Hnet-Qwin/.env`

**Advantages**:
- ✅ Project-specific
- ✅ Easy to manage
- ✅ Automatically loaded by Python scripts
- ✅ Not committed to version control

**Usage**:
```python
# Python scripts automatically load from .env
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')
```

### Option 2: System Environment Variables

**Location**: Shell profile (`.bashrc`, `.zshrc`, etc.)

**Setup**:
```bash
# Add to ~/.zshrc or ~/.bashrc
export HUGGINGFACE_TOKEN="your_token_here"
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Reload shell
source ~/.zshrc
```

**Advantages**:
- ✅ Available system-wide
- ✅ Persistent across sessions

**Disadvantages**:
- ❌ Affects all projects
- ❌ Harder to manage multiple projects

### Option 3: Kaggle-Specific Setup

**Location**: `~/.kaggle/kaggle.json`

**Setup**:
```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Create credentials file
echo '{
  "username": "your_username",
  "key": "your_api_key"
}' > ~/.kaggle/kaggle.json

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json
```

## 🔑 Getting Your Credentials

### Hugging Face Token

1. **Go to**: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **Click**: "Create new token"
3. **Name**: "Amharic-LLM-Development"
4. **Type**: Select "Write" for development
5. **Permissions**: Select:
   - ✅ Read access to contents of all repos under your personal namespace
   - ✅ Write access to contents/settings of all repos under your personal namespace
   - ✅ Make calls to Inference Providers
   - ✅ Make calls to your Inference Endpoints
6. **Copy**: The generated token (starts with `hf_`)

### Kaggle API Credentials

1. **Go to**: [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. **Scroll to**: "API" section
3. **Click**: "Create New API Token"
4. **Download**: `kaggle.json` file
5. **Extract**: Username and key from the file

## 🛠️ Loading Environment Variables in Code

### Python Scripts

```python
#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access credentials
hf_token = os.getenv('HUGGINGFACE_TOKEN')
kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')

# Use in Hugging Face
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/DialoGPT-medium",
    use_auth_token=hf_token
)

# Use in Kaggle API
import kaggle
kaggle.api.authenticate()  # Uses ~/.kaggle/kaggle.json or env vars
```

### Jupyter Notebooks

```python
# Cell 1: Load environment
import os
from dotenv import load_dotenv

# Load from .env file
load_dotenv()

# Or set directly in notebook (not recommended for production)
# os.environ['HUGGINGFACE_TOKEN'] = 'your_token_here'

# Cell 2: Use credentials
hf_token = os.getenv('HUGGINGFACE_TOKEN')
print(f"Token loaded: {'✅' if hf_token else '❌'}")
```

### Shell Scripts

```bash
#!/bin/bash

# Load from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Use environment variables
echo "Using Hugging Face token: ${HUGGINGFACE_TOKEN:0:10}..."
echo "Using Kaggle username: $KAGGLE_USERNAME"
```

## 🔒 Security Best Practices

### ✅ Do's

- ✅ Use `.env` files for local development
- ✅ Add `.env` to `.gitignore` (already done)
- ✅ Use environment variables in production
- ✅ Rotate tokens regularly
- ✅ Use minimal required permissions
- ✅ Set proper file permissions (`chmod 600`)

### ❌ Don'ts

- ❌ Never commit credentials to version control
- ❌ Don't hardcode tokens in source code
- ❌ Don't share tokens in chat/email
- ❌ Don't use production tokens for development
- ❌ Don't store tokens in public locations

## 🧪 Testing Your Setup

### Test Hugging Face Connection

```python
#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

try:
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("❌ HUGGINGFACE_TOKEN not found")
        exit(1)
    
    # Test token
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-medium",
        use_auth_token=token
    )
    print("✅ Hugging Face token is valid")
    
except Exception as e:
    print(f"❌ Hugging Face token error: {e}")
```

### Test Kaggle Connection

```python
#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import kaggle

load_dotenv()

try:
    # Test Kaggle API
    kaggle.api.authenticate()
    datasets = kaggle.api.dataset_list(user=os.getenv('KAGGLE_USERNAME'), page_size=1)
    print("✅ Kaggle API is working")
    
except Exception as e:
    print(f"❌ Kaggle API error: {e}")
```

### Quick Test Script

```bash
# Run this to test your setup
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('🔑 Environment Variables Status:')
print(f'HUGGINGFACE_TOKEN: {"✅" if os.getenv("HUGGINGFACE_TOKEN") else "❌"}')
print(f'KAGGLE_USERNAME: {"✅" if os.getenv("KAGGLE_USERNAME") else "❌"}')
print(f'KAGGLE_KEY: {"✅" if os.getenv("KAGGLE_KEY") else "❌"}')
"
```

## 🚀 Project-Specific Usage

### Gradio Apps

```python
# huggingface_spaces_app.py
from dotenv import load_dotenv
load_dotenv()

class AmharicLLMInterface:
    def __init__(self):
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        # Use token for model loading
```

### Training Scripts

```python
# kaggle_amharic_trainer.ipynb
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure Kaggle
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
```

### Data Collection

```python
# amharic_data_collector.py
from dotenv import load_dotenv
load_dotenv()

class AmharicDataCollector:
    def __init__(self):
        self.kaggle_username = os.getenv('KAGGLE_USERNAME')
        # Use for dataset uploads
```

## 🔧 Troubleshooting

### Common Issues

**1. "Token not found" Error**
```bash
# Check if .env file exists
ls -la .env

# Check file contents (be careful not to expose tokens)
grep -v "your_" .env | head -5
```

**2. "Permission denied" Error**
```bash
# Fix Kaggle permissions
chmod 600 ~/.kaggle/kaggle.json
```

**3. "Invalid token" Error**
```bash
# Regenerate tokens from respective platforms
# Update .env file with new tokens
```

**4. Environment not loading**
```python
# Add this to debug
import os
print("Current working directory:", os.getcwd())
print(".env file exists:", os.path.exists('.env'))
```

## 📚 Additional Resources

- [Hugging Face Token Documentation](https://huggingface.co/docs/hub/security-tokens)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Python-dotenv Documentation](https://python-dotenv.readthedocs.io/)
- [Environment Variables Best Practices](https://12factor.net/config)

---

**🔐 Remember: Keep your credentials secure and never commit them to version control!**
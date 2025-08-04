# 🤔 Why Does the System Ask for API Credentials?

## Quick Answer
The Amharic Enhanced LLM system asks for API credentials because it needs to:
1. **Download training data** from Kaggle (requires Kaggle API)
2. **Deploy models** to HuggingFace (requires HF token)

## What Each Credential Does

### 🔑 Kaggle API (`kaggle.json`)
**Purpose**: Download large Amharic datasets for training
- **Used in**: Phase 2 (Model Training)
- **What it does**: Downloads datasets like "Amharic News Dataset" from Kaggle
- **Required for**: `python quick_start.py --phase train` or `--phase all`

### 🤗 HuggingFace Token (`HF_TOKEN`)
**Purpose**: Upload and share your trained models
- **Used in**: Phase 4 (Production Deployment)
- **What it does**: Publishes your Amharic model to HuggingFace Hub
- **Required for**: `python quick_start.py --phase deploy` or `--phase all`

## 🚀 Simple Alternative: Skip These Phases

If you don't want to configure credentials right now, you can run individual phases:

```bash
# ✅ These work WITHOUT credentials:
python quick_start.py --phase data     # Data collection
python quick_start.py --phase eval     # Evaluation
python quick_start.py --phase monitor  # Monitoring

# ❌ These NEED credentials:
python quick_start.py --phase train    # Needs Kaggle API
python quick_start.py --phase deploy   # Needs HF token
python quick_start.py --phase all      # Needs both
```

## 🔧 Easy Setup Options

### Option 1: Use Environment Variables (Simplest)
```bash
# Set temporary credentials (for current session only)
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
export HF_TOKEN="your_huggingface_token"

# Then run the system
python quick_start.py --phase all
```

### Option 2: Create .env File
```bash
# Create .env file in the project directory
echo "KAGGLE_USERNAME=your_username" > .env
echo "KAGGLE_KEY=your_api_key" >> .env
echo "HF_TOKEN=your_hf_token" >> .env
```

### Option 3: Skip Credentials (Demo Mode)
```bash
# Run without training/deployment
python quick_start.py --phase data
python quick_start.py --phase eval
python quick_start.py --phase monitor
```

## 🎯 What You Can Do Right Now

1. **Demo the system** without credentials:
   ```bash
   python system_showcase.py
   ```

2. **Check what's already working**:
   ```bash
   python quick_start.py --status
   ```

3. **Run evaluation** (no credentials needed):
   ```bash
   python quick_start.py --phase eval
   ```

## 🔍 Why This Happens

The system is designed to be **production-ready**, which means:
- It can download real datasets (needs Kaggle)
- It can deploy to cloud platforms (needs HuggingFace)
- It follows industry best practices for ML pipelines

But you can absolutely use it in **demo mode** without any external services!

---

**TL;DR**: The credentials are for downloading data and deploying models. You can skip them and still use most of the system!
# 🚀 Production Setup Guide - Amharic Enhanced LLM

## Overview
This guide will help you configure the necessary API credentials and run the complete Amharic Enhanced LLM pipeline.

## Prerequisites
- ✅ Python 3.8+ installed
- ✅ All dependencies installed (already done)
- ✅ Amharic Enhanced LLM system files (already present)

## Step 1: 🔑 Configure Kaggle API Credentials

### 1.1 Create Kaggle Account
- Visit [https://www.kaggle.com](https://www.kaggle.com)
- Sign up for a free account if you don't have one

### 1.2 Download API Credentials
1. **Login to Kaggle**
2. **Navigate to Account Settings**:
   - Click on your profile picture (top-right corner)
   - Select "Account" from the dropdown menu
3. **Generate API Token**:
   - Scroll down to the "API" section
   - Click "Create New API Token" button
   - This downloads `kaggle.json` file containing your credentials

### 1.3 Install Credentials
```bash
# Create .kaggle directory in your home folder
mkdir -p ~/.kaggle

# Move the downloaded kaggle.json file
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

# Set secure permissions
chmod 600 ~/.kaggle/kaggle.json
```

### 1.4 Verify Kaggle Setup
```bash
# Test Kaggle API
kaggle --version
kaggle datasets list --max-size 1
```

## Step 2: 🤗 Configure HuggingFace Token

### 2.1 Get HuggingFace Token
1. **Visit [https://huggingface.co](https://huggingface.co)**
2. **Create account** (if needed) and **login**
3. **Navigate to Settings**:
   - Click on your profile picture
   - Select "Settings"
4. **Generate Token**:
   - Go to "Access Tokens" tab
   - Click "New token"
   - Choose "Write" permissions
   - Copy the generated token

### 2.2 Set Environment Variable
```bash
# Add to your shell profile (~/.zshrc, ~/.bashrc, etc.)
echo 'export HF_TOKEN="your_actual_token_here"' >> ~/.zshrc

# Reload shell configuration
source ~/.zshrc

# Or set for current session only
export HF_TOKEN="your_actual_token_here"
```

### 2.3 Verify HuggingFace Setup
```bash
# Test HuggingFace CLI
huggingface-cli whoami
```

## Step 3: 🚀 Run Complete Pipeline

### 3.1 Quick Test
```bash
# Navigate to project directory
cd /Users/mekdesyared/Amharic-Hnet-Qwin/amharic-hnet

# Check system status
python quick_start.py --status
```

### 3.2 Run Individual Phases
```bash
# Phase 1: Enhanced Data Collection
python quick_start.py --phase data

# Phase 2: Model Training (requires Kaggle API)
python quick_start.py --phase train

# Phase 3: Evaluation & Benchmarking
python quick_start.py --phase eval

# Phase 4: Production Deployment (requires HF token)
python quick_start.py --phase deploy

# Phase 5: Monitoring & Analytics
python quick_start.py --phase monitor
```

### 3.3 Run Complete Pipeline
```bash
# Run all phases together
python quick_start.py --phase all
```

## Step 4: 📊 Monitor Performance

### 4.1 Real-time Monitoring
```bash
# Start monitoring dashboard
python quick_start.py --phase monitor
```

### 4.2 View Results
```bash
# Check evaluation results
cat amharic_evaluation_report.json

# View daily reports
ls -la daily_report_*.json

# Check monitoring database
ls -la monitoring.db
```

## Troubleshooting

### Common Issues

1. **Kaggle API Error**:
   ```
   OSError: Could not find kaggle.json
   ```
   **Solution**: Ensure `kaggle.json` is in `~/.kaggle/` with correct permissions

2. **HuggingFace Authentication Error**:
   ```
   Error: Token not found
   ```
   **Solution**: Set `HF_TOKEN` environment variable

3. **Permission Denied**:
   ```
   Permission denied: kaggle.json
   ```
   **Solution**: Run `chmod 600 ~/.kaggle/kaggle.json`

### Verification Commands
```bash
# Check Kaggle credentials
ls -la ~/.kaggle/kaggle.json
kaggle datasets list --max-size 1

# Check HuggingFace token
echo $HF_TOKEN
huggingface-cli whoami

# Check system status
python quick_start.py --status
```

## Next Steps

Once setup is complete:

1. **🎯 Run Complete Pipeline**: `python quick_start.py --phase all`
2. **📊 Monitor Performance**: `python quick_start.py --phase monitor`
3. **🔍 Analyze Results**: Review generated reports and metrics
4. **🚀 Deploy to Production**: Use deployment scripts for your target platform

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Review error messages carefully
4. Ensure API credentials are correctly configured

---

**Ready to proceed? Let's configure your credentials and run the complete pipeline!**
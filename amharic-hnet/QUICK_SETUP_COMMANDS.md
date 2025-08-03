# 🚀 Quick Setup Commands - Final Steps

## ✅ Completed
- **Kaggle API**: Successfully configured at `/Users/mekdesyared/.kaggle/kaggle.json`

## 🔄 Remaining Steps

### 1. Configure HuggingFace Token

**Option A: Get Token and Set Environment Variable**
```bash
# 1. Visit https://huggingface.co/settings/tokens
# 2. Click "New token" → Choose "Write" permissions → Copy token
# 3. Set environment variable (replace YOUR_TOKEN with actual token)
export HF_TOKEN="YOUR_ACTUAL_TOKEN_HERE"

# 4. Add to shell profile for persistence
echo 'export HF_TOKEN="YOUR_ACTUAL_TOKEN_HERE"' >> ~/.zshrc
source ~/.zshrc
```

**Option B: Skip HuggingFace for Now (Run Limited Pipeline)**
```bash
# You can run most phases without HuggingFace token
# Only deployment phase requires it
```

### 2. Test System Status
```bash
# Check current system status
python quick_start.py --status
```

### 3. Run Available Phases

**With Kaggle API Only:**
```bash
# Phase 1: Data Collection (✅ Ready)
python quick_start.py --phase data

# Phase 2: Model Training (✅ Ready - Kaggle configured)
python quick_start.py --phase train

# Phase 3: Evaluation (✅ Ready)
python quick_start.py --phase eval

# Phase 5: Monitoring (✅ Ready)
python quick_start.py --phase monitor
```

**With Both APIs Configured:**
```bash
# Phase 4: Deployment (Requires HF_TOKEN)
python quick_start.py --phase deploy

# Complete Pipeline (All phases)
python quick_start.py --phase all
```

### 4. Verify Configurations

**Check Kaggle API:**
```bash
kaggle --version
kaggle datasets list --max-size 1
```

**Check HuggingFace Token:**
```bash
echo $HF_TOKEN
huggingface-cli whoami
```

## 🎯 Recommended Next Actions

1. **Immediate Test** (works with current setup):
   ```bash
   python quick_start.py --status
   python quick_start.py --phase train
   ```

2. **Complete Setup** (if you want full deployment):
   - Get HuggingFace token from https://huggingface.co/settings/tokens
   - Set `HF_TOKEN` environment variable
   - Run: `python quick_start.py --phase all`

3. **Monitor Results**:
   ```bash
   python quick_start.py --phase monitor
   ```

---

**🎉 Your Kaggle API is ready! You can now run the training phase and most other features.**
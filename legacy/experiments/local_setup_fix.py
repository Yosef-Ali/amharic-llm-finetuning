#!/usr/bin/env python3
"""
Local Setup Fix and Implementation Guide for Amharic H-Net
This script fixes environment issues and sets up everything for local development
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import platform

class AmharicLLMLocalSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / "venv"
        self.env_file = self.project_root / ".env"
        self.env_example = self.project_root / ".env.example"
        
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*60)
        print(f"🚀 {text}")
        print("="*60)
        
    def check_python_version(self):
        """Check if Python version is appropriate"""
        self.print_header("Checking Python Version")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8+ is required")
            print(f"Current version: {sys.version}")
            return False
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    
    def setup_virtual_environment(self):
        """Create and activate virtual environment"""
        self.print_header("Setting Up Virtual Environment")
        
        # Create venv if it doesn't exist
        if not self.venv_path.exists():
            print("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)])
            print("✅ Virtual environment created")
        else:
            print("✅ Virtual environment already exists")
        
        # Get activation command based on OS
        if platform.system() == "Windows":
            activate_cmd = str(self.venv_path / "Scripts" / "activate.bat")
            pip_path = str(self.venv_path / "Scripts" / "pip")
        else:
            activate_cmd = f"source {self.venv_path}/bin/activate"
            pip_path = str(self.venv_path / "bin" / "pip")
        
        print(f"\n📌 To activate virtual environment, run:")
        print(f"   {activate_cmd}")
        
        return pip_path
    
    def setup_environment_file(self):
        """Create .env file with local settings (no API keys needed)"""
        self.print_header("Setting Up Environment File")
        
        local_env_content = """# Amharic LLM Local Development Environment
# This configuration allows local development without external APIs

# =============================================================================
# LOCAL DEVELOPMENT MODE
# =============================================================================
OFFLINE_MODE=true
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# =============================================================================
# LOCAL PATHS
# =============================================================================
AMHARIC_DATA_DIR=./data
AMHARIC_MODEL_DIR=./models
AMHARIC_RESULTS_DIR=./results
MODEL_CACHE_DIR=./models/cache

# =============================================================================
# DUMMY CREDENTIALS FOR OFFLINE DEVELOPMENT
# =============================================================================
# These are dummy values that allow the code to run without actual API access
HUGGINGFACE_TOKEN=offline_dummy_token
HF_TOKEN=offline_dummy_token
KAGGLE_USERNAME=offline_user
KAGGLE_KEY=offline_key

# =============================================================================
# LOCAL MODEL SETTINGS
# =============================================================================
# Use local model instead of downloading
USE_LOCAL_MODEL=true
LOCAL_MODEL_PATH=./models/amharic-gpt2-local

# =============================================================================
# TRAINING SETTINGS FOR CPU
# =============================================================================
CUDA_VISIBLE_DEVICES=-1  # Force CPU usage
USE_MIXED_PRECISION=false
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=5e-5

# =============================================================================
# LOCAL SERVER SETTINGS
# =============================================================================
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

# =============================================================================
# DEVELOPMENT FLAGS
# =============================================================================
SKIP_API_VALIDATION=true
USE_CACHED_DATA=true
ENABLE_MOCK_RESPONSES=true
"""
        
        # Create .env file
        with open(self.env_file, 'w') as f:
            f.write(local_env_content)
        
        print("✅ Created .env file for local development")
        print("📌 No API keys required - using offline mode")
        
    def install_dependencies(self, pip_path):
        """Install required dependencies"""
        self.print_header("Installing Dependencies")
        
        # Create requirements_local.txt for local development
        requirements_content = """# Core dependencies for local development
torch>=1.10.0
transformers>=4.25.0
datasets>=2.0.0
tokenizers>=0.13.0
numpy>=1.21.0
pandas>=1.3.0
tqdm>=4.62.0
gradio>=3.20.0
flask>=2.0.0
requests>=2.26.0
beautifulsoup4>=4.10.0
lxml>=4.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
wandb>=0.12.0
tensorboard>=2.8.0
pytest>=6.2.0
black>=21.12b0
flake8>=4.0.0

# Amharic-specific
ethiopic-text>=0.1.0
"""
        
        requirements_file = self.project_root / "requirements_local.txt"
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        
        print("Installing packages...")
        subprocess.run([pip_path, "install", "-r", str(requirements_file)])
        print("✅ Dependencies installed")
        
    def create_directory_structure(self):
        """Create necessary directories"""
        self.print_header("Creating Directory Structure")
        
        directories = [
            "data/raw",
            "data/processed",
            "data/collected",
            "models/checkpoints",
            "models/final",
            "models/cache",
            "notebooks",
            "scripts",
            "logs",
            "results",
            "src/models",
            "src/training",
            "src/data",
            "src/evaluation"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {dir_path}")
            
    def create_offline_scripts(self):
        """Create scripts for offline development"""
        self.print_header("Creating Offline Development Scripts")
        
        # Create local data collector
        collector_script = '''#!/usr/bin/env python3
"""
Local Amharic Data Collector - Works Offline
"""

import json
import random
from pathlib import Path
from datetime import datetime

class LocalAmharicDataCollector:
    def __init__(self):
        self.data_dir = Path("data/collected")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample Amharic sentences for offline development
        self.sample_sentences = [
            "ሰላም ወንድሜ እንዴት ነህ?",
            "ኢትዮጵያ ውብ ሀገር ናት።",
            "አማርኛ መማር እፈልጋለሁ።",
            "ዛሬ ጥሩ ቀን ነው።",
            "መጽሐፍ ማንበብ እወዳለሁ።",
            "ቡና መጠጣት ባህላችን ነው።",
            "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
            "ትምህርት የልማት መሰረት ነው።",
            "ሰላም እና ፍቅር ይሁንላችሁ።",
            "የአማርኛ ቋንቋ ታሪክ ረጅም ነው።"
        ]
        
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic Amharic data for development"""
        print(f"Generating {num_samples} synthetic Amharic samples...")
        
        data = []
        for i in range(num_samples):
            # Combine random sentences
            num_sentences = random.randint(3, 8)
            text = " ".join(random.choices(self.sample_sentences, k=num_sentences))
            
            data.append({
                "id": f"synthetic_{i:04d}",
                "text": text,
                "source": "synthetic",
                "timestamp": datetime.now().isoformat(),
                "word_count": len(text.split()),
                "char_count": len(text)
            })
        
        # Save data
        output_file = self.data_dir / f"synthetic_amharic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {num_samples} samples to {output_file}")
        return data
    
    def create_training_data(self):
        """Create formatted training data"""
        print("Creating training data format...")
        
        # Load all collected data
        all_texts = []
        for json_file in self.data_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_texts.extend([item['text'] for item in data])
        
        # Save as plain text for training
        train_file = Path("data/processed/train.txt")
        train_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write('\\n\\n'.join(all_texts))
        
        print(f"✅ Created training file with {len(all_texts)} samples")
        
if __name__ == "__main__":
    collector = LocalAmharicDataCollector()
    collector.generate_synthetic_data(1000)
    collector.create_training_data()
'''
        
        with open("local_data_collector.py", 'w') as f:
            f.write(collector_script)
        
        # Create local trainer script
        trainer_script = '''#!/usr/bin/env python3
"""
Local Amharic Model Trainer - Works Offline
"""

import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    TextDataset, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
from pathlib import Path
import os

class LocalAmharicTrainer:
    def __init__(self):
        self.model_dir = Path("models/amharic-gpt2-local")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Force CPU usage for local development
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.device = torch.device('cpu')
        
    def create_small_model(self):
        """Create a small GPT-2 model for local testing"""
        print("Creating small Amharic GPT-2 model...")
        
        # Small configuration for CPU training
        config = GPT2Config(
            vocab_size=5000,  # Small vocab for testing
            n_positions=256,  # Short sequences
            n_embd=256,       # Small embedding
            n_layer=4,        # Few layers
            n_head=4,         # Few attention heads
        )
        
        model = GPT2LMHeadModel(config)
        model.to(self.device)
        
        # Create a simple tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Save model and tokenizer
        model.save_pretrained(self.model_dir)
        tokenizer.save_pretrained(self.model_dir)
        
        print(f"✅ Model saved to {self.model_dir}")
        return model, tokenizer
    
    def train_local(self):
        """Train model locally on CPU"""
        print("Starting local training...")
        
        # Load or create model
        if (self.model_dir / "config.json").exists():
            print("Loading existing model...")
            model = GPT2LMHeadModel.from_pretrained(self.model_dir)
            tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
        else:
            model, tokenizer = self.create_small_model()
        
        # Training arguments for CPU
        training_args = TrainingArguments(
            output_dir="./results",
            overwrite_output_dir=True,
            num_train_epochs=1,  # Quick training
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=100,
            prediction_loss_only=True,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            no_cuda=True,  # Force CPU
        )
        
        # Load dataset
        train_file = Path("data/processed/train.txt")
        if train_file.exists():
            dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=str(train_file),
                block_size=128
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )
            
            # Train
            print("Training model (this will take a few minutes on CPU)...")
            trainer.train()
            
            # Save final model
            trainer.save_model(self.model_dir)
            print(f"✅ Training complete! Model saved to {self.model_dir}")
        else:
            print("❌ No training data found. Run local_data_collector.py first!")
            
if __name__ == "__main__":
    trainer = LocalAmharicTrainer()
    trainer.train_local()
'''
        
        with open("local_trainer.py", 'w') as f:
            f.write(trainer_script)
        
        # Create local inference script
        inference_script = '''#!/usr/bin/env python3
"""
Local Amharic Model Inference - Works Offline
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr
from pathlib import Path

class LocalAmharicInference:
    def __init__(self):
        self.model_dir = Path("models/amharic-gpt2-local")
        self.device = torch.device('cpu')
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the local model"""
        if self.model is None:
            print("Loading model...")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_dir)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded")
    
    def generate_text(self, prompt, max_length=100, temperature=0.8):
        """Generate Amharic text"""
        self.load_model()
        
        # Encode prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    def create_gradio_interface(self):
        """Create web interface"""
        def generate_wrapper(prompt, max_length, temperature):
            return self.generate_text(prompt, int(max_length), temperature)
        
        interface = gr.Interface(
            fn=generate_wrapper,
            inputs=[
                gr.Textbox(label="የአማርኛ ጽሑፍ ያስገቡ (Enter Amharic text)", 
                          placeholder="ሰላም...", lines=2),
                gr.Slider(50, 200, value=100, step=10, label="Maximum Length"),
                gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature"),
            ],
            outputs=gr.Textbox(label="የመነጨ ጽሑፍ (Generated text)", lines=5),
            title="🇪🇹 Amharic Text Generator (Local)",
            description="Generate Amharic text using locally trained model",
            examples=[
                ["ሰላም ወንድሜ", 100, 0.8],
                ["ኢትዮጵያ", 150, 0.9],
                ["ዛሬ ጥሩ", 100, 0.7],
            ]
        )
        
        return interface
    
    def run_server(self):
        """Run Gradio server"""
        print("Starting Gradio server...")
        interface = self.create_gradio_interface()
        interface.launch(server_name="127.0.0.1", server_port=7860, share=False)

if __name__ == "__main__":
    inference = LocalAmharicInference()
    
    # Check if model exists
    if not (inference.model_dir / "config.json").exists():
        print("❌ No trained model found. Run local_trainer.py first!")
    else:
        inference.run_server()
'''
        
        with open("local_inference.py", 'w') as f:
            f.write(inference_script)
        
        print("✅ Created local_data_collector.py")
        print("✅ Created local_trainer.py")
        print("✅ Created local_inference.py")
        
        # Make scripts executable on Unix-like systems
        if platform.system() != "Windows":
            for script in ["local_data_collector.py", "local_trainer.py", "local_inference.py"]:
                os.chmod(script, 0o755)
    
    def create_step_by_step_guide(self):
        """Create implementation guide"""
        self.print_header("Creating Step-by-Step Implementation Guide")
        
        guide_content = """# 🚀 Amharic LLM Local Implementation Guide

## ✅ Setup Complete! Here's Your Step-by-Step Implementation:

### 📋 Phase 1: Data Collection (Today)

1. **Generate Training Data**:
   ```bash
   python local_data_collector.py
   ```
   This creates synthetic Amharic data for development.

2. **Verify Data**:
   ```bash
   ls -la data/collected/
   cat data/processed/train.txt | head -20
   ```

### 🤖 Phase 2: Model Training (Day 2-3)

1. **Train Small Model Locally**:
   ```bash
   python local_trainer.py
   ```
   This trains a small GPT-2 model on your CPU (takes ~30 mins).

2. **Monitor Training**:
   ```bash
   tail -f logs/training.log
   ```

### 🎯 Phase 3: Testing & Inference (Day 4)

1. **Run Local Server**:
   ```bash
   python local_inference.py
   ```
   Opens browser at http://localhost:7860

2. **Test Generation**:
   - Try prompts: "ሰላም", "ኢትዮጵያ", "የአማርኛ ቋንቋ"
   - Adjust temperature for creativity
   - Test different lengths

### 📈 Phase 4: Improvement Cycle (Week 2)

1. **Collect Real Data**:
   - Modify `local_data_collector.py` to scrape websites
   - Add Wikipedia Amharic articles
   - Include news sources

2. **Scale Model**:
   - Increase model size in `local_trainer.py`
   - Add more layers and parameters
   - Implement attention improvements

3. **Add Features**:
   - Conversational support
   - Instruction following
   - Multi-turn dialogue

### 🔧 Advanced Implementation (Week 3-4)

1. **Implement LoRA**:
   ```python
   # Add to local_trainer.py
   from peft import LoraConfig, get_peft_model
   
   lora_config = LoraConfig(
       r=16,
       lora_alpha=32,
       target_modules=["c_attn", "c_proj"],
       lora_dropout=0.1,
   )
   ```

2. **Add Curriculum Learning**:
   - Sort data by complexity
   - Train on simple → complex
   - Implement adaptive scheduling

3. **Enhance Generation**:
   - Add beam search
   - Implement nucleus sampling
   - Add repetition penalty

### 📊 Monitoring & Evaluation

1. **Track Metrics**:
   ```python
   # Add to training
   wandb.init(project="amharic-llm-local")
   wandb.log({"loss": loss, "perplexity": perplexity})
   ```

2. **Evaluate Quality**:
   - Amharic character accuracy
   - Grammar coherence
   - Cultural appropriateness

### 🚀 Scaling to Production

1. **When Ready for Cloud**:
   - Upload to Kaggle for GPU training
   - Deploy to Hugging Face Spaces
   - Create API endpoints

2. **Performance Optimization**:
   - Implement quantization
   - Add caching
   - Optimize inference

## 💡 Tips for Success

1. **Start Small**: The local scripts work on CPU - perfect for development
2. **Iterate Quickly**: Test changes frequently with small models
3. **Focus on Data**: Quality data > model size
4. **Document Progress**: Keep notes on what works

## 🆘 Troubleshooting

- **Out of Memory**: Reduce batch_size in trainer
- **Slow Training**: Normal on CPU - be patient
- **Poor Generation**: Need more/better training data

## 📚 Next Learning Steps

1. Study the transformer architecture
2. Learn about Amharic morphology
3. Understand attention mechanisms
4. Practice prompt engineering

---

**You're all set! Start with Step 1 and work through systematically. 
The local setup allows you to learn and experiment without cloud costs.**
"""
        
        with open("IMPLEMENTATION_GUIDE.md", 'w') as f:
            f.write(guide_content)
        
        print("✅ Created IMPLEMENTATION_GUIDE.md")
    
    def run_complete_setup(self):
        """Run the complete setup process"""
        print("\n🇪🇹 Amharic H-Net Local Setup & Fix")
        print("="*60)
        
        # Check Python version
        if not self.check_python_version():
            return
        
        # Setup virtual environment
        pip_path = self.setup_virtual_environment()
        
        # Setup environment file
        self.setup_environment_file()
        
        # Create directory structure
        self.create_directory_structure()
        
        # Install dependencies (optional - can be skipped if slow)
        response = input("\nInstall dependencies now? (y/n): ")
        if response.lower() == 'y':
            self.install_dependencies(pip_path)
        
        # Create offline scripts
        self.create_offline_scripts()
        
        # Create implementation guide
        self.create_step_by_step_guide()
        
        # Final instructions
        self.print_header("Setup Complete! 🎉")
        print("\nNext Steps:")
        print("1. Activate virtual environment:")
        if platform.system() == "Windows":
            print(f"   {self.venv_path}\\Scripts\\activate.bat")
        else:
            print(f"   source {self.venv_path}/bin/activate")
        print("\n2. Generate training data:")
        print("   python local_data_collector.py")
        print("\n3. Train your model:")
        print("   python local_trainer.py")
        print("\n4. Run inference server:")
        print("   python local_inference.py")
        print("\n📖 See IMPLEMENTATION_GUIDE.md for detailed instructions!")
        
if __name__ == "__main__":
    setup = AmharicLLMLocalSetup()
    setup.run_complete_setup()

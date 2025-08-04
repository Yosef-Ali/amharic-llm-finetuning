#!/usr/bin/env python3
"""
Smart Amharic Training Script
Follows troubleshooting guidelines with configurable batch size and gradient accumulation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartAmharicTrainer:
    """Smart trainer with troubleshooting features"""
    
    def __init__(self, config=None):
        # Default configuration following troubleshooting guidelines
        self.config = config or {
            "batch_size": 2,  # Reduced from 4 for memory issues
            "gradient_accumulation_steps": 8,  # Compensate with more accumulation
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "max_length": 256,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "output_dir": "./results",
            "model_name": "gpt2",
            "data_dir": "./data"
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Adjust batch size for CPU
        if self.device.type == "cpu":
            self.config["batch_size"] = min(self.config["batch_size"], 2)
            logger.info("Adjusted batch size for CPU training")
        
        self.model = None
        self.tokenizer = None
        self.losses = []
        
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        logger.info("Setting up model and tokenizer...")
        
        try:
            # Try to load local model first
            model_path = Path("models/amharic-gpt2-local")
            if model_path.exists():
                logger.info("Loading local trained model...")
                self.model = GPT2LMHeadModel.from_pretrained(model_path)
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            else:
                logger.info("Creating new model...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.config["model_name"])
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Create smaller model for local training
                config = GPT2Config(
                    vocab_size=self.tokenizer.vocab_size,
                    n_positions=self.config["max_length"],
                    n_embd=256,  # Smaller embedding
                    n_layer=4,   # Fewer layers
                    n_head=4,    # Fewer heads
                )
                self.model = GPT2LMHeadModel(config)
                
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise
            
        self.model.to(self.device)
        logger.info("✅ Model and tokenizer ready")
        
    def load_training_data(self):
        """Load and prepare training data"""
        logger.info("Loading training data...")
        
        data_files = []
        data_dir = Path(self.config["data_dir"])
        
        # Look for data files
        for pattern in ["*.json", "*.txt"]:
            data_files.extend(list(data_dir.glob(f"**/{pattern}")))
            
        if not data_files:
            logger.warning("No data found! Generating synthetic data...")
            return self.generate_synthetic_data()
            
        texts = []
        for file_path in data_files[:100]:  # Limit for testing
            try:
                if file_path.suffix == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'content' in data:
                            texts.append(data['content'])
                        elif isinstance(data, list):
                            texts.extend([item.get('content', str(item)) for item in data])
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
                
        if not texts:
            logger.warning("No valid text found! Generating synthetic data...")
            return self.generate_synthetic_data()
            
        logger.info(f"Loaded {len(texts)} texts")
        return self.create_dataset(texts)
        
    def generate_synthetic_data(self):
        """Generate synthetic Amharic data for testing"""
        logger.info("Generating synthetic Amharic data...")
        
        synthetic_texts = [
            "ሰላም! እንዴት ነዎት? ዛሬ ቆንጆ ቀን ነው።",
            "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ሀገር ናት። ታሪካዊ እና ባህላዊ ሀብት የበዛባት ሀገር ናት።",
            "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት። በዚህ ከተማ ውስጥ ብዙ ታሪካዊ ቦታዎች አሉ።",
            "ቡና የኢትዮጵያ ባህላዊ መጠጥ ነው። በቤተሰብ እና ጓደኞች መካከል የመተሳሰብ ምልክት ነው።",
            "ኢንጀራ የኢትዮጵያ ባህላዊ ምግብ ነው። ከተፍ በተባለ እህል የሚሰራ ነው።",
            "ትምህርት በሰው ልጅ ህይወት ውስጥ በጣም አስፈላጊ ነው። ዕውቀት ሀብት ነው።",
            "ስፖርት ለጤንነት እና ለመዝናኛ በጣም ጠቃሚ ነው። እግር ኳስ በኢትዮጵያ ተወዳጅ ስፖርት ነው።",
            "ቤተሰብ በሰው ህይወት ውስጥ መሰረታዊ ነው። እርስ በርስ መወዳደር እና መደጋገፍ አስፈላጊ ነው።"
        ] * 50  # Repeat to get more data
        
        return self.create_dataset(synthetic_texts)
        
    def create_dataset(self, texts):
        """Create dataset from texts"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config["max_length"],
                return_tensors="pt"
            )
            
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
        
    def is_quality_amharic(self, text):
        """Check Amharic character ratio - following troubleshooting guidelines"""
        if not text or len(text) < 10:
            return False
            
        amharic_chars = len([c for c in text if '\u1200' <= c <= '\u137F'])
        ratio = amharic_chars / len(text) if len(text) > 0 else 0
        return ratio > 0.7
        
    def train(self):
        """Main training function with monitoring"""
        logger.info("🚀 Starting smart training...")
        
        # Setup
        self.setup_model_and_tokenizer()
        train_dataset = self.load_training_data()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config["output_dir"],
            overwrite_output_dir=True,
            num_train_epochs=self.config["num_epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            learning_rate=self.config["learning_rate"],
            warmup_steps=self.config["warmup_steps"],
            logging_steps=self.config["logging_steps"],
            save_steps=self.config["save_steps"],
            eval_steps=self.config["eval_steps"],
            save_strategy="steps",
            evaluation_strategy="no",  # Disable eval for simplicity
            load_best_model_at_end=False,
            report_to=None,  # Disable wandb for local training
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),  # Only use fp16 on GPU
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        model_save_path = Path("models/amharic-gpt2-local")
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        
        logger.info(f"✅ Model saved to {model_save_path}")
        
        # Plot training curves
        self.plot_training_curves()
        
        return trainer
        
    def plot_training_curves(self):
        """Plot loss curves - following troubleshooting guidelines"""
        try:
            # Read training logs
            log_file = Path(self.config["output_dir"]) / "trainer_state.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    trainer_state = json.load(f)
                    
                if 'log_history' in trainer_state:
                    steps = []
                    losses = []
                    
                    for entry in trainer_state['log_history']:
                        if 'loss' in entry:
                            steps.append(entry.get('step', 0))
                            losses.append(entry['loss'])
                    
                    if losses:
                        plt.figure(figsize=(10, 6))
                        plt.plot(steps, losses, 'b-', linewidth=2)
                        plt.title('Training Loss Over Time')
                        plt.xlabel('Steps')
                        plt.ylabel('Loss')
                        plt.grid(True, alpha=0.3)
                        plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
                        plt.close()
                        logger.info("📊 Training loss plot saved as training_loss.png")
                        
        except Exception as e:
            logger.warning(f"Could not plot training curves: {e}")
            
    def test_conversation(self):
        """Test conversational features - following troubleshooting guidelines"""
        logger.info("🧪 Testing conversational features...")
        
        conversation_tests = [
            ("ሰላም", "greeting"),
            ("ስምህ ማን ነው?", "identity"),
            ("ስለ ኢትዮጵያ ንገረኝ", "knowledge")
        ]
        
        for prompt, test_type in conversation_tests:
            try:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                inputs = inputs.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Test ({test_type}): '{prompt}' -> '{response}'")
                
            except Exception as e:
                logger.error(f"Test failed for '{prompt}': {e}")

def main():
    """Main function"""
    logger.info("🇪🇹 Smart Amharic Training Started")
    
    # Configuration
    config = {
        "batch_size": 2,  # Following troubleshooting guidelines
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "max_length": 256,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "output_dir": "./results",
        "model_name": "gpt2",
        "data_dir": "./data"
    }
    
    # Create trainer
    trainer = SmartAmharicTrainer(config)
    
    # Train
    trainer.train()
    
    # Test
    trainer.test_conversation()
    
    logger.info("✅ Training completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Run: python smart_amharic_app.py")
    logger.info("2. Open: http://localhost:7860")
    logger.info("3. Test conversations!")

if __name__ == "__main__":
    main()
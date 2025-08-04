#!/usr/bin/env python3
"""
Fix the training script to use the actual dataset
"""
import os
from pathlib import Path
import sys

def create_proper_training_script():
    """Create a training script that uses the actual data"""
    
    script_content = '''#!/usr/bin/env python3
"""
Proper training script that uses your collected data
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
import os
from pathlib import Path
import json
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.cpu_hnet import CPUAmharicHNet, count_parameters

class AmharicTextDataset(Dataset):
    """Dataset that loads from your actual data files"""
    def __init__(self, data_dir, max_length=128, max_samples=None):
        self.max_length = max_length
        self.samples = []
        
        # Find all text files in data directories
        data_paths = [
            Path("data/collected"),
            Path("data/processed"), 
            Path("data/augmented"),
            Path("data/training"),
            Path("data/raw")
        ]
        
        print("🔍 Searching for data files...")
        text_files = []
        for path in data_paths:
            if path.exists():
                text_files.extend(path.glob("**/*.txt"))
                text_files.extend(path.glob("**/*.json"))
                text_files.extend(path.glob("**/*.jsonl"))
        
        print(f"📂 Found {len(text_files)} potential data files")
        
        # Load samples
        for file_path in tqdm(text_files, desc="Loading data"):
            try:
                if file_path.suffix == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            if line and self._is_amharic(line):
                                self.samples.append(line)
                                if max_samples and len(self.samples) >= max_samples:
                                    break
                
                elif file_path.suffix in ['.json', '.jsonl']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file_path.suffix == '.jsonl':
                            for line in f:
                                try:
                                    data = json.loads(line)
                                    text = data.get('text', '') or data.get('content', '')
                                    if text and self._is_amharic(text):
                                        self.samples.append(text)
                                except:
                                    pass
                        else:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    text = item.get('text', '') or item.get('content', '')
                                    if text and self._is_amharic(text):
                                        self.samples.append(text)
                
                if max_samples and len(self.samples) >= max_samples:
                    break
                    
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")
        
        print(f"✅ Loaded {len(self.samples)} Amharic text samples")
        
        if len(self.samples) == 0:
            print("❌ No Amharic data found! Using fallback samples...")
            self.samples = [
                "ሰላም አለም", "እንዴት ነህ", "አዲስ አበባ ከተማ",
                "ኢትዮጵያ ሀገር", "አማርኛ ቋንቋ"
            ] * 100
    
    def _is_amharic(self, text):
        """Check if text contains Amharic characters"""
        return any('\\u1200' <= char <= '\\u137F' for char in text)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Convert to bytes
        text_bytes = list(text.encode('utf-8'))
        
        # Truncate or pad
        if len(text_bytes) > self.max_length:
            text_bytes = text_bytes[:self.max_length]
        else:
            text_bytes.extend([0] * (self.max_length - len(text_bytes)))
        
        return torch.tensor(text_bytes, dtype=torch.long)

def train_with_real_data(max_epochs=50, batch_size=16, learning_rate=1e-3):
    """Train using the actual collected data"""
    
    print("🚀 Starting Amharic H-Net Training with Real Data")
    
    # Load dataset
    dataset = AmharicTextDataset("data", max_length=128, max_samples=30000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Create model
    model = CPUAmharicHNet()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"🧠 Model size: {count_parameters(model):,} parameters")
    print(f"📊 Dataset size: {len(dataset)} samples")
    print(f"🎯 Training for {max_epochs} epochs...")
    
    # Training loop
    model.train()
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(max_epochs):
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)
            logits = outputs['logits']
            
            # Prepare for loss
            logits = logits.view(-1, 256)
            targets = batch.view(-1)
            
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / num_batches
        print(f"\\n✅ Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'dataset_size': len(dataset)
            }, 'best_model.pt')
            print(f"💾 New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️  Early stopping at epoch {epoch+1}")
                break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    print("\\n🎉 Training complete!")
    return model

if __name__ == "__main__":
    # First, run the data finder to see what we have
    print("🔍 Analyzing available data...")
    os.system("python scripts/find_all_data.py")
    
    print("\\n" + "="*50 + "\\n")
    
    # Now train with the real data
    model = train_with_real_data(max_epochs=50, batch_size=16)
'''
    
    # Save the new training script
    with open('train_with_real_data.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created 'train_with_real_data.py' that will use your actual dataset!")
    print("\nTo use it:")
    print("1. First run: python scripts/find_all_data.py")
    print("2. Then run: python train_with_real_data.py")

if __name__ == "__main__":
    create_proper_training_script()

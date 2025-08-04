import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.cpu_hnet import CPUAmharicHNet, count_parameters

def train_cpu_amharic_hnet():
    """CPU-only training - VERY fast iterations"""
    
    # Setup for CPU efficiency
    torch.set_num_threads(4)  # Use 4 CPU cores max
    
    print("🚀 Starting CPU-Only Amharic H-Net Training")
    print("📊 Setting up training data...")
    
    # Create TINY dataset for proof of concept
    sample_amharic = [
        "ሰላም አለም",
        "እንዴት ነህ",
        "አዲስ አበባ ከተማ",
        "ኢትዮጵያ ሀገር",
        "አማርኛ ቋንቋ",
        "ጤና ይስጥልኝ",
        "እንደምን አለህ",
        "መልካም ቀን",
        "እንኳን ደህና መጣህ",
        "ደህና ሁን"
    ] * 50  # 500 tiny samples
    
    # Convert to bytes
    byte_data = []
    max_length = 32  # Very short sequences for CPU training
    
    for text in sample_amharic:
        text_bytes = [b for b in text.encode('utf-8')]
        if len(text_bytes) <= max_length:
            # Pad to max_length bytes
            text_bytes.extend([0] * (max_length - len(text_bytes)))
            byte_data.append(text_bytes)
    
    print(f"📊 Created {len(byte_data)} training samples")
    
    # Convert to tensors
    data_tensor = torch.tensor(byte_data, dtype=torch.long)
    dataset = TensorDataset(data_tensor, data_tensor)  # Input = Target for language modeling
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create tiny model
    model = CPUAmharicHNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    print(f"🧠 Model size: {count_parameters(model):,} parameters")
    print("🎯 Training for CPU efficiency...")
    
    # FAST training loop
    model.train()
    for epoch in range(10):  # 10 epochs for better learning
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            logits = outputs['logits']
            
            # Reshape for loss computation
            logits = logits.view(-1, 256)  # (batch*seq, vocab)
            targets = targets.view(-1)     # (batch*seq,)
            
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                compression = outputs.get('compression_ratio', 1.0)
                print(f"Epoch {epoch+1:2d}, Batch {batch_idx:2d} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Compression: {compression:.1f}x")
        
        avg_loss = epoch_loss / num_batches
        print(f"✅ Epoch {epoch+1:2d} | Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab_size': 256,
                'model_config': {
                    'd_model': 64,
                    'max_seq_len': 32
                }
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = 'tiny_amharic_hnet.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': 256,
        'model_config': {
            'd_model': 64,
            'max_seq_len': 32
        },
        'training_samples': len(byte_data),
        'final_loss': avg_loss
    }, final_model_path)
    
    print(f"✅ Training complete! Model saved as '{final_model_path}'")
    
    # Test generation
    test_generation(model)
    
    return model, final_model_path

def test_generation(model):
    """Test the tiny model"""
    print("\n🎭 Testing Model Generation...")
    model.eval()
    
    test_cases = [
        "ሰላም",      # hello
        "እንዴት",     # how
        "አዲስ",      # new
        "ኢትዮጵያ",   # Ethiopia
    ]
    
    for test_text in test_cases:
        test_bytes = [b for b in test_text.encode('utf-8')]
        test_bytes.extend([0] * (32 - len(test_bytes)))  # Pad to 32
        
        input_tensor = torch.tensor([test_bytes], dtype=torch.long)
        
        with torch.no_grad():
            # Test forward pass
            outputs = model(input_tensor)
            logits = outputs['logits']
            
            # Get predictions
            predicted = torch.argmax(logits, dim=-1)
            
            # Try to decode
            predicted_bytes = predicted[0].tolist()
            try:
                # Filter out padding and invalid bytes
                real_bytes = [b for b in predicted_bytes if b != 0 and b < 256]
                decoded = bytes(real_bytes).decode('utf-8', errors='ignore')
                print(f"   Input: '{test_text}' → Output: '{decoded}'")
            except Exception as e:
                print(f"   Input: '{test_text}' → Raw bytes (needs more training)")
            
            # Test generation
            try:
                generated = model.generate(input_tensor, max_length=16, temperature=0.8)
                gen_bytes = generated[0].tolist()
                gen_real_bytes = [b for b in gen_bytes if b != 0 and b < 256]
                gen_decoded = bytes(gen_real_bytes).decode('utf-8', errors='ignore')
                print(f"   Generated: '{gen_decoded}'")
            except Exception as e:
                print(f"   Generation failed: {e}")
    
    print("🎭 Generation test complete!")

def load_trained_model(model_path='tiny_amharic_hnet.pt'):
    """Load a trained model"""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = CPUAmharicHNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    if 'final_loss' in checkpoint:
        print(f"📊 Final training loss: {checkpoint['final_loss']:.4f}")
    if 'training_samples' in checkpoint:
        print(f"📊 Trained on {checkpoint['training_samples']} samples")
    
    return model

if __name__ == "__main__":
    print("🇪🇹 CPU-Only Amharic H-Net Training Script")
    print("=" * 50)
    
    # Check if model already exists
    if os.path.exists('tiny_amharic_hnet.pt'):
        print("📋 Found existing model. Options:")
        print("1. Load existing model")
        print("2. Train new model (overwrites existing)")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            model = load_trained_model()
            if model:
                test_generation(model)
        else:
            model, model_path = train_cpu_amharic_hnet()
    else:
        model, model_path = train_cpu_amharic_hnet()
    
    print("\n🎉 Script completed successfully!")
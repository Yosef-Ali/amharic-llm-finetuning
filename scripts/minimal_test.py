#!/usr/bin/env python3
"""
Quick training sanity check script
"""
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def minimal_training_test():
    """Run a minimal training test to check if everything works"""
    print("=== MINIMAL TRAINING TEST ===\n")
    
    try:
        # Try to import the model
        print("1. Checking imports...")
        from amharichnet.models.hnet import AmharicHNet, HNetConfig
        print("   ✅ Model imports successful")
        
        # Try to create a small model
        print("\n2. Creating minimal model...")
        config = HNetConfig(
            vocab_size=100,  # Very small for testing
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2,
            chunk_size=4,
            max_seq_length=32
        )
        model = AmharicHNet(config)
        print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Check data
        print("\n3. Checking for data...")
        data_dir = Path("data/prepared")
        if data_dir.exists():
            files = list(data_dir.glob("*.txt"))
            if files:
                print(f"   ✅ Found {len(files)} data files")
                # Try to read a bit of data
                with open(files[0], 'r', encoding='utf-8') as f:
                    sample = f.read(100)
                    print(f"   Sample: {sample[:50]}...")
            else:
                print("   ❌ No text files in data/prepared/")
        else:
            print("   ❌ No data/prepared directory")
        
        # Test forward pass
        print("\n4. Testing forward pass...")
        import torch
        dummy_input = torch.randint(0, 100, (1, 32))  # batch_size=1, seq_len=32
        
        start_time = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        end_time = time.time()
        
        print(f"   ✅ Forward pass successful in {end_time - start_time:.4f}s")
        print(f"   Output shape: {output.shape}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    minimal_training_test()

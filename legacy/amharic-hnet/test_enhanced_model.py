#!/usr/bin/env python3
"""
Test script for Enhanced H-Net model with 1000-article corpus
"""

import torch
import pickle
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_train import EnhancedHNet, EnhancedAmharicTokenizer

def test_enhanced_model():
    """Test the enhanced H-Net model"""
    print("🔍 Testing Enhanced H-Net with 1000-Article Corpus")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Load tokenizer
    tokenizer_path = "models/enhanced_tokenizer.pkl"
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        return
    
    tokenizer = EnhancedAmharicTokenizer()
    tokenizer.load(tokenizer_path)
    print(f"📚 Loaded tokenizer with {tokenizer.vocab_size} characters")
    
    # Load model
    model_path = "models/enhanced_hnet/best_model.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    # Initialize model
    model = EnhancedHNet(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=3,
        dropout=0.2
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"🤖 Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"📊 Training epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"📉 Validation loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")
    
    # Test generation
    print("\n🎯 Testing text generation:")
    test_prompts = [
        "ኢትዮጵያ",
        "አዲስ አበባ", 
        "ባህል",
        "ትምህርት",
        "ሳይንስ"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        try:
            generated = model.generate(
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=80,
                temperature=0.8,
                device=device
            )
            print(f"✨ Generated: '{generated}'")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    # Test without prompt
    print(f"\n🎲 Random generation (no prompt):")
    try:
        generated = model.generate(
            tokenizer=tokenizer,
            prompt="",
            max_length=100,
            temperature=0.9,
            device=device
        )
        print(f"✨ Generated: '{generated}'")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
    
    print("\n✅ Model testing completed!")

if __name__ == "__main__":
    test_enhanced_model()
#!/usr/bin/env python3
"""
Quick Test Script for Amharic H-Net Model Performance
Step 2: Test generation quality during training
"""

import torch
import pickle
import os
from enhanced_train import EnhancedHNet, EnhancedAmharicTokenizer

def load_model_and_tokenizer(model_path="models/enhanced_hnet_epoch_1.pt", tokenizer_path="models/enhanced_tokenizer.pkl"):
    """Load trained model and tokenizer"""
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = EnhancedAmharicTokenizer()
    tokenizer.load(tokenizer_path)
    
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedHNet(vocab_size=tokenizer.vocab_size)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print(f"Model file {model_path} not found. Using untrained model.")
    
    model.to(device)
    model.eval()
    return model, tokenizer, device

def test_generation(model, tokenizer, device, prompts):
    """Test text generation with various prompts"""
    print("\n" + "="*60)
    print("AMHARIC TEXT GENERATION TEST")
    print("="*60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nTest {i}: Prompt = '{prompt}'")
        print("-" * 40)
        
        try:
            # Generate with different temperatures
            for temp in [0.5, 0.8, 1.0]:
                generated = model.generate(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_length=100,
                    temperature=temp,
                    device=device
                )
                print(f"Temperature {temp}: {generated}")
        except Exception as e:
            print(f"Error generating text: {e}")
        
        print()

def evaluate_model_metrics(model, tokenizer, device):
    """Evaluate basic model metrics"""
    print("\n" + "="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    
    # Model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Vocabulary info
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Most Common Characters: {list(tokenizer.char_to_idx.keys())[:10]}")
    
    # Test basic encoding/decoding
    test_text = "ሰላም ዓለም"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nEncoding Test:")
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Encoding Success: {test_text == decoded}")

def main():
    """Main testing function"""
    print("Starting Amharic H-Net Quick Test...")
    
    # Test prompts in Amharic
    test_prompts = [
        "ሰላም",  # Hello
        "ኢትዮጵያ",  # Ethiopia
        "አዲስ አበባ",  # Addis Ababa
        "ትምህርት",  # Education
        "ቤተሰብ"   # Family
    ]
    
    try:
        # Load model and tokenizer
        model, tokenizer, device = load_model_and_tokenizer()
        
        # Run evaluations
        evaluate_model_metrics(model, tokenizer, device)
        test_generation(model, tokenizer, device, test_prompts)
        
        print("\n" + "="*60)
        print("QUICK TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
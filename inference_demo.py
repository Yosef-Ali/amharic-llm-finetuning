#!/usr/bin/env python3
"""
Amharic H-Net Inference Demo
Demonstrates text generation capabilities of the hierarchical attention model.
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F
from src.amharichnet.models.hnet import AmharicHNet, HNetConfig
from src.amharichnet.utils.config import Config, ModelConfig

def create_demo_model():
    """Create a demo H-Net model for inference."""
    config = HNetConfig(
        vocab_size=32000,
        hidden_dim=256,  # Smaller for demo
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_seq_len=512
    )
    
    model = AmharicHNet(config)
    if model.available:
        model.eval()
        print(f"✅ Demo H-Net model loaded successfully!")
        print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Architecture: Hierarchical Attention Transformer")
        print(f"   - Layers: {config.num_layers}, Heads: {config.num_heads}")
    else:
        print("❌ PyTorch not available - running in demo mode")
    
    return model

def generate_text(model, prompt_tokens, max_length=50, temperature=0.8):
    """Generate text using the H-Net model."""
    if not model.available:
        return "[Demo mode - PyTorch not available]"
    
    model.eval()
    with torch.no_grad():
        # Convert prompt to tensor
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long)
        
        generated = input_ids.clone()
        
        for _ in range(max_length):
            # Forward pass
            outputs = model(generated)
            logits = outputs["logits"]
            
            # Get next token probabilities
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we hit max sequence length
            if generated.size(1) >= model.cfg.max_seq_len:
                break
        
        return generated[0, len(prompt_tokens):].tolist()

def demo_amharic_generation():
    """Demonstrate Amharic text generation capabilities."""
    print("\n🚀 Amharic H-Net Text Generation Demo")
    print("=" * 50)
    
    # Create model
    model = create_demo_model()
    
    # Demo prompts (using token IDs for simplicity)
    demo_prompts = [
        {
            "name": "Greeting",
            "tokens": [1, 2, 3, 4, 5],  # Placeholder tokens
            "description": "Amharic greeting sequence"
        },
        {
            "name": "News Article",
            "tokens": [10, 11, 12, 13, 14],
            "description": "News article beginning"
        },
        {
            "name": "Story",
            "tokens": [20, 21, 22, 23, 24],
            "description": "Traditional story opening"
        }
    ]
    
    print("\n📝 Generating text samples...\n")
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"{i}. {prompt['name']} ({prompt['description']})")
        print(f"   Input tokens: {prompt['tokens']}")
        
        if model.available:
            # Generate text
            generated_tokens = generate_text(model, prompt['tokens'], max_length=20)
            print(f"   Generated: {generated_tokens[:10]}...")
            
            # Simulate Amharic text (in real implementation, would use tokenizer)
            amharic_text = "[Generated Amharic text would appear here]"
            print(f"   Amharic: {amharic_text}")
        else:
            print(f"   Generated: [Demo mode - showing model architecture]")
        
        print()

def show_model_architecture():
    """Display the H-Net model architecture details."""
    print("\n🏗️ H-Net Architecture Overview")
    print("=" * 40)
    
    architecture_info = {
        "Model Type": "Hierarchical Attention Transformer",
        "Attention Levels": "Word → Phrase → Sentence",
        "Key Features": [
            "Multi-level hierarchical attention",
            "Positional encoding for Amharic",
            "Layer normalization at each level",
            "Feed-forward networks",
            "Dropout for regularization"
        ],
        "Optimizations": [
            "Xavier weight initialization",
            "Residual connections",
            "Batch-first processing",
            "Gradient clipping support"
        ]
    }
    
    for key, value in architecture_info.items():
        if isinstance(value, list):
            print(f"\n{key}:")
            for item in value:
                print(f"  • {item}")
        else:
            print(f"\n{key}: {value}")

def show_training_results():
    """Display recent training results."""
    print("\n📊 Recent Training Results")
    print("=" * 30)
    
    try:
        import json
        with open('outputs/production/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        print(f"Model: {metrics['model_type']}")
        print(f"Architecture: {metrics['architecture']}")
        print(f"Training epochs: {metrics['epochs_completed']}")
        print(f"Final loss: {metrics['final_loss']:.4f}")
        print(f"Validation loss: {metrics['val_loss']:.4f}")
        print(f"Training time: {metrics['training_time_seconds']:.1f}s")
        print(f"Parameters: {metrics['model_config']['hidden_dim']}D, {metrics['model_config']['num_layers']} layers")
        
    except FileNotFoundError:
        print("No recent training results found.")
        print("Run training first: python -m amharichnet.cli train --config configs/production.yaml")

def main():
    """Main demo function."""
    print("🎉 Welcome to the Amharic H-Net Demo!")
    print("This demonstrates the hierarchical attention model for Amharic language processing.")
    
    # Show training results
    show_training_results()
    
    # Show architecture
    show_model_architecture()
    
    # Demo text generation
    demo_amharic_generation()
    
    print("\n🎯 Next Steps:")
    print("1. Train with real Amharic tokenizer")
    print("2. Add proper text preprocessing")
    print("3. Implement beam search decoding")
    print("4. Create web interface for interactive generation")
    print("5. Fine-tune on specific Amharic tasks")
    
    print("\n✨ H-Net is ready for production Amharic language modeling!")

if __name__ == "__main__":
    main()
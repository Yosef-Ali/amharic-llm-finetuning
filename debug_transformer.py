#!/usr/bin/env python3
"""Simple debug script for Transformer H-Net."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from amharichnet.models.transformer_hnet import TransformerHNet, TransformerHNetConfig

def debug_transformer():
    print("🔍 Debugging Transformer H-Net")
    
    # Simple config
    config = TransformerHNetConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=64
    )
    
    print(f"Config: {config.hidden_dim} dim, {config.num_heads} heads")
    print(f"Head dim: {config.hidden_dim // config.num_heads}")
    
    try:
        # Create model
        model = TransformerHNet(config)
        print(f"✅ Model created: {model.get_num_params():,} parameters")
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        print(f"Input shape: {input_ids.shape}")
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        
        print(f"✅ Forward pass successful")
        print(f"Loss: {outputs['loss'].item() if outputs['loss'] is not None else 'None'}")
        print(f"Logits shape: {outputs['logits'].shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_transformer()
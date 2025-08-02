#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Conversion script for Amharic H-Net models.

This script converts original H-Net models to the improved format.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, Optional

import torch
from tqdm import tqdm

# Import original model components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_train import EnhancedHNet, AmharicHNetDataset

# Import improved model components
from improved_model import HNetTransformer
from hybrid_tokenizer import HybridAmharicTokenizer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ModelConverter:
    """Converter for Amharic H-Net models."""
    
    def __init__(self, 
                 original_model_path: str,
                 original_tokenizer_path: Optional[str] = None,
                 output_dir: str = "./converted_model",
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize model converter.
        
        Args:
            original_model_path: Path to the original model
            original_tokenizer_path: Path to the original tokenizer (optional)
            output_dir: Directory to save converted model
            device: Device to use
        """
        self.original_model_path = original_model_path
        self.original_tokenizer_path = original_tokenizer_path
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def convert(self):
        """Convert the model."""
        logger.info(f"Converting model from {self.original_model_path}")
        
        # Load original model
        logger.info("Loading original model...")
        try:
            original_model = EnhancedHNet.load_from_checkpoint(self.original_model_path)
            original_model.to(self.device)
            original_model.eval()
            logger.info("Original model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading original model: {e}")
            return
        
        # Create improved model
        logger.info("Creating improved model...")
        try:
            # Get original model configuration
            original_config = self._extract_original_config(original_model)
            
            # Create improved model with similar configuration
            improved_model = HNetTransformer(
                vocab_size=original_config["vocab_size"],
                d_model=original_config["d_model"],
                n_layers=original_config["n_layers"],
                n_heads=original_config["n_heads"],
                d_ff=original_config["d_ff"],
                dropout=original_config["dropout"],
                pad_idx=original_config["pad_idx"]
            )
            improved_model.to(self.device)
            
            # Transfer weights
            self._transfer_weights(original_model, improved_model)
            
            # Save improved model
            improved_model.save_pretrained(self.output_dir)
            logger.info(f"Improved model saved to {self.output_dir}")
        except Exception as e:
            logger.error(f"Error creating improved model: {e}")
            return
        
        # Convert tokenizer
        if self.original_tokenizer_path:
            logger.info("Converting tokenizer...")
            try:
                self._convert_tokenizer()
                logger.info(f"Tokenizer saved to {self.output_dir}")
            except Exception as e:
                logger.error(f"Error converting tokenizer: {e}")
        
        logger.info("Conversion completed successfully")
    
    def _extract_original_config(self, original_model: EnhancedHNet) -> Dict[str, Any]:
        """Extract configuration from original model.
        
        Args:
            original_model: Original model
            
        Returns:
            Dictionary of configuration parameters
        """
        # Extract configuration
        config = {
            "vocab_size": original_model.vocab_size,
            "d_model": original_model.d_model,
            "n_layers": original_model.n_layers,
            "n_heads": original_model.n_heads,
            "d_ff": original_model.d_model * 4,  # Typical ratio
            "dropout": original_model.dropout,
            "pad_idx": 0  # Default padding index
        }
        
        # Save configuration
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        
        return config
    
    def _transfer_weights(self, original_model: EnhancedHNet, improved_model: HNetTransformer):
        """Transfer weights from original model to improved model.
        
        Args:
            original_model: Original model
            improved_model: Improved model
        """
        logger.info("Transferring weights...")
        
        # Transfer embedding weights
        improved_model.embedding.weight.data.copy_(original_model.embedding.weight.data)
        
        # Transfer LSTM weights (first layer only)
        if hasattr(original_model, 'lstm') and hasattr(improved_model, 'lstm'):
            # Transfer weights for the first LSTM layer
            for i in range(min(len(original_model.lstm.weight_ih_l0), len(improved_model.lstm.weight_ih_l0))):
                improved_model.lstm.weight_ih_l0[i].data.copy_(original_model.lstm.weight_ih_l0[i].data)
                improved_model.lstm.weight_hh_l0[i].data.copy_(original_model.lstm.weight_hh_l0[i].data)
                improved_model.lstm.bias_ih_l0[i].data.copy_(original_model.lstm.bias_ih_l0[i].data)
                improved_model.lstm.bias_hh_l0[i].data.copy_(original_model.lstm.bias_hh_l0[i].data)
        
        # Transfer output projection weights
        if hasattr(original_model, 'output_projection') and hasattr(improved_model, 'output_projection'):
            improved_model.output_projection.weight.data.copy_(original_model.output_projection.weight.data)
            if hasattr(original_model.output_projection, 'bias') and hasattr(improved_model.output_projection, 'bias'):
                improved_model.output_projection.bias.data.copy_(original_model.output_projection.bias.data)
        
        logger.info("Weights transferred successfully")
    
    def _convert_tokenizer(self):
        """Convert tokenizer."""
        # Create hybrid tokenizer
        tokenizer = HybridAmharicTokenizer()
        
        # Load original tokenizer vocabulary
        # Note: This assumes the original tokenizer has a vocabulary file
        vocab_path = os.path.join(self.original_tokenizer_path, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            
            # Initialize tokenizer with vocabulary
            tokenizer._initialize_vocab()
            
            # Add vocabulary items
            for token, idx in vocab.items():
                if token not in tokenizer.token_to_id:
                    tokenizer.token_to_id[token] = idx
                    tokenizer.id_to_token[idx] = token
        else:
            # Initialize with default vocabulary
            tokenizer._initialize_vocab()
        
        # Save tokenizer
        tokenizer.save(self.output_dir)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert Amharic H-Net model to improved format")
    
    # Model arguments
    parser.add_argument("--original_model_path", type=str, required=True, help="Path to the original model")
    parser.add_argument("--original_tokenizer_path", type=str, help="Path to the original tokenizer")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./converted_model", help="Directory to save converted model")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Create converter
    converter = ModelConverter(
        original_model_path=args.original_model_path,
        original_tokenizer_path=args.original_tokenizer_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Convert model
    converter.convert()


if __name__ == "__main__":
    main()
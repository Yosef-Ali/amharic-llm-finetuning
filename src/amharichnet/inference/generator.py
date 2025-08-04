"""Amharic text generation using trained H-Net model."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from ..data.amharic_tokenizer import AmharicSubwordTokenizer
from ..models.hnet import create_model
from ..utils.config import Config


class AmharicTextGenerator:
    """Amharic text generator using trained H-Net model."""
    
    def __init__(self, model_path: str, config_path: str, tokenizer_path: str = None):
        """Initialize the generator.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to training configuration
            tokenizer_path: Path to tokenizer vocabulary (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize tokenizer
        self.tokenizer = AmharicSubwordTokenizer()
        if tokenizer_path:
            self.tokenizer.load_vocab(tokenizer_path)
        else:
            # Try default path
            try:
                self.tokenizer.load_vocab("models/tokenizer/amharic_vocab.json")
            except FileNotFoundError:
                print("⚠️  Using basic tokenizer (vocab not found)")
        
        # Load model
        self.model = self._load_model(model_path)
        
        print(f"✅ AmharicTextGenerator initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Vocab size: {self.tokenizer.vocab_size}")
        print(f"   - Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_config(self, config_path: str) -> Config:
        """Load training configuration."""
        try:
            # This is a simplified config loader
            # You might need to adjust based on your actual Config class
            with open(config_path) as f:
                import yaml
                config_dict = yaml.safe_load(f)
            
            # Create a simple config object
            class SimpleConfig:
                def __init__(self, config_dict):
                    for key, value in config_dict.items():
                        if isinstance(value, dict):
                            setattr(self, key, SimpleConfig(value))
                        else:
                            setattr(self, key, value)
            
            return SimpleConfig(config_dict)
        except Exception as e:
            print(f"⚠️  Config loading failed: {e}")
            # Return default config
            class DefaultConfig:
                class model:
                    vocab_size = 3087
                    hidden_dim = 256
                    num_layers = 4
                    num_heads = 8
                    dropout = 0.15
                    max_seq_len = 128
            return DefaultConfig()
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        try:
            # Create model
            model = create_model(self.config)
            
            if not model.available:
                print("⚠️  Model not available, using dummy model")
                return model
            
            # Load checkpoint if exists
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                if hasattr(model, 'net') and 'model_state' in checkpoint:
                    model.net.load_state_dict(checkpoint['model_state'])
                    print(f"✅ Loaded model from {model_path}")
                else:
                    print(f"⚠️  Checkpoint format not recognized")
            else:
                print(f"⚠️  Checkpoint not found: {model_path}")
            
            model.eval()
            return model.to(self.device)
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            # Return a simple fallback
            class DummyModel:
                def __init__(self):
                    self.available = False
                def eval(self): pass
                def to(self, device): return self
            return DummyModel()
    
    def generate(
        self, 
        prompt: str = "", 
        max_length: int = 100, 
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[str]:
        """Generate Amharic text.
        
        Args:
            prompt: Starting text (can be empty)
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated text strings
        """
        if not self.model.available:
            return [f"ይህ የሙከራ ጽሑፍ ነው። {prompt} ተጨማሪ የአማርኛ ጽሑፍ..."]
        
        results = []
        
        for _ in range(num_return_sequences):
            try:
                # Encode prompt
                if prompt:
                    input_ids = self.tokenizer.encode(prompt, max_len=64)
                else:
                    # Start with BOS token
                    input_ids = [self.tokenizer.special_tokens.get("<bos>", 2)]
                
                input_ids = torch.tensor([input_ids], device=self.device)
                
                generated_ids = input_ids.clone()
                
                # Generate tokens
                with torch.no_grad():
                    for _ in range(max_length):
                        # Forward pass
                        if hasattr(self.model, 'net'):
                            outputs = self.model.net(generated_ids.float())
                        else:
                            # Fallback
                            outputs = generated_ids.float()
                        
                        # Get next token logits
                        if len(outputs.shape) >= 2:
                            next_token_logits = outputs[:, -1, :] if len(outputs.shape) == 3 else outputs[:, -1:]
                        else:
                            next_token_logits = outputs.unsqueeze(0)
                        
                        # Apply temperature
                        next_token_logits = next_token_logits / temperature
                        
                        # Apply top-k filtering
                        if top_k > 0:
                            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                            next_token_logits[indices_to_remove] = float('-inf')
                        
                        # Apply top-p filtering
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                            next_token_logits[indices_to_remove] = float('-inf')
                        
                        # Sample next token
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        # Add to sequence
                        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                        
                        # Check for EOS token
                        if next_token.item() == self.tokenizer.special_tokens.get("<eos>", 3):
                            break
                
                # Decode generated text
                generated_text = self.tokenizer.decode(generated_ids[0].tolist())
                results.append(generated_text)
                
            except Exception as e:
                print(f"⚠️  Generation error: {e}")
                results.append(f"የጽሑፍ ማዘገያ ስህተት። {prompt}")
        
        return results
    
    def complete_text(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Complete a given Amharic text prompt.
        
        Args:
            prompt: Text to complete
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Completed text
        """
        generated = self.generate(
            prompt=prompt,
            max_length=max_new_tokens,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            num_return_sequences=1
        )
        return generated[0] if generated else prompt
    
    def generate_creative(self, theme: str = "ኢትዮጵያ") -> str:
        """Generate creative Amharic text on a theme.
        
        Args:
            theme: Theme or topic for generation
            
        Returns:
            Generated creative text
        """
        prompt = f"በ{theme} ጉዳይ ላይ"
        return self.complete_text(prompt, max_new_tokens=80)


def create_generator(
    model_path: str = "outputs/amharic_optimized_training/checkpoints/ckpt.pt",
    config_path: str = "configs/amharic_optimized.yaml"
) -> AmharicTextGenerator:
    """Create an Amharic text generator.
    
    Args:
        model_path: Path to trained model checkpoint
        config_path: Path to training configuration
        
    Returns:
        Initialized AmharicTextGenerator
    """
    return AmharicTextGenerator(model_path, config_path)
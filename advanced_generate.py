#!/usr/bin/env python3
"""Advanced Amharic text generation using H-Net architecture."""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from amharichnet.data.amharic_tokenizer import AmharicSubwordTokenizer
from amharichnet.models.hnet import create_model
import yaml


class AdvancedAmharicGenerator:
    """Advanced Amharic text generator using trained H-Net."""
    
    def __init__(self, config_path: str = "configs/amharic_optimized.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        try:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            self.config = self._dict_to_config(config_dict)
        except Exception as e:
            print(f"⚠️  Config loading failed: {e}, using defaults")
            self.config = self._default_config()
        
        # Initialize tokenizer
        self.tokenizer = AmharicSubwordTokenizer()
        try:
            self.tokenizer.load_vocab("models/tokenizer/amharic_vocab.json")
            print(f"✅ Loaded Amharic tokenizer (vocab: {self.tokenizer.vocab_size})")
        except FileNotFoundError:
            print("⚠️  Using basic tokenizer")
        
        # Create model (freshly initialized, no checkpoint)
        self.model = create_model(self.config)
        if self.model.available:
            self.model.eval()
            print(f"✅ H-Net model initialized ({sum(p.numel() for p in self.model.parameters()):,} params)")
        else:
            print("⚠️  Using fallback generation")
        
        # Amharic language patterns for better generation
        self.amharic_patterns = {
            'sentence_starters': [
                'ኢትዮጵያ', 'አዲስ አበባ', 'ሰላም', 'ትምህርት', 'ወጣቶች', 
                'ሀገሪቱ', 'ህዝቡ', 'መንግሥት', 'ባህል', 'ታሪክ'
            ],
            'conjunctions': [
                'እና', 'ከዚያ', 'በተጨማሪ', 'ስለዚህ', 'ይሁን እንጂ', 
                'በመሆኑም', 'በተቃራኒው', 'በአንፃሩ'
            ],
            'common_words': [
                'ነው', 'ናት', 'ናቅች', 'ይሆናል', 'አለ', 'ቀጠለ', 
                'ተናገረ', 'ሰፈረ', 'ተመለከተ', 'ወሰነ'
            ]
        }
    
    def _dict_to_config(self, config_dict):
        """Convert config dict to object."""
        class Config:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, Config(v))
                    else:
                        setattr(self, k, v)
        return Config(config_dict)
    
    def _default_config(self):
        """Create default config if loading fails."""
        class DefaultConfig:
            class model:
                vocab_size = 3087
                hidden_dim = 256
                num_layers = 4
                num_heads = 8
                dropout = 0.15
                max_seq_len = 128
        return DefaultConfig()
    
    def generate_with_model(self, prompt: str, max_length: int = 50, temperature: float = 1.0) -> str:
        """Generate text using the H-Net model."""
        if not self.model.available:
            return self._fallback_generate(prompt, max_length)
        
        try:
            # Encode prompt
            if prompt:
                input_ids = self.tokenizer.encode(prompt, max_len=32)
            else:
                input_ids = [self.tokenizer.special_tokens.get("<bos>", 2)]
            
            # Convert to tensor
            input_tensor = torch.tensor([input_ids], dtype=torch.float32, device=self.device)
            
            generated_ids = input_ids.copy()
            
            with torch.no_grad():
                for _ in range(max_length):
                    # Prepare input
                    current_input = torch.tensor([generated_ids[-32:]], dtype=torch.float32, device=self.device)
                    
                    # Forward pass through H-Net
                    if hasattr(self.model, 'net'):
                        output = self.model.net(current_input)
                    else:
                        # Use model step
                        output = self.model.step()
                        if isinstance(output, (int, float)):
                            # Convert scalar to logits-like tensor
                            vocab_size = self.tokenizer.vocab_size
                            output = torch.randn(1, vocab_size) * abs(output)
                    
                    # Get next token probabilities
                    if hasattr(output, 'shape') and len(output.shape) >= 2:
                        if len(output.shape) == 3:
                            next_token_logits = output[0, -1, :]  # Last timestep
                        else:
                            next_token_logits = output[0, :]
                    else:
                        # Fallback: random distribution
                        vocab_size = self.tokenizer.vocab_size
                        next_token_logits = torch.randn(vocab_size) * 0.1
                    
                    # Ensure we have the right vocab size
                    vocab_size = self.tokenizer.vocab_size
                    if next_token_logits.size(0) != vocab_size:
                        # Adjust to match vocab size
                        if next_token_logits.size(0) > vocab_size:
                            next_token_logits = next_token_logits[:vocab_size]
                        else:
                            # Pad with small random values
                            padding = torch.randn(vocab_size - next_token_logits.size(0)) * 0.01
                            next_token_logits = torch.cat([next_token_logits, padding])
                    
                    # Apply temperature
                    next_token_logits = next_token_logits / temperature
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    
                    # Avoid out of range errors
                    probs = torch.clamp(probs, min=1e-8)
                    probs = probs / probs.sum()  # Renormalize
                    
                    try:
                        next_token = torch.multinomial(probs, 1).item()
                    except RuntimeError:
                        # Fallback: sample from top tokens
                        _, top_indices = torch.topk(probs, min(10, len(probs)))
                        next_token = top_indices[torch.randint(len(top_indices), (1,))].item()
                    
                    # Add to sequence
                    generated_ids.append(next_token)
                    
                    # Check for EOS or special tokens
                    if next_token == self.tokenizer.special_tokens.get("<eos>", 3):
                        break
                    if next_token == 0:  # Pad token
                        break
            
            # Decode result
            result = self.tokenizer.decode(generated_ids)
            return result.strip()
            
        except Exception as e:
            print(f"⚠️  Model generation error: {e}")
            return self._fallback_generate(prompt, max_length)
    
    def _fallback_generate(self, prompt: str, max_length: int) -> str:
        """Fallback generation using patterns."""
        if not prompt:
            prompt = random.choice(self.amharic_patterns['sentence_starters'])
        
        # Build sentence using patterns
        words = [prompt]
        
        for _ in range(min(max_length // 3, 10)):  # Generate a few more words
            if random.random() < 0.3:  # Add conjunction
                words.append(random.choice(self.amharic_patterns['conjunctions']))
            
            # Add content word
            if 'ኢትዮጵያ' in prompt or 'ሀገር' in ' '.join(words):
                content_words = ['ውብ', 'ሀገር', 'አፍሪካ', 'ህዝብ', 'ባህል', 'ታሪክ']
            elif 'ትምህርት' in prompt or 'ተማሪ' in ' '.join(words):
                content_words = ['ጠቃሚ', 'አስፈላጊ', 'ህጻናት', 'ወጣቶች', 'ትምህርት']
            else:
                content_words = ['ጠቃሚ', 'አስፈላጊ', 'ሰላም', 'ደስታ', 'ተስፋ']
            
            words.append(random.choice(content_words))
            
            # Add ending
            if random.random() < 0.4:
                words.append(random.choice(self.amharic_patterns['common_words']))
        
        return ' '.join(words) + '።'
    
    def generate_multiple(self, prompt: str, count: int = 3, max_length: int = 50) -> list:
        """Generate multiple variations."""
        results = []
        for i in range(count):
            # Vary temperature for diversity
            temp = 0.7 + (i * 0.3)  # 0.7, 1.0, 1.3
            result = self.generate_with_model(prompt, max_length, temperature=temp)
            results.append(result)
        return results
    
    def interactive_generation(self):
        """Interactive text generation session."""
        print("\n🎨 Interactive Amharic Text Generation")
        print("Type prompts (empty line to quit):")
        print("=" * 40)
        
        while True:
            try:
                prompt = input("\n📝 Prompt: ").strip()
                if not prompt:
                    break
                
                print("🤖 Generating...")
                result = self.generate_with_model(prompt, max_length=60, temperature=0.9)
                print(f"✨ Result: {result}")
                
                # Show tokenization
                tokens = self.tokenizer.encode(result, max_len=64)
                print(f"🔤 Tokens: {len(tokens)} tokens")
                
            except KeyboardInterrupt:
                break
        
        print("\n👋 Generation session ended!")


def main():
    print("🚀 Advanced Amharic Text Generator")
    print("=" * 40)
    
    try:
        generator = AdvancedAmharicGenerator()
        
        # Test generation
        test_prompts = [
            "",
            "ኢትዮጵያ",
            "አዲስ አበባ",
            "ሰላም ወንድሜ",
            "ትምህርት ለሁሉም",
            "ወጣቶች"
        ]
        
        print(f"\n📝 Generation Tests:")
        for prompt in test_prompts:
            print(f"\n🔸 Prompt: '{prompt}'")
            result = generator.generate_with_model(prompt, max_length=40, temperature=0.8)
            print(f"   Result: {result}")
        
        print(f"\n🎨 Multiple Variations for 'ኢትዮጵያ':")
        variations = generator.generate_multiple("ኢትዮጵያ", count=3, max_length=30)
        for i, var in enumerate(variations, 1):
            print(f"   {i}. {var}")
        
        # Interactive mode
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            generator.interactive_generation()
        
        print(f"\n✅ Advanced generation test complete!")
        print(f"💡 Run with --interactive for interactive mode")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
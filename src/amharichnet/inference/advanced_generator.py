"""
Advanced Amharic Text Generator with Beam Search and Dynamic Sampling
Enhanced generation capabilities for the Transformer H-Net model
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import math
import heapq
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amharichnet.models.transformer_hnet import TransformerHNet, TransformerHNetConfig
from amharichnet.data.amharic_tokenizer import AmharicSubwordTokenizer


@dataclass
class GenerationConfig:
    """Configuration for advanced text generation."""
    max_length: int = 100
    min_length: int = 10
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    num_beams: int = 1
    num_return_sequences: int = 1
    early_stopping: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    diversity_penalty: float = 0.0
    temperature_decay: float = 0.95
    adaptive_temperature: bool = True


class BeamSearchNode:
    """Node for beam search algorithm."""
    
    def __init__(self, sequence: torch.Tensor, score: float, past_key_values=None):
        self.sequence = sequence
        self.score = score
        self.past_key_values = past_key_values
        self.length = len(sequence)
    
    def __lt__(self, other):
        return self.score < other.score


class AdvancedAmharicGenerator:
    """Advanced Amharic text generator with multiple generation strategies."""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = AmharicSubwordTokenizer()
        try:
            vocab_path = "models/tokenizer/amharic_vocab.json"
            if Path(vocab_path).exists():
                self.tokenizer.load_vocab(vocab_path)
                print(f"✅ Tokenizer loaded (vocab: {self.tokenizer.vocab_size:,})")
            else:
                print("⚠️  Using basic tokenizer")
        except Exception as e:
            print(f"⚠️  Tokenizer loading failed: {e}")
        
        # Initialize model
        self.model = None
        self.config = TransformerHNetConfig(vocab_size=self.tokenizer.vocab_size)
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            # Create new model
            self.model = TransformerHNet(self.config)
            self.model.to(self.device)
            print(f"✅ New model created ({self.model.get_num_params():,} parameters)")
    
    def load_model(self, model_path: str):
        """Load pre-trained model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = TransformerHNet(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = TransformerHNet(self.config)
            self.model.to(self.device)
    
    def save_model(self, model_path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'tokenizer_vocab_size': self.tokenizer.vocab_size
        }
        torch.save(checkpoint, model_path)
        print(f"✅ Model saved to {model_path}")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(text)
        else:
            # Fallback to basic tokenization
            tokens = [hash(token) % self.config.vocab_size for token in text.split()]
        
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text."""
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(tokens.cpu().tolist())
        else:
            # Fallback - return placeholder
            return " ".join([f"tok_{t}" for t in tokens.cpu().tolist()])
    
    def apply_repetition_penalty(self, 
                               logits: torch.Tensor, 
                               input_ids: torch.Tensor, 
                               penalty: float = 1.1) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        if penalty == 1.0:
            return logits
        
        # Get unique tokens in input
        unique_ids = torch.unique(input_ids)
        
        # Apply penalty
        for token_id in unique_ids:
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
        
        return logits
    
    def adaptive_temperature_schedule(self, 
                                    step: int, 
                                    initial_temp: float, 
                                    decay_rate: float = 0.95) -> float:
        """Calculate adaptive temperature based on generation step."""
        return initial_temp * (decay_rate ** step)
    
    def top_k_top_p_filtering(self, 
                            logits: torch.Tensor,
                            top_k: int = 50,
                            top_p: float = 0.9,
                            temperature: float = 1.0) -> torch.Tensor:
        """Apply top-k and top-p filtering to logits."""
        # Apply temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def beam_search_generate(self, 
                           input_ids: torch.Tensor,
                           config: GenerationConfig) -> List[torch.Tensor]:
        """Generate text using beam search algorithm."""
        if not self.model or not self.model.available:
            return [input_ids]
        
        batch_size = input_ids.shape[0]
        if batch_size != 1:
            # Beam search works with single batch
            input_ids = input_ids[:1]
        
        # Initialize beam
        initial_node = BeamSearchNode(input_ids[0], 0.0)
        beam = [initial_node]
        completed_sequences = []
        
        self.model.eval()
        with torch.no_grad():
            for step in range(config.max_length):
                candidates = []
                
                for node in beam:
                    if len(node.sequence) >= config.max_length:
                        completed_sequences.append(node)
                        continue
                    
                    # Forward pass
                    current_input = node.sequence.unsqueeze(0)
                    outputs = self.model(current_input, past_key_values=node.past_key_values, use_cache=True)
                    
                    logits = outputs["logits"][0, -1, :]  # Get last token logits
                    past_key_values = outputs["past_key_values"]
                    
                    # Apply repetition penalty
                    logits = self.apply_repetition_penalty(
                        logits.unsqueeze(0), current_input, config.repetition_penalty
                    ).squeeze(0)
                    
                    # Apply temperature and filtering
                    filtered_logits = self.top_k_top_p_filtering(
                        logits.unsqueeze(0),
                        config.top_k,
                        config.top_p,
                        config.temperature
                    ).squeeze(0)
                    
                    # Get top-k candidates
                    probs = F.softmax(filtered_logits, dim=-1)
                    top_probs, top_indices = torch.topk(probs, config.num_beams)
                    
                    # Create new candidates
                    for prob, token_id in zip(top_probs, top_indices):
                        new_sequence = torch.cat([node.sequence, token_id.unsqueeze(0)])
                        
                        # Calculate score (log probability + length penalty)
                        log_prob = torch.log(prob + 1e-8)
                        length_penalty = ((len(new_sequence) + 5) / 6) ** config.length_penalty
                        score = (node.score + log_prob) / length_penalty
                        
                        new_node = BeamSearchNode(new_sequence, score.item(), past_key_values)
                        candidates.append(new_node)
                
                # Select top candidates for next beam
                candidates.sort(reverse=True)  # Sort by score (descending)
                beam = candidates[:config.num_beams]
                
                # Early stopping check
                if config.early_stopping and len(completed_sequences) >= config.num_return_sequences:
                    break
        
        # Combine completed sequences with remaining beam
        all_sequences = completed_sequences + beam
        all_sequences.sort(reverse=True)
        
        # Return top sequences
        results = []
        for i in range(min(config.num_return_sequences, len(all_sequences))):
            results.append(all_sequences[i].sequence.unsqueeze(0))
        
        return results
    
    def greedy_generate(self,
                       input_ids: torch.Tensor,  
                       config: GenerationConfig) -> torch.Tensor:
        """Generate text using greedy decoding."""
        if not self.model or not self.model.available:
            return input_ids
        
        self.model.eval()
        generated = input_ids.clone()
        past_key_values = None
        
        with torch.no_grad():
            for step in range(config.max_length):
                # Forward pass
                outputs = self.model(generated, past_key_values=past_key_values, use_cache=True)
                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs["past_key_values"]
                
                # Apply repetition penalty
                logits = self.apply_repetition_penalty(logits, generated, config.repetition_penalty)
                
                # Apply temperature
                if config.adaptive_temperature:
                    current_temp = self.adaptive_temperature_schedule(step, config.temperature, config.temperature_decay)
                else:
                    current_temp = config.temperature
                
                # Apply filtering
                filtered_logits = self.top_k_top_p_filtering(
                    logits, config.top_k, config.top_p, current_temp
                )
                
                # Select next token (greedy)
                next_token = torch.argmax(filtered_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check stopping criteria
                if next_token.item() == config.eos_token_id:
                    break
                
                if generated.shape[1] >= config.max_length:
                    break
        
        return generated
    
    def sampling_generate(self,
                         input_ids: torch.Tensor,
                         config: GenerationConfig) -> torch.Tensor:
        """Generate text using sampling."""
        if not self.model or not self.model.available:
            return input_ids
        
        self.model.eval()
        generated = input_ids.clone()
        past_key_values = None
        
        with torch.no_grad():
            for step in range(config.max_length):
                # Forward pass
                outputs = self.model(generated, past_key_values=past_key_values, use_cache=True)
                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs["past_key_values"]
                
                # Apply repetition penalty
                logits = self.apply_repetition_penalty(logits, generated, config.repetition_penalty)
                
                # Apply temperature
                if config.adaptive_temperature:
                    current_temp = self.adaptive_temperature_schedule(step, config.temperature, config.temperature_decay)
                else:
                    current_temp = config.temperature
                
                # Apply filtering
                filtered_logits = self.top_k_top_p_filtering(
                    logits, config.top_k, config.top_p, current_temp
                )
                
                # Sample next token
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check stopping criteria
                if next_token.item() == config.eos_token_id:
                    break
                
                if generated.shape[1] >= config.max_length:
                    break
        
        return generated
    
    def generate(self,
                prompt: str = "",
                generation_strategy: str = "sampling",
                **kwargs) -> str:
        """Main generation method with multiple strategies."""
        
        # Create generation config
        config = GenerationConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Encode input
        if prompt:
            input_ids = self.encode_text(prompt)
        else:
            # Start with random token or BOS token
            input_ids = torch.tensor([[1]], dtype=torch.long, device=self.device)
        
        # Generate based on strategy
        if generation_strategy == "beam_search" and config.num_beams > 1:
            sequences = self.beam_search_generate(input_ids, config)
            if sequences:
                generated_ids = sequences[0]
            else:
                generated_ids = input_ids
        elif generation_strategy == "greedy":
            generated_ids = self.greedy_generate(input_ids, config)
        else:  # Default to sampling
            generated_ids = self.sampling_generate(input_ids, config)
        
        # Decode output
        if prompt:
            # Return only the generated part (exclude input)
            input_length = input_ids.shape[1]
            output_ids = generated_ids[:, input_length:]
        else:
            output_ids = generated_ids
        
        # Decode to text
        generated_text = self.decode_tokens(output_ids[0])
        
        # Clean up output
        generated_text = generated_text.strip()
        
        return generated_text
    
    def generate_multiple(self,
                         prompt: str = "",
                         num_generations: int = 3,
                         generation_strategy: str = "sampling",
                         **kwargs) -> List[str]:
        """Generate multiple diverse outputs."""
        results = []
        
        # Increase diversity for multiple generations
        base_temp = kwargs.get('temperature', 1.0)
        base_top_p = kwargs.get('top_p', 0.9)
        
        for i in range(num_generations):
            # Add slight variation to parameters for diversity
            temp_variation = base_temp + (i * 0.1)
            top_p_variation = max(0.1, base_top_p - (i * 0.05))
            
            modified_kwargs = kwargs.copy()
            modified_kwargs['temperature'] = temp_variation
            modified_kwargs['top_p'] = top_p_variation
            
            result = self.generate(prompt, generation_strategy, **modified_kwargs)
            results.append(result)
        
        return results


def main():
    """Demo of advanced generation capabilities."""
    print("🚀 Advanced Amharic Text Generator Demo")
    print("=" * 50)
    
    # Initialize generator
    generator = AdvancedAmharicGenerator()
    
    # Test prompts
    test_prompts = [
        "ኢትዮጵያ",
        "አዲስ አበባ",
        "ትምህርት በጣም",
        ""  # Empty prompt
    ]
    
    # Test different generation strategies
    strategies = ["sampling", "greedy", "beam_search"]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        print("-" * 30)
        
        for strategy in strategies:
            print(f"\n🎯 Strategy: {strategy}")
            
            if strategy == "beam_search":
                result = generator.generate(
                    prompt=prompt,
                    generation_strategy=strategy,
                    max_length=50,
                    num_beams=3,
                    temperature=0.8,
                    repetition_penalty=1.2
                )
            else:
                result = generator.generate(
                    prompt=prompt,
                    generation_strategy=strategy,
                    max_length=50,
                    temperature=0.9,
                    top_k=40,
                    top_p=0.85
                )
            
            print(f"Result: {result}")
    
    print(f"\n✅ Demo completed!")


if __name__ == "__main__":
    main()
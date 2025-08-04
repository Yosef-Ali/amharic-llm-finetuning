import torch
import torch.nn.functional as F
import re
import time
import random
import os
import sys
from collections import Counter

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the existing modules
from enhanced_train import EnhancedHNet, EnhancedAmharicTokenizer
from practical_fix import PracticalAmharicGenerator

class ImprovedAmharicGenerator:
    """Enhanced text generation for Amharic with advanced sampling techniques."""
    
    def __init__(self, model_path, tokenizer_path, templates_file=None):
        """Initialize the generator with model, tokenizer, and optional templates.
        
        Args:
            model_path: Path to the pretrained model
            tokenizer_path: Path to the tokenizer
            templates_file: Optional path to templates file
        """
        print(f"Loading model from {model_path}...")
        self.model = EnhancedHNet.from_pretrained(model_path)
        self.model.eval()  # Set to evaluation mode
        
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = EnhancedAmharicTokenizer.from_pretrained(tokenizer_path)
        
        # Initialize cache for generated text
        self.cache = {}
        
        # Load templates if provided
        self.templates = {}
        if templates_file and os.path.exists(templates_file):
            self._load_templates(templates_file)
    
    def _load_templates(self, file_path):
        """Load templates from file.
        
        Args:
            file_path: Path to templates file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                current_category = None
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if line.startswith('[') and line.endswith(']'):
                        current_category = line[1:-1]
                        self.templates[current_category] = []
                    elif current_category:
                        self.templates[current_category].append(line)
            print(f"Loaded {sum(len(v) for v in self.templates.values())} templates from {file_path}")
        except Exception as e:
            print(f"Error loading templates: {e}")
    
    def _get_category(self, prompt):
        """Map prompt to a template category.
        
        Args:
            prompt: The input prompt
            
        Returns:
            The matched category or 'general'
        """
        categories = {
            'ኢትዮጵያ': 'country',
            'አዲስ አበባ': 'city',
            'ባህል': 'culture',
            'ትምህርት': 'education',
            'ፖለቲካ': 'politics',
            'ኢኮኖሚ': 'economy',
            'ስፖርት': 'sports',
            'ጤና': 'health'
        }
        
        for key, category in categories.items():
            if key in prompt:
                return category
        
        # Default to general category
        return 'general'
    
    def generate_with_nucleus_sampling(self, prompt, max_length=100, 
                                      top_p=0.92, temperature=0.8, 
                                      repetition_penalty=1.3, 
                                      repetition_window=20,
                                      use_enhanced_penalty=True,
                                      use_cache=True):
        """Generate text using nucleus sampling with enhanced repetition penalty.
        
        Args:
            prompt: The input prompt
            max_length: Maximum length of generated text
            top_p: Nucleus sampling parameter (0-1)
            temperature: Sampling temperature
            repetition_penalty: Base repetition penalty
            repetition_window: Window size for repetition detection
            use_enhanced_penalty: Whether to use enhanced n-gram based repetition penalty
            use_cache: Whether to use the cache
            
        Returns:
            The generated text
        """
        # Check cache first if enabled
        cache_key = f"{prompt}_{max_length}_{top_p}_{temperature}_{repetition_penalty}_{use_enhanced_penalty}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Convert to device
        device = next(self.model.parameters()).device
        
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids]).to(device)
        
        # Track generated tokens for repetition penalty
        generated_tokens = input_ids.copy()
        
        # Start generation
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_tensor)
                next_token_logits = outputs[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Get the last n tokens for repetition checking
                last_tokens = generated_tokens[-repetition_window:]
                
                # Count token frequencies in the window
                token_counts = Counter(last_tokens)
                
                if use_enhanced_penalty:
                    # Enhanced repetition penalty with n-gram detection
                    # Track n-grams (2-grams and 3-grams) for more sophisticated repetition detection
                    ngram_penalty = {}
                    
                    # Check for 2-grams and 3-grams in the recent tokens
                    if len(last_tokens) >= 2:
                        for i in range(len(last_tokens) - 1):
                            bigram = (last_tokens[i], last_tokens[i+1])
                            # Count occurrences of this bigram
                            bigram_count = 0
                            for j in range(len(last_tokens) - 1):
                                if last_tokens[j] == bigram[0] and last_tokens[j+1] == bigram[1]:
                                    bigram_count += 1
                            
                            # If this bigram repeats, penalize its second token more
                            if bigram_count > 1:
                                ngram_penalty[bigram[1]] = max(ngram_penalty.get(bigram[1], 1.0), 
                                                              1.0 + (0.3 * bigram_count))
                    
                    # Check for 3-grams
                    if len(last_tokens) >= 3:
                        for i in range(len(last_tokens) - 2):
                            trigram = (last_tokens[i], last_tokens[i+1], last_tokens[i+2])
                            # Count occurrences of this trigram
                            trigram_count = 0
                            for j in range(len(last_tokens) - 2):
                                if (last_tokens[j] == trigram[0] and 
                                    last_tokens[j+1] == trigram[1] and 
                                    last_tokens[j+2] == trigram[2]):
                                    trigram_count += 1
                            
                            # If this trigram repeats, penalize its last token more heavily
                            if trigram_count > 1:
                                ngram_penalty[trigram[2]] = max(ngram_penalty.get(trigram[2], 1.0), 
                                                              1.0 + (0.5 * trigram_count))
                    
                    # Calculate text length factor - penalty decreases as text gets longer
                    # This prevents the model from being too constrained in longer generations
                    text_length = len(generated_tokens)
                    length_factor = max(0.8, min(1.0, 1.0 - (text_length / 1000)))
                    
                    # Apply dynamic penalty based on frequency and n-gram detection
                    for token_id, count in token_counts.items():
                        if count > 1:
                            # Base penalty increases with frequency
                            base_penalty = repetition_penalty * (1 + 0.2 * (count - 1))
                            
                            # Apply additional n-gram penalty if detected
                            ngram_factor = ngram_penalty.get(token_id, 1.0)
                            
                            # Combine penalties and apply length scaling
                            dynamic_penalty = base_penalty * ngram_factor * length_factor
                            next_token_logits[:, token_id] /= dynamic_penalty
                else:
                    # Use the original simpler repetition penalty if enhanced is disabled
                    for token_id, count in token_counts.items():
                        if count > 1:
                            # Simple penalty based on frequency
                            dynamic_penalty = repetition_penalty * (1 + 0.2 * (count - 1))
                            next_token_logits[:, token_id] /= dynamic_penalty
                
                # Apply nucleus sampling (top-p)
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # Add to generated tokens
                generated_tokens.append(next_token.item())
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                
                # Check if we've generated an end token
                if next_token.item() == self.tokenizer.eos_token_id or \
                   self.tokenizer.decode([next_token.item()]) == "።":
                    # If we've generated a sentence ending, check if we have enough text
                    if len(generated_tokens) > len(input_ids) + 10:  # At least 10 new tokens
                        break
        
        # Decode the generated tokens
        result = self.tokenizer.decode(generated_tokens)
        
        # Cache the result if enabled
        if use_cache:
            self.cache[cache_key] = result
        
        return result
    
    def generate_with_template(self, prompt, max_length=100, use_enhanced_penalty=True):
        """Generate text using templates when available.
        
        Args:
            prompt: The input prompt
            max_length: Maximum length of generated text
            use_enhanced_penalty: Whether to use enhanced n-gram based repetition penalty
            
        Returns:
            The generated text
        """
        if not self.templates:
            # No templates available, fall back to nucleus sampling
            return self.generate_with_nucleus_sampling(prompt, max_length, use_enhanced_penalty=use_enhanced_penalty)
        
        # Determine the appropriate category
        category = self._get_category(prompt)
        
        # Check if we have templates for this category
        if category in self.templates and self.templates[category]:
            # Select a template randomly
            template = random.choice(self.templates[category])
            
            # Replace placeholder with prompt if present
            if '{prompt}' in template:
                template = template.replace('{prompt}', prompt)
            
            # Generate continuation from template
            continuation_length = max_length - len(template)
            if continuation_length > 0:
                continuation = self.generate_with_nucleus_sampling(
                    template, 
                    max_length=continuation_length,
                    use_cache=False,  # Don't cache intermediate results
                    use_enhanced_penalty=use_enhanced_penalty
                )
                # Extract only the continuation part
                continuation = continuation[len(template):]
                result = template + continuation
            else:
                result = template
        else:
            # Fall back to nucleus sampling
            result = self.generate_with_nucleus_sampling(prompt, max_length, use_enhanced_penalty=use_enhanced_penalty)
        
        return result
    
    def generate(self, prompt, max_length=100, use_template=True, use_enhanced_penalty=True):
        """Main generation method that selects the best approach.
        
        Args:
            prompt: The input prompt
            max_length: Maximum length of generated text
            use_template: Whether to use templates when available
            use_enhanced_penalty: Whether to use enhanced n-gram based repetition penalty
            
        Returns:
            The generated text
        """
        if use_template and self.templates:
            return self.generate_with_template(prompt, max_length, use_enhanced_penalty=use_enhanced_penalty)
        else:
            return self.generate_with_nucleus_sampling(prompt, max_length, use_enhanced_penalty=use_enhanced_penalty)


def compare_generation_methods(model_path, tokenizer_path, templates_file=None, prompts=None):
    """Compare different generation methods.
    
    Args:
        model_path: Path to the pretrained model
        tokenizer_path: Path to the tokenizer
        templates_file: Optional path to templates file
        prompts: List of prompts to test, or None for defaults
    """
    if prompts is None:
        prompts = [
            "ኢትዮጵያ",
            "አዲስ አበባ",
            "የኢትዮጵያ ባህል",
            "ትምህርት በኢትዮጵያ"
        ]
    
    # Load the original practical generator
    try:
        practical_generator = PracticalAmharicGenerator(model_path, tokenizer_path)
    except Exception as e:
        print(f"Error loading practical generator: {e}")
        practical_generator = None
    
    # Load the improved generator
    improved_generator = ImprovedAmharicGenerator(model_path, tokenizer_path, templates_file)
    
    # Compare generation methods
    for prompt in prompts:
        print(f"\n{'='*80}\nPrompt: {prompt}\n{'='*80}")
        
        # Original generation with repetition penalty
        if practical_generator:
            start_time = time.time()
            original_output = practical_generator.generate_with_repetition_penalty(prompt)
            original_time = time.time() - start_time
            print(f"\n--- Original with Repetition Penalty (took {original_time:.3f}s) ---")
            print(original_output)
        
        # Improved generation with nucleus sampling
        start_time = time.time()
        improved_output = improved_generator.generate_with_nucleus_sampling(prompt)
        improved_time = time.time() - start_time
        print(f"\n--- Improved with Nucleus Sampling (took {improved_time:.3f}s) ---")
        print(improved_output)
        
        # Template-based generation if templates are available
        if templates_file and os.path.exists(templates_file):
            start_time = time.time()
            template_output = improved_generator.generate_with_template(prompt)
            template_time = time.time() - start_time
            print(f"\n--- Template-based Generation (took {template_time:.3f}s) ---")
            print(template_output)


def main():
    """Main function to demonstrate improved generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Amharic text generation')
    parser.add_argument('--model', type=str, default='models/enhanced_hnet/model.pt',
                        help='Path to the model')
    parser.add_argument('--tokenizer', type=str, default='models/enhanced_hnet/tokenizer.json',
                        help='Path to the tokenizer')
    parser.add_argument('--templates', type=str, default=None,
                        help='Path to templates file')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt for generation')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different generation methods')
    
    args = parser.parse_args()
    
    # Ensure model path is absolute
    if not os.path.isabs(args.model):
        args.model = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model)
    
    # Ensure tokenizer path is absolute
    if not os.path.isabs(args.tokenizer):
        args.tokenizer = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.tokenizer)
    
    # Ensure templates path is absolute if provided
    if args.templates and not os.path.isabs(args.templates):
        args.templates = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.templates)
    
    if args.compare:
        # Compare different generation methods
        prompts = [args.prompt] if args.prompt else None
        compare_generation_methods(args.model, args.tokenizer, args.templates, prompts)
    else:
        # Simple generation demo
        generator = ImprovedAmharicGenerator(args.model, args.tokenizer, args.templates)
        
        if args.prompt:
            prompt = args.prompt
        else:
            prompt = input("Enter a prompt: ")
        
        print("\nGenerating...")
        start_time = time.time()
        output = generator.generate(prompt)
        generation_time = time.time() - start_time
        
        print(f"\nGenerated text (took {generation_time:.3f}s):")
        print(output)


if __name__ == "__main__":
    main()
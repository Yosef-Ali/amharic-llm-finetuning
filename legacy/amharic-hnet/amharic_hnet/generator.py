# Amharic H-Net: Text Generator Implementation

import os
import json
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from .model import HNetTransformer


class ImprovedAmharicGenerator:
    """Improved text generator for Amharic language using the H-Net model."""
    
    def __init__(
        self,
        model: HNetTransformer,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the generator with a model and tokenizer.
        
        Args:
            model: The HNetTransformer model
            tokenizer: The tokenizer for the model
            device: The device to run the model on ('cpu', 'cuda', or None for auto-detection)
            cache_dir: Directory to cache generated outputs for faster retrieval
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        # Set up cache
        self.cache_dir = cache_dir
        self.cache = {}
        if cache_dir and os.path.exists(cache_dir):
            self._load_cache()
    
    def _load_cache(self):
        """Load the generation cache from disk."""
        cache_file = os.path.join(self.cache_dir, "generation_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save the generation cache to disk."""
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, "generation_cache.json")
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error saving cache: {e}")
    
    def _get_cache_key(self, prompt, **kwargs):
        """Generate a cache key for the given prompt and generation parameters."""
        # Sort kwargs to ensure consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        kwargs_str = ",".join([f"{k}={v}" for k, v in sorted_kwargs])
        return f"{prompt}|{kwargs_str}"
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        min_length: int = 10,
        do_sample: bool = True,
        num_beams: int = 5,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        use_cache: bool = True,
        clean_up_tokenization_spaces: bool = True,
        return_generation_time: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, float]]:
        """Generate text based on the given prompt.
        
        Args:
            prompt: The input text to continue from
            max_length: Maximum length of the generated text
            min_length: Minimum length of the generated text
            do_sample: Whether to use sampling or greedy decoding
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeating tokens
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            use_cache: Whether to use the generation cache
            clean_up_tokenization_spaces: Whether to clean up tokenization spaces
            return_generation_time: Whether to return the generation time
            **kwargs: Additional arguments for the model's generate method
            
        Returns:
            The generated text or a tuple of (generated text, generation time)
        """
        # Check cache first if enabled
        if use_cache and self.cache_dir:
            cache_key = self._get_cache_key(
                prompt,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                **kwargs
            )
            
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if return_generation_time:
                    return cached_result["text"], cached_result["time"]
                return cached_result["text"]
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Start timing
        start_time = time.time()
        
        # Generate text
        with torch.no_grad():
            output_sequences = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                **kwargs
            )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(
            output_sequences[0],
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        
        # Update cache if enabled
        if use_cache and self.cache_dir:
            self.cache[cache_key] = {
                "text": generated_text,
                "time": generation_time
            }
            self._save_cache()
        
        if return_generation_time:
            return generated_text, generation_time
        return generated_text
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts in batches.
        
        Args:
            prompts: List of input prompts
            batch_size: Batch size for generation
            **kwargs: Additional arguments for generate_text
            
        Returns:
            List of generated texts corresponding to each prompt
        """
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize all prompts in the batch
            batch_inputs = self.tokenizer(batch_prompts, padding=True, return_tensors="pt").to(self.device)
            
            # Generate text for the batch
            with torch.no_grad():
                output_sequences = self.model.generate(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    **kwargs
                )
            
            # Decode each sequence
            for j, sequence in enumerate(output_sequences):
                generated_text = self.tokenizer.decode(
                    sequence,
                    clean_up_tokenization_spaces=kwargs.get("clean_up_tokenization_spaces", True)
                )
                results.append(generated_text)
        
        return results
    
    def interactive_generate(
        self,
        initial_prompt: Optional[str] = None,
        max_turns: int = 10,
        **kwargs
    ) -> None:
        """Run an interactive generation session in the console.
        
        Args:
            initial_prompt: Optional starting prompt
            max_turns: Maximum number of conversation turns
            **kwargs: Additional arguments for generate_text
        """
        print("=== Amharic H-Net Interactive Generation ===")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'settings' to view current generation settings.")
        
        context = initial_prompt if initial_prompt else ""
        
        for _ in range(max_turns):
            if context:
                print(f"\nContext: {context}\n")
            
            user_input = input("> ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Ending session.")
                break
            
            if user_input.lower() == "settings":
                print("\nCurrent generation settings:")
                for k, v in kwargs.items():
                    print(f"{k}: {v}")
                continue
            
            # Update context with user input
            if context:
                prompt = f"{context}\n{user_input}"
            else:
                prompt = user_input
            
            # Generate response
            generated_text, gen_time = self.generate_text(
                prompt=prompt,
                return_generation_time=True,
                **kwargs
            )
            
            # Extract the new content (remove the prompt)
            new_content = generated_text[len(prompt):]
            
            print(f"\nGenerated (in {gen_time:.2f}s):")
            print(new_content)
            
            # Update context
            context = generated_text
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """Load a generator from pretrained model and tokenizer.
        
        Args:
            model_path: Path to the pretrained model
            tokenizer_path: Path to the tokenizer (defaults to model_path if None)
            device: Device to load the model on
            cache_dir: Directory to cache generated outputs
            **kwargs: Additional arguments for model loading
            
        Returns:
            An instance of ImprovedAmharicGenerator
        """
        from transformers import AutoTokenizer
        
        # Load the model
        model = HNetTransformer.from_pretrained(model_path, **kwargs)
        
        # Load the tokenizer
        if tokenizer_path is None:
            tokenizer_path = model_path
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        return cls(model, tokenizer, device, cache_dir)
    
    def save_pretrained(self, save_directory: str):
        """Save the generator's model and tokenizer.
        
        Args:
            save_directory: Directory to save the model and tokenizer
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save the model
        self.model.save_pretrained(save_directory)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save the cache if it exists
        if self.cache and self.cache_dir:
            cache_dir = os.path.join(save_directory, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(cache_dir, "generation_cache.json")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
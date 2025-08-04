#!/usr/bin/env python3
"""Amharic text generation CLI."""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from amharichnet.inference import create_generator


def main():
    parser = argparse.ArgumentParser(description="Generate Amharic text using trained H-Net model")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt to start generation")
    parser.add_argument("--length", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--model", type=str, 
                       default="outputs/amharic_optimized_training/checkpoints/ckpt.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str,
                       default="configs/amharic_optimized.yaml", 
                       help="Path to model config")
    
    args = parser.parse_args()
    
    print("🚀 Initializing Amharic Text Generator...")
    
    try:
        # Create generator
        generator = create_generator(
            model_path=args.model,
            config_path=args.config
        )
        
        print(f"\n📝 Generating text...")
        print(f"   Prompt: '{args.prompt}'")
        print(f"   Max length: {args.length}")
        print(f"   Temperature: {args.temperature}")
        print(f"   Samples: {args.num_samples}")
        
        # Generate text
        results = generator.generate(
            prompt=args.prompt,
            max_length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_samples
        )
        
        print(f"\n📄 Generated Text:")
        print("=" * 50)
        
        for i, text in enumerate(results, 1):
            print(f"\nSample {i}:")
            print(f"'{text}'")
            print("-" * 30)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def demo():
    """Run a quick demo of text generation."""
    print("🎯 Amharic Text Generation Demo")
    print("=" * 40)
    
    try:
        generator = create_generator()
        
        # Demo prompts
        prompts = [
            "ኢትዮጵያ",
            "አዲስ አበባ",
            "ሰላም",
            "ወንድሜ",
            ""  # Empty prompt
        ]
        
        for prompt in prompts:
            print(f"\n📝 Prompt: '{prompt}' → ", end="")
            result = generator.complete_text(prompt, max_new_tokens=30)
            print(f"'{result}'")
        
        print(f"\n🎨 Creative generation:")
        creative = generator.generate_creative("ጥበብ")
        print(f"'{creative}'")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments, run demo
        demo()
    else:
        # Run CLI
        main()
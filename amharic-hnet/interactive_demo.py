#!/usr/bin/env python3
"""
Interactive Demo of Enhanced H-Net Model
Real-time prompt testing with various Amharic inputs
"""

import torch
import time
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_train import EnhancedHNet, EnhancedAmharicTokenizer

class AmharicHNetDemo:
    def __init__(self):
        """Initialize the demo with trained model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = EnhancedAmharicTokenizer()
        self.tokenizer.load("models/enhanced_tokenizer.pkl")
        
        # Load model
        self.model = EnhancedHNet(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=256,
            hidden_dim=512,
            num_layers=3,
            dropout=0.2
        )
        
        checkpoint = torch.load("models/enhanced_hnet/best_model.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("🤖 Enhanced H-Net Model Loaded Successfully!")
        print(f"📱 Device: {self.device}")
        print(f"📚 Vocabulary: {self.tokenizer.vocab_size} characters")
        print("=" * 60)
    
    def generate_text(self, prompt, max_length=80, temperature=0.8):
        """Generate text from prompt"""
        start_time = time.time()
        
        try:
            generated = self.model.generate(
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                device=self.device
            )
            
            generation_time = time.time() - start_time
            return generated, generation_time
            
        except Exception as e:
            return f"Error: {e}", 0
    
    def demo_prompts(self):
        """Run demo with predefined prompts"""
        
        demo_sets = [
            {
                "category": "🌍 Geography & Places",
                "prompts": [
                    "ኢትዮጵያ",
                    "አዲስ አበባ", 
                    "ላሊበላ",
                    "አክሱም",
                    "ባህር ዳር"
                ]
            },
            {
                "category": "🎭 Culture & Traditions", 
                "prompts": [
                    "ባህል",
                    "ቡና",
                    "እንጀራ",
                    "በዓል",
                    "ሙዚቃ"
                ]
            },
            {
                "category": "📚 Education & Science",
                "prompts": [
                    "ትምህርት",
                    "ዩኒቨርሲቲ",
                    "ሳይንስ",
                    "ምርምር",
                    "መፅሃፍ"
                ]
            },
            {
                "category": "💻 Technology & Modern Life",
                "prompts": [
                    "ኮምፒዩተር",
                    "ኢንተርኔት", 
                    "ሞባይል",
                    "ቴክኖሎጂ",
                    "ፕሮግራም"
                ]
            },
            {
                "category": "🏛️ Society & Government",
                "prompts": [
                    "መንግስት",
                    "ዲሞክራሲ",
                    "ህዝብ",
                    "ፍትህ",
                    "ሰላም"
                ]
            }
        ]
        
        for demo_set in demo_sets:
            print(f"\n{demo_set['category']}")
            print("=" * 50)
            
            for prompt in demo_set['prompts']:
                generated, gen_time = self.generate_text(prompt, max_length=100, temperature=0.7)
                
                # Calculate metrics
                amharic_chars = sum(1 for c in generated if '\u1200' <= c <= '\u137F')
                amharic_ratio = amharic_chars / len(generated) if generated else 0
                
                print(f"\n📝 Prompt: '{prompt}'")
                print(f"⚡ Generated ({gen_time:.3f}s): '{generated}'")
                print(f"📊 Length: {len(generated)} chars | Amharic: {amharic_ratio:.1%} | Speed: {len(generated)/gen_time:.1f} chars/sec")
                print("-" * 50)
    
    def interactive_mode(self):
        """Interactive prompt mode"""
        print("\n🎯 INTERACTIVE MODE")
        print("Enter Amharic prompts (or 'quit' to exit):")
        print("=" * 50)
        
        while True:
            try:
                prompt = input("\n🔤 Enter prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not prompt:
                    continue
                
                # Get generation parameters
                try:
                    max_len = int(input("📏 Max length (default 80): ") or "80")
                    temp = float(input("🌡️ Temperature (default 0.7): ") or "0.7")
                except ValueError:
                    max_len, temp = 80, 0.7
                
                # Generate
                generated, gen_time = self.generate_text(prompt, max_len, temp)
                
                # Display result
                amharic_chars = sum(1 for c in generated if '\u1200' <= c <= '\u137F')
                amharic_ratio = amharic_chars / len(generated) if generated else 0
                
                print(f"\n✨ RESULT:")
                print(f"📝 Input: '{prompt}'")
                print(f"🤖 Output: '{generated}'")
                print(f"📊 Stats: {len(generated)} chars | {amharic_ratio:.1%} Amharic | {gen_time:.3f}s | {len(generated)/gen_time:.1f} chars/sec")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main demo function"""
    print("🚀 ENHANCED H-NET INTERACTIVE DEMO")
    print("1000-Article Amharic Corpus Training")
    print("=" * 60)
    
    try:
        demo = AmharicHNetDemo()
        
        # Run predefined demos
        demo.demo_prompts()
        
        # Ask for interactive mode
        response = input("\n🤔 Would you like to try interactive mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            demo.interactive_mode()
        
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("Please ensure the model has been trained first.")
    except Exception as e:
        print(f"❌ Error initializing demo: {e}")

if __name__ == "__main__":
    main()
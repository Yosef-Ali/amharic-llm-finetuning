#!/usr/bin/env python3
"""Simple Amharic text generation without checkpoints."""

import sys
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from amharichnet.data.amharic_tokenizer import AmharicSubwordTokenizer


class SimpleAmharicGenerator:
    """Simple Amharic text generator using patterns."""
    
    def __init__(self):
        # Initialize tokenizer
        self.tokenizer = AmharicSubwordTokenizer()
        try:
            self.tokenizer.load_vocab("models/tokenizer/amharic_vocab.json")
            print(f"✅ Loaded Amharic tokenizer (vocab: {self.tokenizer.vocab_size})")
        except FileNotFoundError:
            print("⚠️  Using basic tokenizer")
        
        # Common Amharic patterns and continuations
        self.patterns = {
            "ኢትዮጵያ": [
                "ውብ ሀገር ናት",
                "በአፍሪካ ቀንድ ትገኛለች",
                "ባለ ብዙ ቋንቋዎች ሀገር ናት",
                "የጥንት ሥልጣኔ ያላት ሀገር ናት",
                "ብዙ ብሔሮችና ብሔረሰቦች የሚኖሩባት ሀገር ናት"
            ],
            "አዲስ አበባ": [
                "የኢትዮጵያ ዋና ከተማ ናት",
                "የአፍሪካ መዲና ተብላ ትጠራለች",
                "በሰላሳ ዓመታት ውስጥ በፍጥነት እያደገች ትገኛለች",
                "ዓለም አቀፍ ድርጅቶች ዋና መሥሪያ ቤት ናት"
            ],
            "ሰላም": [
                "ለሁሉም ሰው ይሁን",
                "የማንም ጠላት አይደለም",
                "በውሸት የሚገኝ አይደለም",
                "በፍቅር እና በመከባበር ይገኛል"
            ],
            "ወንድሜ": [
                "እንዴት ነህ?",
                "በሰላም ነህ?",
                "ጤና ይስጥልን",
                "ደህና ነህ?"
            ],
            "ትምህርት": [
                "ሀገር ለመገንባት አስፈላጊ ነው",
                "ለእያንዳንዱ ሰው መብት ነው",
                "የወደፊት ተስፋ ነው",
                "በጥራት መሰጠት አለበት"
            ],
            "ወጣቶች": [
                "የሀገሪቱ አንጋፋ ተስፋ ናቸው",
                "በትምህርት እና በሥራ ላይ ማተኮር አለባቸው",
                "ለሀገራቸው እድገት አስተዋጽኦ ማድረግ አለባቸው"
            ]
        }
        
        # General Amharic sentence patterns
        self.general_patterns = [
            "በዚህ ጊዜ",
            "በተጨማሪም",
            "ስለዚህ",
            "በመሆኑም",
            "ይህ ማለት",
            "በአንፃሩ",
            "በእርግጥ",
            "በተለይም"
        ]
        
        # Common Amharic endings
        self.endings = [
            "ነው።",
            "ናት።",
            "ናቸው።",
            "ይሆናል።",
            "አለ።",
            "ተብሏል።",
            "ይላል።",
            "ይባላል።"
        ]
    
    def generate_continuation(self, prompt: str) -> str:
        """Generate continuation for a given prompt."""
        prompt = prompt.strip()
        
        # Check for direct patterns
        for key, continuations in self.patterns.items():
            if key in prompt:
                return f"{prompt} {random.choice(continuations)}"
        
        # Generate based on prompt analysis
        if not prompt:
            # Start with a common phrase
            starters = [
                "ኢትዮጵያ ውብ ሀገር ናት።",
                "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
                "ሰላም ለሁሉም ይሁን።",
                "ትምህርት ለሁሉም ህጻናት መብት ነው።",
                "ወጣቶች የሀገሪቱ ተስፋ ናቸው።"
            ]
            return random.choice(starters)
        
        # Add a general continuation
        continuation = random.choice(self.general_patterns)
        ending = random.choice(self.endings)
        
        # Try to create a meaningful continuation
        if "ኢትዮጵያ" in prompt or "ሀገር" in prompt:
            middle = "ይህ ሀገር ብዙ ውብ ባህሎች ያሉት"
        elif "ትምህርት" in prompt or "ተማሪ" in prompt:
            middle = "ትምህርት ለሁሉም ህጻናት አስፈላጊ"
        elif "ሰላም" in prompt:
            middle = "ሰላም ለሁሉም ሰው አስፈላጊ"
        else:
            middle = "ይህ ጉዳይ በጣም አስፈላጊ"
        
        return f"{prompt} {continuation} {middle} {ending}"
    
    def generate_multiple(self, prompt: str, count: int = 3) -> list:
        """Generate multiple variations."""
        results = []
        for _ in range(count):
            result = self.generate_continuation(prompt)
            results.append(result)
        return results
    
    def test_tokenizer(self, text: str):
        """Test tokenization on generated text."""
        print(f"\n🔤 Tokenizer Test:")
        print(f"   Text: '{text}'")
        
        # Encode
        tokens = self.tokenizer.encode(text, max_len=64)
        print(f"   Tokens ({len(tokens)}): {tokens[:10]}...")
        
        # Decode
        decoded = self.tokenizer.decode(tokens)
        print(f"   Decoded: '{decoded}'")
        
        return tokens, decoded


def main():
    print("🎯 Simple Amharic Text Generator")
    print("=" * 40)
    
    generator = SimpleAmharicGenerator()
    
    # Test prompts
    test_prompts = [
        "",
        "ኢትዮጵያ",
        "አዲስ አበባ", 
        "ሰላም ወንድሜ",
        "ትምህርት",
        "ወጣቶች"
    ]
    
    print(f"\n📝 Text Generation Tests:")
    for prompt in test_prompts:
        print(f"\n🔸 Prompt: '{prompt}'")
        result = generator.generate_continuation(prompt)
        print(f"   Result: '{result}'")
        
        # Test tokenizer
        generator.test_tokenizer(result)
    
    print(f"\n🎨 Multiple Variations:")
    variations = generator.generate_multiple("ኢትዮጵያ", count=3)
    for i, var in enumerate(variations, 1):
        print(f"   {i}. {var}")
    
    print(f"\n✅ Generation test complete!")


if __name__ == "__main__":
    main()
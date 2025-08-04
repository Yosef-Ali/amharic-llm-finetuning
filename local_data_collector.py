#!/usr/bin/env python3
"""
Local Amharic Data Collector - Works Offline
"""

import json
import random
from pathlib import Path
from datetime import datetime

class LocalAmharicDataCollector:
    def __init__(self):
        self.data_dir = Path("data/collected")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample Amharic sentences for offline development
        self.sample_sentences = [
            "ሰላም ወንድሜ እንዴት ነህ?",
            "ኢትዮጵያ ውብ ሀገር ናት።",
            "አማርኛ መማር እፈልጋለሁ።",
            "ዛሬ ጥሩ ቀን ነው።",
            "መጽሐፍ ማንበብ እወዳለሁ።",
            "ቡና መጠጣት ባህላችን ነው።",
            "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
            "ትምህርት የልማት መሰረት ነው።",
            "ሰላም እና ፍቅር ይሁንላችሁ።",
            "የአማርኛ ቋንቋ ታሪክ ረጅም ነው።"
        ]
        
    def generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic Amharic data for development"""
        print(f"Generating {num_samples} synthetic Amharic samples...")
        
        data = []
        for i in range(num_samples):
            # Combine random sentences
            num_sentences = random.randint(3, 8)
            text = " ".join(random.choices(self.sample_sentences, k=num_sentences))
            
            data.append({
                "id": f"synthetic_{i:04d}",
                "text": text,
                "source": "synthetic",
                "timestamp": datetime.now().isoformat(),
                "word_count": len(text.split()),
                "char_count": len(text)
            })
        
        # Save data
        output_file = self.data_dir / f"synthetic_amharic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {num_samples} samples to {output_file}")
        return data
    
    def create_training_data(self):
        """Create formatted training data"""
        print("Creating training data format...")
        
        # Load all collected data
        all_texts = []
        for json_file in self.data_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_texts.extend([item['text'] for item in data])
        
        # Save as plain text for training
        train_file = Path("data/processed/train.txt")
        train_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_texts))
        
        print(f"✅ Created training file with {len(all_texts)} samples")
        
if __name__ == "__main__":
    collector = LocalAmharicDataCollector()
    collector.generate_synthetic_data(1000)
    collector.create_training_data()

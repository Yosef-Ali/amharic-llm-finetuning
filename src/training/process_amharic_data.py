import json
import os
from pathlib import Path
import sys
from collections import Counter
import random

def process_amharic_for_cpu():
    """Process Amharic data efficiently for CPU training"""
    
    print("📊 Processing Amharic data for CPU training...")
    print("=" * 50)
    
    # Check available data
    data_dir = Path("data")
    processed_data = []
    
    # Ensure data directories exist
    data_dir.mkdir(exist_ok=True)
    (data_dir / "processed").mkdir(exist_ok=True)
    (data_dir / "collected").mkdir(exist_ok=True)
    
    print(f"📂 Scanning data directory: {data_dir.absolute()}")
    
    # Look for various data file formats
    data_files = []
    for pattern in ["**/*.jsonl", "**/*.json", "**/*.txt"]:
        data_files.extend(data_dir.glob(pattern))
    
    if not data_files:
        print("⚠️  No data files found. Creating sample data for demonstration...")
        create_sample_data(data_dir)
        # Re-scan for the created sample data
        data_files = list(data_dir.glob("**/*.jsonl"))
    
    print(f"📁 Found {len(data_files)} data files")
    
    # Process files efficiently
    for data_file in data_files:
        print(f"   📄 Processing: {data_file.name}")
        
        try:
            if data_file.suffix == '.jsonl':
                processed_count = process_jsonl_file(data_file, processed_data)
            elif data_file.suffix == '.json':
                processed_count = process_json_file(data_file, processed_data)
            elif data_file.suffix == '.txt':
                processed_count = process_txt_file(data_file, processed_data)
            else:
                continue
                
            print(f"      ✅ Added {processed_count} samples")
            
            # Limit for CPU training
            if len(processed_data) >= 2000:
                print(f"   🛑 Reached limit of 2000 samples for CPU training")
                break
                
        except Exception as e:
            print(f"      ❌ Error processing {data_file}: {e}")
            continue
    
    print(f"\n✅ Processed {len(processed_data)} total samples")
    
    if len(processed_data) == 0:
        print("❌ No data processed. Please check your data files.")
        return None, None
    
    # Filter and clean data
    cleaned_data = clean_and_filter_data(processed_data)
    print(f"🧹 After cleaning: {len(cleaned_data)} samples")
    
    # Save processed data
    output_file = data_dir / "processed" / "cpu_training_data.jsonl"
    save_processed_data(cleaned_data, output_file)
    
    # Create byte statistics
    byte_stats = create_byte_statistics(cleaned_data)
    
    # Save statistics
    stats_file = data_dir / "processed" / "byte_statistics.json"
    save_statistics(byte_stats, cleaned_data, stats_file)
    
    print(f"\n📈 Data Processing Summary:")
    print(f"   Total samples: {len(cleaned_data)}")
    print(f"   Vocabulary size: {len(byte_stats)} unique bytes")
    print(f"   Average text length: {sum(len(item['text']) for item in cleaned_data) / len(cleaned_data):.1f} characters")
    print(f"   Average byte length: {sum(item['byte_length'] for item in cleaned_data) / len(cleaned_data):.1f} bytes")
    
    return cleaned_data, byte_stats

def process_jsonl_file(file_path, processed_data):
    """Process JSONL file"""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                text = extract_text_from_item(item)
                
                if is_valid_amharic_text(text):
                    processed_item = create_processed_item(text, str(file_path), line_num)
                    processed_data.append(processed_item)
                    count += 1
                    
                    if len(processed_data) >= 2000:
                        break
                        
            except json.JSONDecodeError:
                continue
            except Exception as e:
                continue
    
    return count

def process_json_file(file_path, processed_data):
    """Process JSON file"""
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle different JSON structures
        if isinstance(data, list):
            for i, item in enumerate(data):
                text = extract_text_from_item(item)
                if is_valid_amharic_text(text):
                    processed_item = create_processed_item(text, str(file_path), i)
                    processed_data.append(processed_item)
                    count += 1
                    
                    if len(processed_data) >= 2000:
                        break
        elif isinstance(data, dict):
            text = extract_text_from_item(data)
            if is_valid_amharic_text(text):
                processed_item = create_processed_item(text, str(file_path), 0)
                processed_data.append(processed_item)
                count += 1
                
    except Exception as e:
        pass
    
    return count

def process_txt_file(file_path, processed_data):
    """Process plain text file"""
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                text = line.strip()
                if is_valid_amharic_text(text):
                    processed_item = create_processed_item(text, str(file_path), line_num)
                    processed_data.append(processed_item)
                    count += 1
                    
                    if len(processed_data) >= 2000:
                        break
                        
    except Exception as e:
        pass
    
    return count

def extract_text_from_item(item):
    """Extract text from various item formats"""
    if isinstance(item, str):
        return item
    elif isinstance(item, dict):
        # Try common text field names
        for field in ['text', 'content', 'body', 'message', 'amharic', 'am']:
            if field in item and isinstance(item[field], str):
                return item[field]
        # If no common field, try to find any string value
        for value in item.values():
            if isinstance(value, str) and len(value.strip()) > 0:
                return value
    
    return ""

def is_valid_amharic_text(text):
    """Check if text is valid for training"""
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    
    # Length constraints for CPU training
    if len(text) < 3 or len(text) > 200:
        return False
    
    # Check for Amharic characters (Unicode range for Ethiopic script)
    amharic_chars = sum(1 for char in text if '\u1200' <= char <= '\u137F')
    
    # Must have at least some Amharic characters
    if amharic_chars < 2:
        return False
    
    # Byte length constraint
    byte_length = len(text.encode('utf-8'))
    if byte_length > 300:  # Too long for CPU training
        return False
    
    return True

def create_processed_item(text, source, index):
    """Create a processed data item"""
    return {
        'text': text.strip(),
        'length': len(text.strip()),
        'byte_length': len(text.strip().encode('utf-8')),
        'source': source,
        'index': index
    }

def clean_and_filter_data(data):
    """Clean and filter the processed data"""
    cleaned = []
    seen_texts = set()
    
    for item in data:
        text = item['text']
        
        # Remove duplicates
        if text in seen_texts:
            continue
        seen_texts.add(text)
        
        # Additional cleaning
        text = text.strip()
        
        # Skip if too short after cleaning
        if len(text) < 3:
            continue
        
        # Update item with cleaned text
        item['text'] = text
        item['length'] = len(text)
        item['byte_length'] = len(text.encode('utf-8'))
        
        cleaned.append(item)
    
    # Sort by length for better training
    cleaned.sort(key=lambda x: x['length'])
    
    return cleaned

def save_processed_data(data, output_file):
    """Save processed data to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"💾 Saved processed data to: {output_file}")

def create_byte_statistics(data):
    """Create byte-level statistics"""
    byte_counter = Counter()
    
    for item in data:
        for byte_val in item['text'].encode('utf-8'):
            byte_counter[byte_val] += 1
    
    return dict(byte_counter)

def save_statistics(byte_stats, data, stats_file):
    """Save statistics to file"""
    stats = {
        'total_samples': len(data),
        'vocabulary_size': len(byte_stats),
        'total_bytes': sum(byte_stats.values()),
        'average_text_length': sum(len(item['text']) for item in data) / len(data),
        'average_byte_length': sum(item['byte_length'] for item in data) / len(data),
        'most_common_bytes': sorted(byte_stats.items(), key=lambda x: x[1], reverse=True)[:20],
        'byte_distribution': byte_stats
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Saved statistics to: {stats_file}")
    print(f"📈 Most common bytes: {stats['most_common_bytes'][:5]}")

def create_sample_data(data_dir):
    """Create sample Amharic data for demonstration"""
    print("🔧 Creating sample Amharic data...")
    
    sample_texts = [
        "ሰላም አለም እንዴት ነህ",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት",
        "አማርኛ በኢትዮጵያ የሚነገር ቋንቋ ነው",
        "ጤና ይስጥልኝ እንደምን አለህ",
        "መልካም ቀን ደህና ሁን",
        "እንኳን ደህና መጣህ እንኳን ደህና ወጣህ",
        "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ሀገር ናት",
        "አዲስ አበባ ዩኒቨርሲቲ ትልቅ ዩኒቨርሲቲ ነው",
        "የኢትዮጵያ ህዝብ በተለያዩ ቋንቋዎች ይነጋገራል",
        "ቡና የኢትዮጵያ ባህላዊ መጠጥ ነው",
        "ኢንጀራ የኢትዮጵያ ባህላዊ ምግብ ነው",
        "የኢትዮጵያ ባንዲራ አረንጓዴ ቢጫ እና ቀይ ቀለም አለው",
        "ሐበሻ የኢትዮጵያ እና የኤርትራ ህዝብ ስም ነው",
        "የአማርኛ ፊደላት ከግዕዝ ፊደላት የተወሰዱ ናቸው",
        "ኢትዮጵያ የአፍሪካ ቀንድ ሀገር ናት",
        "አዲስ አበባ የአፍሪካ ዲፕሎማሲያዊ ዋና ከተማ ናት",
        "የኢትዮጵያ ህዝብ ቁጥር ከ100 ሚሊዮን በላይ ነው",
        "ኢትዮጵያ 13 ወር ፀሐይ አላት",
        "የኢትዮጵያ አዲስ ዓመት በሴፕቴምበር ይከበራል",
        "ኢትዮጵያ የተለያዩ ብሔር ብሔረሰቦች የሚኖሩባት ሀገር ናት"
    ]
    
    # Expand the sample data
    expanded_samples = []
    for text in sample_texts:
        expanded_samples.append(text)
        
        # Create variations
        words = text.split()
        if len(words) > 2:
            # Create shorter versions
            for i in range(2, len(words)):
                expanded_samples.append(" ".join(words[:i]))
    
    # Add more variations
    additional_phrases = [
        "ሰላም", "እንዴት ነህ", "አዲስ አበባ", "ኢትዮጵያ", "አማርኛ",
        "ጤና ይስጥልኝ", "መልካም ቀን", "ደህና ሁን", "እንኳን ደህና",
        "ቡና", "ኢንጀራ", "ሐበሻ", "አፍሪካ", "ዩኒቨርሲቲ"
    ]
    
    expanded_samples.extend(additional_phrases * 10)  # Repeat for more data
    
    # Shuffle the data
    random.shuffle(expanded_samples)
    
    # Save sample data
    sample_file = data_dir / "collected" / "sample_amharic_data.jsonl"
    sample_file.parent.mkdir(exist_ok=True)
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        for i, text in enumerate(expanded_samples):
            item = {
                'text': text,
                'source': 'sample_data',
                'id': i
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {len(expanded_samples)} sample texts in {sample_file}")

def split_data_for_training(data, train_ratio=0.8, val_ratio=0.1):
    """Split data into train/validation/test sets"""
    random.shuffle(data)
    
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    print("🇪🇹 Amharic Data Processing for CPU Training")
    print("=" * 50)
    
    # Process the data
    processed_data, byte_stats = process_amharic_for_cpu()
    
    if processed_data:
        print("\n🎯 Data processing completed successfully!")
        
        # Split data for training
        train_data, val_data, test_data = split_data_for_training(processed_data)
        
        print(f"\n📊 Data Split:")
        print(f"   Training: {len(train_data)} samples")
        print(f"   Validation: {len(val_data)} samples")
        print(f"   Test: {len(test_data)} samples")
        
        # Save split data
        data_dir = Path("data/processed")
        
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            split_file = data_dir / f"{split_name}_data.jsonl"
            with open(split_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"💾 Saved {split_name} data to: {split_file}")
        
        print("\n✅ Ready for CPU training!")
        print("💡 Next step: Run 'python cpu_train_amharic.py' to train the model")
    else:
        print("❌ Data processing failed")
import os
import json
from pathlib import Path

def check_dataset_status():
    """Check what datasets are available and their sizes"""
    print("=== DATASET CHECK ===\n")
    
    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ No data directory found!")
        return
    
    # Check for prepared data
    prepared_dir = data_dir / "prepared"
    if prepared_dir.exists():
        print("✅ Found prepared data directory")
        files = list(prepared_dir.glob("*.txt"))
        print(f"   - Found {len(files)} text files")
        
        total_size = 0
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"   - {f.name}: {size_mb:.2f} MB")
        
        print(f"\n   Total size: {total_size:.2f} MB")
    else:
        print("❌ No prepared data found")
    
    # Check for tokenizer
    tokenizer_path = data_dir / "tokenizer"
    if tokenizer_path.exists():
        print("\n✅ Found tokenizer directory")
        files = list(tokenizer_path.glob("*"))
        for f in files:
            print(f"   - {f.name}")
    else:
        print("\n❌ No tokenizer found")
    
    # Check for raw data
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        print("\n✅ Found raw data directory")
        files = list(raw_dir.glob("*"))
        print(f"   - {len(files)} files")
    
    # Check config
    config_files = list(Path("configs").glob("*.yaml"))
    if config_files:
        print("\n✅ Found config files:")
        for cf in config_files:
            print(f"   - {cf.name}")
            # Try to read dataset info
            try:
                with open(cf) as f:
                    content = f.read()
                    if "dataset" in content.lower():
                        print("     Contains dataset configuration")
            except:
                pass

if __name__ == "__main__":
    check_dataset_status()

#!/usr/bin/env python3
"""
Find and analyze all data files in the project
"""
import os
from pathlib import Path
import json

def analyze_data_directory(base_path="data"):
    """Find all data files and analyze their content"""
    print("=== DATA DIRECTORY ANALYSIS ===\n")
    
    data_stats = {}
    total_files = 0
    total_size = 0
    total_lines = 0
    
    # Check each subdirectory
    subdirs = ["raw", "collected", "processed", "augmented", "training", "prepared"]
    
    for subdir in subdirs:
        dir_path = Path(base_path) / subdir
        if not dir_path.exists():
            print(f"❌ {subdir:12} - Does not exist")
            continue
            
        # Find all text files
        text_files = list(dir_path.glob("**/*.txt")) + \
                    list(dir_path.glob("**/*.json")) + \
                    list(dir_path.glob("**/*.jsonl"))
        
        if not text_files:
            print(f"⚠️  {subdir:12} - Empty (no text files)")
            continue
        
        dir_size = 0
        dir_lines = 0
        file_info = []
        
        for file in text_files[:5]:  # Show first 5 files
            size_mb = file.stat().st_size / (1024 * 1024)
            dir_size += size_mb
            
            # Try to count lines
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                    dir_lines += lines
                    file_info.append(f"     - {file.name}: {size_mb:.2f}MB, {lines:,} lines")
            except:
                file_info.append(f"     - {file.name}: {size_mb:.2f}MB, [error reading]")
        
        total_files += len(text_files)
        total_size += dir_size
        total_lines += dir_lines
        
        print(f"✅ {subdir:12} - {len(text_files)} files, {dir_size:.2f}MB")
        for info in file_info:
            print(info)
        if len(text_files) > 5:
            print(f"     ... and {len(text_files) - 5} more files")
        print()
    
    print(f"\n📊 TOTAL SUMMARY:")
    print(f"   - Total files: {total_files}")
    print(f"   - Total size: {total_size:.2f}MB")
    print(f"   - Total lines: {total_lines:,}")
    
    # Check for specific Amharic data patterns
    print("\n🔍 Looking for Amharic content...")
    amharic_files = []
    
    for subdir in subdirs:
        dir_path = Path(base_path) / subdir
        if dir_path.exists():
            for file in dir_path.glob("**/*"):
                if file.is_file() and file.suffix in ['.txt', '.json', '.jsonl']:
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            sample = f.read(200)
                            # Check for Amharic characters
                            if any('\u1200' <= char <= '\u137F' for char in sample):
                                amharic_files.append(file)
                                if len(amharic_files) <= 3:
                                    print(f"   ✅ Found Amharic in: {file.relative_to(base_path)}")
                                    print(f"      Sample: {sample[:50]}...")
                    except:
                        pass
    
    print(f"\n   Total files with Amharic content: {len(amharic_files)}")
    
    # Save summary
    summary = {
        "total_files": total_files,
        "total_size_mb": total_size,
        "total_lines": total_lines,
        "amharic_files": len(amharic_files),
        "directories": {
            subdir: len(list((Path(base_path) / subdir).glob("**/*"))) 
            if (Path(base_path) / subdir).exists() else 0
            for subdir in subdirs
        }
    }
    
    with open("data_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n💾 Summary saved to data_analysis_summary.json")

if __name__ == "__main__":
    analyze_data_directory()

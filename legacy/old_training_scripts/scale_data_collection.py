#!/usr/bin/env python3
"""
Scale Data Collection Script
Runs multiple collection cycles to reach 10K+ samples
"""

import subprocess
import time
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_collection_cycle(cycle_num: int, target_per_cycle: int = 2000):
    """Run a single collection cycle"""
    logger.info(f"🔄 Starting collection cycle {cycle_num} (target: {target_per_cycle} samples)")
    
    try:
        # Modify the enhanced_data_collector to use different target
        result = subprocess.run(
            ['python', 'enhanced_data_collector.py'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Cycle {cycle_num} completed successfully")
            return True
        else:
            logger.error(f"❌ Cycle {cycle_num} failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Cycle {cycle_num} timed out")
        return False
    except Exception as e:
        logger.error(f"💥 Cycle {cycle_num} error: {e}")
        return False

def count_total_samples():
    """Count total samples across all files"""
    data_dir = Path("data")
    total_samples = 0
    
    # Count JSONL files
    for jsonl_file in data_dir.rglob("*.jsonl"):
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                lines = sum(1 for line in f if line.strip())
                total_samples += lines
                logger.info(f"📄 {jsonl_file.name}: {lines} samples")
        except Exception as e:
            logger.warning(f"⚠️ Error reading {jsonl_file}: {e}")
    
    # Count JSON files
    for json_file in data_dir.rglob("*amharic*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    count = len(data)
                elif isinstance(data, dict):
                    count = len(data.get('articles', [])) + len(data.get('conversations', []))
                else:
                    count = 1
                total_samples += count
                logger.info(f"📄 {json_file.name}: {count} samples")
        except Exception as e:
            logger.warning(f"⚠️ Error reading {json_file}: {e}")
    
    return total_samples

def main():
    """Main scaling function"""
    print("🚀 Scaling Amharic Data Collection to 10K+ Samples")
    print("====================================================")
    
    target_total = 10000
    max_cycles = 5
    
    for cycle in range(1, max_cycles + 1):
        # Check current total
        current_total = count_total_samples()
        logger.info(f"📊 Current total samples: {current_total}")
        
        if current_total >= target_total:
            logger.info(f"🎯 Target reached! Total samples: {current_total}")
            break
        
        remaining = target_total - current_total
        target_this_cycle = min(2000, remaining)
        
        logger.info(f"🎯 Need {remaining} more samples")
        
        # Run collection cycle
        success = run_collection_cycle(cycle, target_this_cycle)
        
        if not success:
            logger.warning(f"⚠️ Cycle {cycle} failed, continuing...")
        
        # Wait between cycles
        if cycle < max_cycles:
            logger.info("⏳ Waiting 30 seconds before next cycle...")
            time.sleep(30)
    
    # Final count
    final_total = count_total_samples()
    logger.info(f"\n🏁 Final Results:")
    logger.info(f"📊 Total samples collected: {final_total}")
    logger.info(f"🎯 Target achievement: {(final_total/target_total)*100:.1f}%")
    
    if final_total >= target_total:
        logger.info("🎉 SUCCESS: Target of 10K+ samples achieved!")
    else:
        logger.info(f"📈 Progress made: {final_total} samples collected")
    
    print("\n✅ Scaling complete!")
    print(f"Total samples: {final_total}")
    print("Next steps:")
    print("1. Run: python smart_train.py")
    print("2. Run: python smart_amharic_app.py")

if __name__ == "__main__":
    main()
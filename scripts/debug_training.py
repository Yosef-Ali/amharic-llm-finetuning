import json
import sys
import os
from pathlib import Path
import time

def check_training_artifacts(run_dir="outputs/run"):
    """Debug helper to check training artifacts"""
    run_path = Path(run_dir)
    
    print("=== TRAINING DEBUG REPORT ===\n")
    
    # 1. Check if directory exists
    if not run_path.exists():
        print(f"❌ ERROR: {run_dir} does not exist!")
        print("   Have you run training yet?")
        return
    
    # 2. Check metrics.json
    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        print("✅ Found metrics.json")
        print(f"   - Training time: {metrics.get('training_time', 'N/A')}s")
        print(f"   - Epochs completed: {metrics.get('epoch', 'N/A')}")
        print(f"   - Best val loss: {metrics.get('best_val_loss', 'N/A')}")
        print(f"   - Final val loss: {metrics.get('val_loss', 'N/A')}")
    else:
        print("❌ No metrics.json found")
    
    # 3. Check checkpoints
    ckpt_dir = run_path / "checkpoints"
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*.pt"))
        print(f"\n✅ Found {len(ckpts)} checkpoint(s):")
        for ckpt in ckpts:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"   - {ckpt.name}: {size_mb:.1f} MB")
    else:
        print("\n❌ No checkpoints directory found")
    
    # 4. Check CSV logs
    csv_files = list(run_path.glob("*.csv"))
    if csv_files:
        print(f"\n✅ Found {len(csv_files)} CSV log file(s):")
        for csv in csv_files:
            print(f"   - {csv.name}")
            # Show first few lines
            with open(csv) as f:
                lines = f.readlines()
                if len(lines) > 1:
                    print(f"     Headers: {lines[0].strip()}")
                    print(f"     Rows: {len(lines)-1}")
    else:
        print("\n❌ No CSV logs found")
    
    # 5. Check config
    config_path = run_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print("\n✅ Found config.json")
        print(f"   - Dataset: {config.get('dataset_name', 'N/A')}")
        print(f"   - Batch size: {config.get('batch_size', 'N/A')}")
        print(f"   - Max epochs: {config.get('max_epochs', 'N/A')}")
        print(f"   - Learning rate: {config.get('learning_rate', 'N/A')}")
    
    print("\n" + "="*30)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_training_artifacts(sys.argv[1])
    else:
        check_training_artifacts()

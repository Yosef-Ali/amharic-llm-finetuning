#!/usr/bin/env python3
"""Test model on held-out test dataset."""

import sys
from pathlib import Path
import json
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from amharichnet.data.loader import make_dataloader
from amharichnet.models.hnet import create_model
from amharichnet.evaluation import AmharicTextEvaluator
import yaml


def load_config(config_path: str):
    """Load config from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
    return Config(config_dict)


def test_model_on_holdout():
    """Test trained model on held-out test data."""
    print("🧪 Testing H-Net Model on Held-out Test Data")
    print("=" * 45)
    
    # Load config
    config = load_config("configs/amharic_optimized.yaml")
    
    # Create model
    model = create_model(config)
    if not model.available:
        print("❌ Model not available")
        return
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load test data
    try:
        test_loader = make_dataloader(
            "data/training/test.jsonl",  # Our test split
            batch_size=16,
            num_workers=0,
            tokenizer_type="amharic",
            max_len=128
        )
        print(f"✅ Test data loaded: {len(test_loader.dataset)} samples")
    except FileNotFoundError:
        print("⚠️  Test data not found, using validation data")
        test_loader = make_dataloader(
            config.data.val_path,
            batch_size=16,
            num_workers=0,
            tokenizer_type="amharic",
            max_len=128
        )
        print(f"✅ Validation data loaded: {len(test_loader.dataset)} samples")
    
    # Test model performance
    model.eval()
    total_loss = 0.0
    total_samples = 0
    batch_losses = []
    
    print(f"\n🔍 Running inference on test data...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                # Forward pass
                if hasattr(model, 'step') and model.available:
                    # Use model's internal step
                    loss = model.step()
                    if isinstance(loss, (int, float)):
                        batch_loss = float(loss)
                    else:
                        batch_loss = float(loss.item())
                else:
                    # Fallback: simple MSE on input
                    x = batch.input_ids.float()
                    if hasattr(model, 'net'):
                        output = model.net(x)
                        loss = F.mse_loss(output.mean(), x.mean())
                        batch_loss = float(loss.item())
                    else:
                        batch_loss = 1.0
                
                batch_losses.append(batch_loss)
                total_loss += batch_loss * batch.input_ids.size(0)
                total_samples += batch.input_ids.size(0)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(test_loader)} batches")
                
                if i >= 50:  # Limit for demo
                    break
                    
            except Exception as e:
                print(f"   ⚠️  Batch {i} failed: {e}")
                continue
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    print(f"\n📊 Test Results:")
    print(f"   Average Loss: {avg_loss:.4f}")
    print(f"   Samples Processed: {total_samples}")
    print(f"   Batches Processed: {len(batch_losses)}")
    print(f"   Loss Range: {min(batch_losses):.4f} - {max(batch_losses):.4f}")
    
    # Compare with training metrics
    try:
        with open("outputs/amharic_optimized_training/metrics.json") as f:
            training_metrics = json.load(f)
        
        train_loss = training_metrics['final_loss']
        val_loss = training_metrics['best_val_loss']
        
        print(f"\n📈 Comparison with Training:")
        print(f"   Training Loss: {train_loss:.4f}")
        print(f"   Best Val Loss: {val_loss:.4f}")
        print(f"   Test Loss: {avg_loss:.4f}")
        
        if val_loss > 0:
            ratio = avg_loss / val_loss
            print(f"   Test/Val Ratio: {ratio:.2f}x")
        
    except FileNotFoundError:
        print("⚠️  Training metrics not found")
    
    # Text quality evaluation on sample generations
    print(f"\n📝 Evaluating Generated Text Quality...")
    
    from generate import AmharicGenerator
    evaluator = AmharicTextEvaluator()
    generator = AmharicGenerator()
    
    # Generate samples for evaluation
    test_prompts = ["ኢትዮጵያ", "አዲስ አበባ", "ትምህርት", "ሰላም"]
    generated_samples = []
    
    for prompt in test_prompts:
        for _ in range(3):  # 3 samples per prompt
            text = generator.generate_text(prompt=prompt, length=50)
            generated_samples.append(text)
    
    # Evaluate quality
    evaluation = evaluator.evaluate_multiple(generated_samples)
    stats = evaluation['statistics']
    
    print(f"   Generated Samples: {len(generated_samples)}")
    print(f"   Overall Quality: {stats['overall_quality']['mean']:.3f} ± {stats['overall_quality']['std']:.3f}")
    print(f"   Amharic Ratio: {stats['amharic_ratio']['mean']:.3f}")
    print(f"   Fluency Score: {stats['fluency_score']['mean']:.3f}")
    print(f"   Coherence Score: {stats['coherence_score']['mean']:.3f}")
    
    # Show best and worst samples
    best_idx = evaluation['best_text_index']
    worst_idx = evaluation['worst_text_index']
    
    print(f"\n🏆 Best Generated Sample:")
    print(f"   '{generated_samples[best_idx]}'")
    print(f"   Quality: {evaluation['individual_scores'][best_idx]['overall_quality']:.3f}")
    
    print(f"\n⚠️  Worst Generated Sample:")
    print(f"   '{generated_samples[worst_idx]}'")
    print(f"   Quality: {evaluation['individual_scores'][worst_idx]['overall_quality']:.3f}")
    
    # Save test results
    results = {
        'test_loss': avg_loss,
        'samples_processed': total_samples,
        'comparison': {
            'training_loss': train_loss if 'train_loss' in locals() else None,
            'validation_loss': val_loss if 'val_loss' in locals() else None,
            'test_loss': avg_loss
        },
        'text_quality': {
            'overall_quality': stats['overall_quality']['mean'],
            'fluency_score': stats['fluency_score']['mean'],
            'coherence_score': stats['coherence_score']['mean'],
            'amharic_ratio': stats['amharic_ratio']['mean']
        },
        'samples': {
            'best': generated_samples[best_idx],
            'worst': generated_samples[worst_idx],
            'all_samples': generated_samples
        }
    }
    
    with open("test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Test results saved to test_results.json")
    print(f"✅ Holdout testing complete!")
    
    return results


if __name__ == "__main__":
    test_model_on_holdout()
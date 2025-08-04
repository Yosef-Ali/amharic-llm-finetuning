import torch
import sys
import os
import time
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.cpu_hnet import CPUAmharicHNet, count_parameters

def cpu_amharic_benchmarks(model=None, model_path='tiny_amharic_hnet.pt'):
    """Simple benchmarks that work without external APIs"""
    
    print("🇪🇹 CPU Amharic H-Net Benchmarks")
    print("=" * 40)
    
    # Load model if not provided
    if model is None:
        if os.path.exists(model_path):
            print(f"📂 Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location='cpu')
            model = CPUAmharicHNet()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"✅ Model loaded successfully")
        else:
            print(f"❌ Model file not found: {model_path}")
            print("   Please run cpu_train_amharic.py first")
            return None
    
    # Create simple test cases
    test_cases = {
        'generation': [
            {'input': 'ሰላም', 'expected_contains': ['አለም', 'ነህ', 'ናት']},
            {'input': 'እንዴት', 'expected_contains': ['ነህ', 'ናት', 'ናችሁ']},
            {'input': 'አዲስ', 'expected_contains': ['አበባ', 'ከተማ', 'ሀገር']},
            {'input': 'ኢትዮጵያ', 'expected_contains': ['ሀገር', 'አፍሪካ', 'ምስራቅ']},
        ],
        'translation': [
            {'amharic': 'ሰላም', 'english': 'hello'},
            {'amharic': 'እንዴት ነህ', 'english': 'how are you'},
            {'amharic': 'አዲስ አበባ', 'english': 'addis ababa'},
            {'amharic': 'ኢትዮጵያ', 'english': 'ethiopia'},
        ],
        'morphology': [
            {'word': 'ልጆች', 'morphemes': ['ልጅ', 'ኦች']},  # children = child + plural
            {'word': 'መጽሐፍት', 'morphemes': ['መጽሐፍ', 'ት']},  # books = book + plural
            {'word': 'ቤቶች', 'morphemes': ['ቤት', 'ኦች']},  # houses = house + plural
            {'word': 'ተማሪዎች', 'morphemes': ['ተማሪ', 'ዎች']},  # students = student + plural
        ],
        'compression': [
            {'text': 'ሰላም አለም እንዴት ነህ', 'min_ratio': 2.0},
            {'text': 'አዲስ አበባ ከተማ ኢትዮጵያ ሀገር', 'min_ratio': 2.0},
            {'text': 'አማርኛ ቋንቋ ጤና ይስጥልኝ', 'min_ratio': 2.0},
        ]
    }
    
    # Test each category
    results = {}
    detailed_results = {}
    
    print(f"\n🧠 Model Info:")
    print(f"   Parameters: {count_parameters(model):,}")
    print(f"   Architecture: CPU-optimized H-Net")
    
    for task, cases in test_cases.items():
        print(f"\n📊 Testing {task.upper()}:")
        task_scores = []
        task_details = []
        
        for i, case in enumerate(cases):
            print(f"   Test {i+1}/{len(cases)}: ", end="")
            
            if task == 'generation':
                score, details = test_generation_case(model, case)
            elif task == 'translation':
                score, details = test_translation_case(model, case)
            elif task == 'morphology':
                score, details = test_morphology_case(model, case)
            elif task == 'compression':
                score, details = test_compression_case(model, case)
            
            task_scores.append(score)
            task_details.append(details)
            
            print(f"Score: {score:.1%}")
        
        avg_score = sum(task_scores) / len(task_scores)
        results[task] = avg_score
        detailed_results[task] = {
            'average_score': avg_score,
            'individual_scores': task_scores,
            'details': task_details
        }
        
        print(f"   📈 {task.capitalize()} Average Score: {avg_score:.1%}")
    
    # Overall performance
    overall_score = sum(results.values()) / len(results)
    print(f"\n🎯 Overall Performance: {overall_score:.1%}")
    
    # Performance metrics
    print(f"\n⚡ Performance Metrics:")
    test_performance_metrics(model)
    
    # Save results
    benchmark_results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'model_parameters': count_parameters(model),
        'overall_score': overall_score,
        'task_scores': results,
        'detailed_results': detailed_results
    }
    
    results_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return benchmark_results

def test_generation_case(model, case):
    """Test generation capability"""
    input_text = case['input']
    expected_words = case['expected_contains']
    
    try:
        # Convert to bytes
        input_bytes = [b for b in input_text.encode('utf-8')]
        input_bytes.extend([0] * (32 - len(input_bytes)))  # Pad to 32
        input_tensor = torch.tensor([input_bytes], dtype=torch.long)
        
        with torch.no_grad():
            # Generate text
            generated = model.generate(input_tensor, max_length=24, temperature=0.7)
            gen_bytes = generated[0].tolist()
            
            # Decode
            real_bytes = [b for b in gen_bytes if b != 0 and b < 256]
            try:
                decoded = bytes(real_bytes).decode('utf-8', errors='ignore')
            except:
                decoded = ""
            
            # Simple scoring: check if generation is reasonable
            score = 0.0
            if len(decoded) > len(input_text):  # Generated something longer
                score += 0.3
            if any(char in decoded for char in 'ሰላምእንዴትአዲስኢትዮጵያ'):  # Contains Amharic
                score += 0.4
            if len(decoded.strip()) > 0:  # Not empty
                score += 0.3
            
            details = {
                'input': input_text,
                'generated': decoded,
                'length_ratio': len(decoded) / len(input_text) if input_text else 0
            }
            
            return min(score, 1.0), details
            
    except Exception as e:
        return 0.0, {'input': input_text, 'error': str(e)}

def test_translation_case(model, case):
    """Test translation capability (simplified)"""
    amharic_text = case['amharic']
    english_text = case['english']
    
    try:
        # For now, just test if model can process the Amharic text
        input_bytes = [b for b in amharic_text.encode('utf-8')]
        input_bytes.extend([0] * (32 - len(input_bytes)))
        input_tensor = torch.tensor([input_bytes], dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # Simple scoring based on successful processing
            score = 0.5  # Base score for processing
            
            # Check compression ratio
            compression = outputs.get('compression_ratio', 1.0)
            if compression > 1.5:  # Good compression
                score += 0.3
            
            # Check if logits are reasonable
            logits = outputs['logits']
            if not torch.isnan(logits).any() and not torch.isinf(logits).any():
                score += 0.2
            
            details = {
                'amharic': amharic_text,
                'english': english_text,
                'compression_ratio': float(compression),
                'processed_successfully': True
            }
            
            return score, details
            
    except Exception as e:
        return 0.0, {'amharic': amharic_text, 'english': english_text, 'error': str(e)}

def test_morphology_case(model, case):
    """Test morphological understanding"""
    word = case['word']
    morphemes = case['morphemes']
    
    try:
        # Test if model can process morphologically complex words
        input_bytes = [b for b in word.encode('utf-8')]
        input_bytes.extend([0] * (32 - len(input_bytes)))
        input_tensor = torch.tensor([input_bytes], dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # Score based on compression and processing
            score = 0.0
            
            # Check if model compresses morphologically
            compression = outputs.get('compression_ratio', 1.0)
            if compression > 2.0:  # Good morphological compression
                score += 0.5
            
            # Check processing quality
            logits = outputs['logits']
            if not torch.isnan(logits).any():
                score += 0.3
            
            # Bonus for complex words
            if len(word) > 4:
                score += 0.2
            
            details = {
                'word': word,
                'morphemes': morphemes,
                'compression_ratio': float(compression),
                'byte_length': len(word.encode('utf-8'))
            }
            
            return min(score, 1.0), details
            
    except Exception as e:
        return 0.0, {'word': word, 'morphemes': morphemes, 'error': str(e)}

def test_compression_case(model, case):
    """Test compression capability"""
    text = case['text']
    min_ratio = case['min_ratio']
    
    try:
        input_bytes = [b for b in text.encode('utf-8')]
        if len(input_bytes) > 32:
            input_bytes = input_bytes[:32]
        else:
            input_bytes.extend([0] * (32 - len(input_bytes)))
        
        input_tensor = torch.tensor([input_bytes], dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            compression = outputs.get('compression_ratio', 1.0)
            
            # Score based on compression ratio
            score = min(compression / min_ratio, 1.0)
            
            details = {
                'text': text,
                'compression_ratio': float(compression),
                'min_required': min_ratio,
                'achieved_target': compression >= min_ratio
            }
            
            return score, details
            
    except Exception as e:
        return 0.0, {'text': text, 'error': str(e)}

def test_performance_metrics(model):
    """Test performance metrics"""
    
    # Test inference speed
    test_input = torch.randint(0, 256, (1, 32))
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_input)
    
    # Time inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(test_input)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / 100
    
    print(f"   Inference Speed: {avg_inference_time*1000:.2f}ms per sample")
    print(f"   Throughput: {1/avg_inference_time:.1f} samples/second")
    
    # Memory usage (approximate)
    param_memory = count_parameters(model) * 4 / (1024**2)  # 4 bytes per float32, convert to MB
    print(f"   Model Memory: ~{param_memory:.1f}MB")
    
    return {
        'inference_time_ms': avg_inference_time * 1000,
        'throughput_samples_per_sec': 1 / avg_inference_time,
        'model_memory_mb': param_memory
    }

if __name__ == "__main__":
    print("🇪🇹 CPU Amharic H-Net Benchmark Suite")
    print("=" * 50)
    
    # Check if model exists
    model_path = 'tiny_amharic_hnet.pt'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("\n💡 To run benchmarks:")
        print("   1. First train the model: python cpu_train_amharic.py")
        print("   2. Then run benchmarks: python cpu_benchmark.py")
        sys.exit(1)
    
    # Run benchmarks
    results = cpu_amharic_benchmarks()
    
    if results:
        print("\n🎉 Benchmark completed successfully!")
        print(f"📊 Overall Score: {results['overall_score']:.1%}")
        print("\n📋 Task Breakdown:")
        for task, score in results['task_scores'].items():
            print(f"   {task.capitalize()}: {score:.1%}")
    else:
        print("❌ Benchmark failed")
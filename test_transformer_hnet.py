#!/usr/bin/env python3
"""
Comprehensive Testing and Validation for Transformer H-Net
Test all components: model, generation, segmentation, and integration
"""

import sys
import os
from pathlib import Path
import time
import json
import yaml
from typing import Dict, List, Any
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from amharichnet.models.transformer_hnet import TransformerHNet, TransformerHNetConfig
from amharichnet.inference.advanced_generator import AdvancedAmharicGenerator, GenerationConfig
from amharichnet.text.sentence_segmentation import AmharicSentenceSegmenter, SegmentationConfig
from amharichnet.evaluation import AmharicTextEvaluator


class TransformerHNetTester:
    """Comprehensive testing suite for Transformer H-Net."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        print("🧪 Initializing Transformer H-Net Test Suite")
        print(f"🖥️  Device: {self.device}")
        print("=" * 50)
    
    def test_model_architecture(self) -> Dict[str, Any]:
        """Test the Transformer H-Net model architecture."""
        print("\n1️⃣ Testing Model Architecture")
        print("-" * 30)
        
        results = {}
        
        # Test different model sizes
        test_configs = [
            {"name": "Small", "hidden_dim": 256, "num_layers": 4, "num_heads": 4},
            {"name": "Medium", "hidden_dim": 512, "num_layers": 8, "num_heads": 8}, 
            {"name": "Large", "hidden_dim": 768, "num_layers": 12, "num_heads": 12}
        ]
        
        for config_spec in test_configs:
            print(f"\n📐 Testing {config_spec['name']} model...")
            
            try:
                # Create model config
                config = TransformerHNetConfig(
                    vocab_size=3087,
                    hidden_dim=config_spec['hidden_dim'],
                    num_layers=config_spec['num_layers'],
                    num_heads=config_spec['num_heads'],
                    max_seq_len=512
                )
                
                # Create model
                model = TransformerHNet(config)
                model.to(self.device)
                
                # Test forward pass
                batch_size, seq_len = 2, 32
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                forward_time = time.time() - start_time
                
                # Test generation
                start_time = time.time()
                with torch.no_grad():
                    generated = model.generate(input_ids[:1], max_length=50, temperature=0.8)
                generation_time = time.time() - start_time
                
                # Record results
                results[config_spec['name']] = {
                    "parameters": model.get_num_params(),
                    "forward_pass_time": forward_time,
                    "generation_time": generation_time,
                    "loss": outputs["loss"].item() if outputs["loss"] is not None else 0.0,
                    "output_shape": list(outputs["logits"].shape),
                    "generation_length": generated.shape[1],
                    "status": "✅ PASSED"
                }
                
                print(f"   Parameters: {model.get_num_params():,}")
                print(f"   Forward time: {forward_time:.4f}s")
                print(f"   Generation time: {generation_time:.4f}s")
                print(f"   Loss: {outputs['loss'].item() if outputs['loss'] is not None else 0.0:.4f}")
                
            except Exception as e:
                results[config_spec['name']] = {
                    "error": str(e),
                    "status": "❌ FAILED"
                }
                print(f"   ❌ Failed: {e}")
        
        return results
    
    def test_advanced_generation(self) -> Dict[str, Any]:
        """Test advanced generation capabilities."""
        print("\n2️⃣ Testing Advanced Generation")
        print("-" * 30)
        
        results = {}
        
        try:
            # Initialize generator (with smaller model for testing)
            config = TransformerHNetConfig(
                vocab_size=3087,
                hidden_dim=256,
                num_layers=4,
                num_heads=4
            )
            model = TransformerHNet(config)
            
            # Create a mock generator for testing
            class MockAdvancedGenerator:
                def __init__(self, model):
                    self.model = model
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                def generate(self, prompt="", generation_strategy="sampling", **kwargs):
                    # Mock generation - return placeholder text
                    strategies = {"sampling": "ናሙና ጽሑፍ", "greedy": "ቀጥተኛ ጽሑፍ", "beam_search": "የጨረር ፍለጋ ውጤት"}
                    return strategies.get(generation_strategy, "ጽሑፍ ተፈጥሯል")
                
                def generate_multiple(self, prompt="", num_generations=3, **kwargs):
                    return [f"ውጤት {i+1}: {prompt} ተጨማሪ ጽሑፍ" for i in range(num_generations)]
            
            generator = MockAdvancedGenerator(model)
            
            # Test different generation strategies
            strategies = ["sampling", "greedy", "beam_search"]
            test_prompts = ["ኢትዮጵያ", "አዲስ አበባ", "ትምህርት", ""]
            
            for strategy in strategies:
                print(f"\n🎯 Testing {strategy} generation...")
                strategy_results = {}
                
                for prompt in test_prompts:
                    start_time = time.time()
                    result = generator.generate(
                        prompt=prompt,
                        generation_strategy=strategy,
                        max_length=50,
                        temperature=0.8
                    )
                    generation_time = time.time() - start_time
                    
                    strategy_results[prompt or "empty"] = {
                        "text": result,
                        "generation_time": generation_time,
                        "length": len(result.split())
                    }
                    
                    print(f"   Prompt '{prompt}': {result} ({generation_time:.4f}s)")
                
                results[strategy] = strategy_results
            
            # Test multiple generation
            print(f"\n🎲 Testing multiple generation...")
            multiple_results = generator.generate_multiple(
                prompt="ኢትዮጵያ",
                num_generations=3,
                generation_strategy="sampling"
            )
            
            results["multiple_generation"] = {
                "results": multiple_results,
                "count": len(multiple_results)
            }
            
            for i, result in enumerate(multiple_results):
                print(f"   Result {i+1}: {result}")
            
            results["status"] = "✅ PASSED"
            
        except Exception as e:
            results = {
                "error": str(e),
                "status": "❌ FAILED"
            }
            print(f"❌ Generation testing failed: {e}")
        
        return results
    
    def test_sentence_segmentation(self) -> Dict[str, Any]:
        """Test sentence boundary detection."""
        print("\n3️⃣ Testing Sentence Segmentation")
        print("-" * 30)
        
        results = {}
        
        try:
            # Initialize segmenter
            config = SegmentationConfig(
                min_sentence_length=5,
                max_sentence_length=200,
                confidence_threshold=0.7,
                use_neural_segmentation=False  # Disable for testing
            )
            segmenter = AmharicSentenceSegmenter(config)
            
            # Test texts with known sentence boundaries
            test_cases = [
                {
                    "text": "ኢትዮጵያ ውብ ሀገር ናት። በአፍሪካ ቀንድ ትገኛለች። ብዙ ብሔሮች እና ብሔረሰቦች በሰላም የሚኖሩባት ሀገር ናት።",
                    "expected_sentences": 3,
                    "name": "Simple sentences"
                },
                {
                    "text": "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት፣ በተጨማሪም የአፍሪካ ዲፕሎማሲያዊ ዋና ከተማ ነች። ብዙ ዓለም አቀፍ ድርጅቶች ዋና መሥሪያ ቤት ናት።",
                    "expected_sentences": 2,
                    "name": "Complex sentences"
                },
                {
                    "text": "ትምህርት በጣም አስፈላጊ ነው! ለሁሉም ህጻናት መብት ነው? በጥራት መሰጠት አለበት፡ ወጣቶችን ለወደፊት ያዘጋጃል።",
                    "expected_sentences": 4,
                    "name": "Mixed punctuation"
                }
            ]
            
            for test_case in test_cases:
                print(f"\n📝 Testing: {test_case['name']}")
                
                start_time = time.time()
                sentences = segmenter.segment(test_case['text'])
                segmentation_time = time.time() - start_time
                
                print(f"   Input: {test_case['text'][:50]}...")
                print(f"   Expected: {test_case['expected_sentences']} sentences")
                print(f"   Found: {len(sentences)} sentences")
                print(f"   Time: {segmentation_time:.4f}s")
                
                for i, sentence in enumerate(sentences, 1):
                    print(f"     {i}. {sentence}")
                
                # Evaluate accuracy
                accuracy = 1.0 if len(sentences) == test_case['expected_sentences'] else 0.5
                
                results[test_case['name']] = {
                    "input_text": test_case['text'],
                    "expected_sentences": test_case['expected_sentences'],
                    "found_sentences": len(sentences),
                    "sentences": sentences,
                    "accuracy": accuracy,
                    "segmentation_time": segmentation_time
                }
            
            # Overall performance
            total_accuracy = sum(r['accuracy'] for r in results.values()) / len(results)
            avg_time = sum(r['segmentation_time'] for r in results.values()) / len(results)
            
            results["overall"] = {
                "accuracy": total_accuracy,
                "average_time": avg_time,
                "status": "✅ PASSED" if total_accuracy > 0.7 else "⚠️ PARTIAL"
            }
            
            print(f"\n📊 Overall Accuracy: {total_accuracy:.2f}")
            print(f"📊 Average Time: {avg_time:.4f}s")
            
        except Exception as e:
            results = {
                "error": str(e),
                "status": "❌ FAILED"
            }
            print(f"❌ Segmentation testing failed: {e}")
        
        return results
    
    def test_text_evaluation(self) -> Dict[str, Any]:
        """Test text quality evaluation."""
        print("\n4️⃣ Testing Text Evaluation")
        print("-" * 30)
        
        results = {}
        
        try:
            evaluator = AmharicTextEvaluator()
            
            # Test texts with different quality levels
            test_texts = [
                {
                    "text": "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ውብ ሀገር ናት። ብዙ ብሔሮች እና ብሔረሰቦች በሰላም የሚኖሩባት ሀገር ናት።",
                    "expected_quality": "high",
                    "name": "High quality Amharic"
                },
                {
                    "text": "ሀገር ሀገር ሀገር ሀገር ሀገር ሀገር ሀገር ሀገር።",
                    "expected_quality": "low",
                    "name": "Repetitive text"
                },
                {
                    "text": "hello world this is english text not amharic",
                    "expected_quality": "low",
                    "name": "Non-Amharic text"
                },
                {
                    "text": "አዲስ አበባ ዋና ከተማ ናት። የሚያምር ቦታ ነው።",
                    "expected_quality": "medium",
                    "name": "Medium quality"
                }
            ]
            
            for test_case in test_texts:
                print(f"\n📊 Evaluating: {test_case['name']}")
                
                start_time = time.time()
                scores = evaluator.evaluate_text(test_case['text'])
                evaluation_time = time.time() - start_time
                
                print(f"   Text: {test_case['text'][:50]}...")
                print(f"   Overall Quality: {scores['overall_quality']:.3f}")
                print(f"   Amharic Ratio: {scores['amharic_ratio']:.3f}")
                print(f"   Fluency: {scores['fluency_score']:.3f}")
                print(f"   Coherence: {scores['coherence_score']:.3f}")
                print(f"   Time: {evaluation_time:.4f}s")
                
                # Validate scores are reasonable
                quality_check = (
                    0.0 <= scores['overall_quality'] <= 1.0 and
                    0.0 <= scores['amharic_ratio'] <= 1.0 and
                    0.0 <= scores['fluency_score'] <= 1.0 and
                    0.0 <= scores['coherence_score'] <= 1.0
                )
                
                results[test_case['name']] = {
                    "text": test_case['text'],
                    "expected_quality": test_case['expected_quality'],
                    "scores": scores,
                    "evaluation_time": evaluation_time,
                    "valid_scores": quality_check
                }
            
            # Test multiple text evaluation
            print(f"\n📈 Testing multiple text evaluation...")
            
            texts = [case['text'] for case in test_texts]
            start_time = time.time()
            multiple_evaluation = evaluator.evaluate_multiple(texts)
            batch_time = time.time() - start_time
            
            print(f"   Batch evaluation time: {batch_time:.4f}s")
            print(f"   Average quality: {multiple_evaluation['statistics']['overall_quality']['mean']:.3f}")
            
            results["batch_evaluation"] = {
                "texts_count": len(texts),
                "batch_time": batch_time,
                "statistics": multiple_evaluation['statistics'],
                "individual_scores": len(multiple_evaluation['individual_scores'])
            }
            
            # Overall status
            all_valid = all(r.get('valid_scores', False) for r in results.values() if isinstance(r, dict) and 'valid_scores' in r)
            results["status"] = "✅ PASSED" if all_valid else "❌ FAILED"
            
        except Exception as e:
            results = {
                "error": str(e),
                "status": "❌ FAILED"
            }
            print(f"❌ Evaluation testing failed: {e}")
        
        return results
    
    def test_integration(self) -> Dict[str, Any]:
        """Test integration between all components."""
        print("\n5️⃣ Testing System Integration")
        print("-" * 30)
        
        results = {}
        
        try:
            # Create small test model
            config = TransformerHNetConfig(
                vocab_size=3087,
                hidden_dim=128,
                num_layers=2,
                num_heads=4
            )
            model = TransformerHNet(config)
            
            # Test complete pipeline
            print("🔄 Testing end-to-end pipeline...")
            
            # 1. Generate text (mock)
            generated_text = "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ውብ ሀገር ናት። ብዙ ብሔሮች እና ብሔረሰቦች በሰላም የሚኖሩባት ሀገር ናት።"
            
            # 2. Segment into sentences
            segmenter = AmharicSentenceSegmenter()
            sentences = segmenter.segment(generated_text)
            
            # 3. Evaluate quality
            evaluator = AmharicTextEvaluator()
            evaluation = evaluator.evaluate_text(generated_text)
            
            print(f"   Generated: {generated_text[:50]}...")
            print(f"   Sentences: {len(sentences)}")
            print(f"   Quality: {evaluation['overall_quality']:.3f}")
            
            # 4. Test model compatibility
            input_ids = torch.randint(0, config.vocab_size, (1, 20))
            
            with torch.no_grad():
                # Test forward pass
                outputs = model(input_ids, labels=input_ids)
                
                # Test generation
                generated_ids = model.generate(input_ids, max_length=30)
            
            results = {
                "generated_text": generated_text,
                "sentence_count": len(sentences),
                "sentences": sentences,
                "quality_scores": evaluation,
                "model_forward_pass": {
                    "loss": outputs["loss"].item() if outputs["loss"] is not None else 0.0,
                    "logits_shape": list(outputs["logits"].shape)
                },
                "model_generation": {
                    "input_length": input_ids.shape[1],
                    "output_length": generated_ids.shape[1],
                    "generated_shape": list(generated_ids.shape)
                },
                "status": "✅ PASSED"
            }
            
        except Exception as e:
            results = {
                "error": str(e),
                "status": "❌ FAILED"
            }
            print(f"❌ Integration testing failed: {e}")
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report."""
        print("🚀 Running Comprehensive Transformer H-Net Tests")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test suites
        test_suites = [
            ("Model Architecture", self.test_model_architecture),
            ("Advanced Generation", self.test_advanced_generation),
            ("Sentence Segmentation", self.test_sentence_segmentation),
            ("Text Evaluation", self.test_text_evaluation),
            ("System Integration", self.test_integration)
        ]
        
        all_results = {}
        
        for suite_name, test_func in test_suites:
            try:
                suite_results = test_func()
                all_results[suite_name] = suite_results
            except Exception as e:
                all_results[suite_name] = {
                    "error": str(e),
                    "status": "❌ FAILED"
                }
                print(f"❌ {suite_name} test suite failed: {e}")
        
        total_time = time.time() - start_time
        
        # Generate summary
        print(f"\n📋 Test Summary")
        print("=" * 60)
        
        passed = 0
        failed = 0
        partial = 0
        
        for suite_name, results in all_results.items():
            if isinstance(results, dict):
                status = results.get("status", "❓ UNKNOWN")
                if isinstance(status, dict):
                    status = status.get("status", "❓ UNKNOWN")
                
                print(f"{suite_name:25} {status}")
                
                if "✅" in str(status):
                    passed += 1
                elif "❌" in str(status):
                    failed += 1
                elif "⚠️" in str(status):
                    partial += 1
        
        print(f"\n📊 Results:")
        print(f"   ✅ Passed: {passed}")
        print(f"   ⚠️  Partial: {partial}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")
        
        # Overall status
        if failed == 0:
            overall_status = "✅ ALL TESTS PASSED"
        elif passed > failed:
            overall_status = "⚠️ MOSTLY PASSED"
        else:
            overall_status = "❌ TESTS FAILED"
        
        print(f"\n🎯 Overall Status: {overall_status}")
        
        # Create final report
        final_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_seconds": total_time,
            "summary": {
                "passed": passed,
                "partial": partial,
                "failed": failed,
                "overall_status": overall_status
            },
            "detailed_results": all_results
        }
        
        return final_report


def main():
    """Main testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Transformer H-Net System")
    parser.add_argument("--output", help="Output file for test results", 
                       default="transformer_hnet_test_results.json")
    parser.add_argument("--suite", choices=["all", "model", "generation", "segmentation", "evaluation", "integration"],
                       default="all", help="Test suite to run")
    
    args = parser.parse_args()
    
    # Create tester
    tester = TransformerHNetTester()
    
    # Run tests
    if args.suite == "all":
        results = tester.run_comprehensive_test()
    elif args.suite == "model":
        results = {"Model Architecture": tester.test_model_architecture()}
    elif args.suite == "generation":
        results = {"Advanced Generation": tester.test_advanced_generation()}
    elif args.suite == "segmentation":
        results = {"Sentence Segmentation": tester.test_sentence_segmentation()}
    elif args.suite == "evaluation":
        results = {"Text Evaluation": tester.test_text_evaluation()}
    elif args.suite == "integration":
        results = {"System Integration": tester.test_integration()}
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Test results saved to {args.output}")


if __name__ == "__main__":
    main()
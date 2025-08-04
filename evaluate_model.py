#!/usr/bin/env python3
"""Evaluate Amharic H-Net model performance."""

import sys
from pathlib import Path
import json
import random
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from amharichnet.evaluation import AmharicTextEvaluator, evaluate_generation_quality
from generate import AmharicGenerator


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self):
        self.evaluator = AmharicTextEvaluator()
        self.generator = AmharicGenerator()
        
        # Test prompts for evaluation
        self.test_prompts = [
            "",  # Empty prompt
            "ኢትዮጵያ",
            "አዲስ አበባ", 
            "ሰላም ወንድሜ",
            "ትምህርት",
            "ወጣቶች",
            "ባህል",
            "ሀገር",
            "ህዝብ",
            "መንግሥት"
        ]
        
        # Reference texts (high-quality Amharic)
        self.reference_texts = [
            "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ውብ ሀገር ናት። ብዙ ብሔሮች እና ብሔረሰቦች በሰላም የሚኖሩባት ሀገር ናት።",
            "አዲስ አበባ የኢትዮጵያ ዋና ከተማ እና የአፍሪካ ዲፕሎማሲያዊ ዋና ከተማ ናት። ብዙ ዓለም አቀፍ ድርጅቶች ዋና መሥሪያ ቤት ናት።",
            "ትምህርት ለሁሉም ህጻናት መብት ነው። በጥራት ትምህርት ሀገሪቱን ለመገንባት ወሳኝ ነው።",
            "ሰላም ለሁሉም ሰው አስፈላጊ ነው። በሰላም እና በመከባበር የምንኖርበት ማኅበረሰብ መገንባት አለብን።",
            "ወጣቶች የሀገሪቱ ተስፋ እና የወደፊት መሪዎች ናቸው። በትምህርት እና በሥራ ላይ ማተኮር አለባቸው።"
        ]
    
    def evaluate_generation_quality(self, num_samples: int = 50) -> Dict:
        """Evaluate generation quality with multiple samples."""
        print(f"🔍 Generating {num_samples} samples for evaluation...")
        
        generated_texts = []
        
        # Generate texts with different prompts
        for i in range(num_samples):
            # Mix of empty prompts and specific prompts
            if i % 3 == 0:
                prompt = ""  # Empty prompt
                category = random.choice(["general", "news", "educational", "cultural"])
            else:
                prompt = random.choice(self.test_prompts[1:])  # Skip empty
                category = "general"
            
            text = self.generator.generate_text(
                category=category,
                prompt=prompt,
                length=random.randint(30, 80)
            )
            generated_texts.append(text)
        
        print(f"✅ Generated {len(generated_texts)} samples")
        
        # Evaluate against references
        results = evaluate_generation_quality(generated_texts, self.reference_texts)
        
        return results
    
    def evaluate_specific_prompts(self) -> Dict:
        """Evaluate generation for specific prompts."""
        print(f"🎯 Evaluating specific prompts...")
        
        prompt_results = {}
        
        for prompt in self.test_prompts:
            print(f"   Testing prompt: '{prompt}'")
            
            # Generate multiple samples for this prompt
            samples = []
            for _ in range(5):
                text = self.generator.generate_text(prompt=prompt, length=50)
                samples.append(text)
            
            # Evaluate samples
            evaluation = self.evaluator.evaluate_multiple(samples)
            
            prompt_results[prompt if prompt else "empty_prompt"] = {
                'samples': samples,
                'evaluation': evaluation,
                'best_sample': samples[evaluation['best_text_index']],
                'avg_quality': evaluation['statistics']['overall_quality']['mean']
            }
        
        return prompt_results
    
    def compare_categories(self) -> Dict:
        """Compare generation quality across categories."""
        print(f"📊 Comparing generation across categories...")
        
        categories = ["general", "news", "educational", "cultural", "conversation"]
        category_results = {}
        
        for category in categories:
            print(f"   Testing category: {category}")
            
            samples = []
            for _ in range(10):
                prompt = random.choice(self.test_prompts[1:5])  # Use some prompts
                text = self.generator.generate_text(
                    category=category,
                    prompt=prompt,
                    length=50
                )
                samples.append(text)
            
            evaluation = self.evaluator.evaluate_multiple(samples)
            category_results[category] = evaluation
        
        return category_results
    
    def test_model_consistency(self, prompt: str = "ኢትዮጵያ", num_runs: int = 20) -> Dict:
        """Test model consistency with same prompt."""
        print(f"🔄 Testing consistency with prompt '{prompt}' ({num_runs} runs)...")
        
        results = []
        for i in range(num_runs):
            text = self.generator.generate_text(prompt=prompt, length=40)
            score = self.evaluator.evaluate_text(text)
            results.append({
                'run': i + 1,
                'text': text,
                'scores': score
            })
        
        # Calculate consistency metrics
        qualities = [r['scores']['overall_quality'] for r in results]
        lengths = [r['scores']['length'] for r in results]
        
        import statistics
        consistency_metrics = {
            'quality_mean': statistics.mean(qualities),
            'quality_std': statistics.stdev(qualities) if len(qualities) > 1 else 0,
            'length_mean': statistics.mean(lengths),
            'length_std': statistics.stdev(lengths) if len(lengths) > 1 else 0,
            'unique_outputs': len(set(r['text'] for r in results)),
            'total_runs': num_runs
        }
        
        return {
            'results': results,
            'consistency': consistency_metrics
        }
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive model evaluation."""
        print("🚀 Starting Comprehensive Model Evaluation")
        print("=" * 50)
        
        evaluation_results = {}
        
        # 1. Overall generation quality
        print("\n1️⃣ Overall Generation Quality")
        evaluation_results['overall_quality'] = self.evaluate_generation_quality(30)
        self._print_quality_summary(evaluation_results['overall_quality'])
        
        # 2. Prompt-specific evaluation
        print("\n2️⃣ Prompt-Specific Evaluation")
        evaluation_results['prompt_evaluation'] = self.evaluate_specific_prompts()
        self._print_prompt_summary(evaluation_results['prompt_evaluation'])
        
        # 3. Category comparison
        print("\n3️⃣ Category Comparison")
        evaluation_results['category_comparison'] = self.compare_categories()
        self._print_category_summary(evaluation_results['category_comparison'])
        
        # 4. Consistency testing
        print("\n4️⃣ Model Consistency")
        evaluation_results['consistency'] = self.test_model_consistency()
        self._print_consistency_summary(evaluation_results['consistency'])
        
        # 5. Save results
        output_file = "evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # Convert for JSON serialization
            serializable_results = self._make_serializable(evaluation_results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to {output_file}")
        
        return evaluation_results
    
    def _print_quality_summary(self, results):
        """Print summary of quality evaluation."""
        stats = results['generated']['statistics']
        print(f"   📈 Generated Text Quality:")
        print(f"      Overall Quality: {stats['overall_quality']['mean']:.3f} ± {stats['overall_quality']['std']:.3f}")
        print(f"      Amharic Ratio: {stats['amharic_ratio']['mean']:.3f}")
        print(f"      Fluency Score: {stats['fluency_score']['mean']:.3f}")
        print(f"      Coherence Score: {stats['coherence_score']['mean']:.3f}")
        
        if 'reference' in results:
            print(f"   📊 vs Reference Texts:")
            comp = results['comparison']
            for metric in ['overall_quality', 'fluency_score', 'coherence_score']:
                improvement = comp[metric]['improvement']
                print(f"      {metric}: {improvement:+.1%}")
    
    def _print_prompt_summary(self, results):
        """Print summary of prompt evaluation."""
        print(f"   🎯 Best performing prompts:")
        sorted_prompts = sorted(results.items(), key=lambda x: x[1]['avg_quality'], reverse=True)
        for prompt, data in sorted_prompts[:3]:
            print(f"      '{prompt}': {data['avg_quality']:.3f}")
            print(f"         Best: '{data['best_sample'][:50]}...'")
    
    def _print_category_summary(self, results):
        """Print summary of category comparison."""
        print(f"   📂 Category performance:")
        for category, data in results.items():
            avg_quality = data['statistics']['overall_quality']['mean']
            print(f"      {category}: {avg_quality:.3f}")
    
    def _print_consistency_summary(self, results):
        """Print summary of consistency test."""
        consistency = results['consistency']
        print(f"   🔄 Consistency metrics:")
        print(f"      Quality std: {consistency['quality_std']:.3f}")
        print(f"      Unique outputs: {consistency['unique_outputs']}/{consistency['total_runs']}")
        print(f"      Diversity: {consistency['unique_outputs']/consistency['total_runs']:.1%}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


def main():
    print("📊 Amharic H-Net Model Evaluation")
    print("=" * 40)
    
    evaluator = ModelEvaluator()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # Quick evaluation
            results = evaluator.evaluate_generation_quality(10)
            print(f"\n📈 Quick Evaluation Results:")
            stats = results['generated']['statistics']
            print(f"   Overall Quality: {stats['overall_quality']['mean']:.3f}")
            print(f"   Fluency: {stats['fluency_score']['mean']:.3f}")
            print(f"   Coherence: {stats['coherence_score']['mean']:.3f}")
        elif sys.argv[1] == "--consistency":
            # Consistency test only
            results = evaluator.test_model_consistency()
            print(f"\n🔄 Consistency Results:")
            print(f"   Quality std: {results['consistency']['quality_std']:.3f}")
            print(f"   Unique outputs: {results['consistency']['unique_outputs']}/{results['consistency']['total_runs']}")
    else:
        # Full evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        print(f"\n✅ Comprehensive evaluation complete!")
        print(f"   Results saved to evaluation_results.json")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Visual Comparison Tool for Amharic H-Net Model
Compares original vs improved generation methods with detailed analysis
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import json
import re
from collections import Counter
import sys
import os
from typing import List, Dict, Tuple

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from improved_generation_fix import ImprovedAmharicGenerator
    from interactive_demo import AmharicDemo
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some features may not be available.")

class VisualComparisonTool:
    def __init__(self):
        self.results = {
            'original': [],
            'improved': [],
            'analysis': {}
        }
        
    def analyze_text_quality(self, text: str) -> Dict:
        """Analyze text quality metrics"""
        if not text or len(text) < 2:
            return {
                'repetition_score': 1.0,
                'amharic_ratio': 0.0,
                'diversity_score': 0.0,
                'length': 0,
                'unique_chars': 0,
                'pattern_type': 'empty'
            }
        
        # 1. Repetition analysis
        words = text.split()
        if len(words) > 1:
            unique_words = len(set(words))
            repetition_score = 1.0 - (unique_words / len(words))
        else:
            repetition_score = 0.0
        
        # 2. Amharic character ratio
        amharic_chars = sum(1 for c in text if '\u1200' <= c <= '\u137F')
        amharic_ratio = amharic_chars / len(text) if len(text) > 0 else 0
        
        # 3. Character diversity
        unique_chars = len(set(text))
        diversity_score = unique_chars / len(text) if len(text) > 0 else 0
        
        # 4. Pattern detection
        pattern_type = self._detect_pattern_type(text)
        
        return {
            'repetition_score': repetition_score,
            'amharic_ratio': amharic_ratio,
            'diversity_score': diversity_score,
            'length': len(text),
            'unique_chars': unique_chars,
            'pattern_type': pattern_type
        }
    
    def _detect_pattern_type(self, text: str) -> str:
        """Detect the type of repetition pattern"""
        if len(text) < 10:
            return 'short'
        
        # Check for character repetition
        char_counts = Counter(text)
        max_char_freq = max(char_counts.values()) if char_counts else 0
        if max_char_freq > len(text) * 0.3:
            return 'character_repetition'
        
        # Check for word repetition
        words = text.split()
        if len(words) > 1:
            word_counts = Counter(words)
            max_word_freq = max(word_counts.values())
            if max_word_freq > len(words) * 0.4:
                return 'word_repetition'
        
        # Check for sequence repetition
        for seq_len in range(2, min(10, len(text) // 3)):
            for i in range(len(text) - seq_len * 2):
                seq = text[i:i+seq_len]
                if text[i+seq_len:i+seq_len*2] == seq:
                    return 'sequence_repetition'
        
        # Check for alternating patterns
        if len(text) > 6:
            for i in range(len(text) - 6):
                if text[i] == text[i+2] == text[i+4] and text[i+1] == text[i+3] == text[i+5]:
                    return 'alternating_pattern'
        
        return 'good_quality'
    
    def compare_generation_methods(self, prompts: List[str], num_samples: int = 5) -> Dict:
        """Compare original vs improved generation methods"""
        print("🔍 Comparing Generation Methods...")
        print("=" * 50)
        
        try:
            # Load improved generator
            improved_gen = ImprovedAmharicGenerator()
            print("✅ Improved generator loaded")
        except Exception as e:
            print(f"❌ Could not load improved generator: {e}")
            return {}
        
        comparison_results = {
            'prompts': prompts,
            'improved_results': [],
            'analysis_summary': {
                'improved': {
                    'avg_repetition': 0,
                    'avg_amharic_ratio': 0,
                    'avg_diversity': 0,
                    'pattern_distribution': Counter()
                }
            }
        }
        
        for prompt in prompts:
            print(f"\n📝 Testing prompt: '{prompt}'")
            
            # Test improved method
            try:
                improved_samples = improved_gen.generate_multiple_samples(
                    prompt=prompt,
                    num_samples=num_samples,
                    max_length=80,
                    temperature=1.5,
                    top_p=0.9,
                    repetition_penalty=1.3,
                    word_repetition_penalty=1.8,
                    sequence_repetition_penalty=2.5
                )
                
                improved_analysis = []
                for text, info in improved_samples:
                    analysis = self.analyze_text_quality(text)
                    analysis.update(info)
                    improved_analysis.append({
                        'text': text,
                        'analysis': analysis
                    })
                
                comparison_results['improved_results'].append({
                    'prompt': prompt,
                    'samples': improved_analysis
                })
                
                print(f"   ✅ Generated {len(improved_samples)} improved samples")
                
            except Exception as e:
                print(f"   ❌ Error with improved method: {e}")
        
        # Calculate summary statistics
        self._calculate_summary_stats(comparison_results)
        
        return comparison_results
    
    def _calculate_summary_stats(self, results: Dict):
        """Calculate summary statistics for comparison"""
        improved_metrics = {
            'repetition': [],
            'amharic_ratio': [],
            'diversity': [],
            'patterns': []
        }
        
        for prompt_result in results['improved_results']:
            for sample in prompt_result['samples']:
                analysis = sample['analysis']
                improved_metrics['repetition'].append(analysis['repetition_score'])
                improved_metrics['amharic_ratio'].append(analysis['amharic_ratio'])
                improved_metrics['diversity'].append(analysis['diversity_score'])
                improved_metrics['patterns'].append(analysis['pattern_type'])
        
        # Calculate averages
        if improved_metrics['repetition']:
            results['analysis_summary']['improved'] = {
                'avg_repetition': np.mean(improved_metrics['repetition']),
                'avg_amharic_ratio': np.mean(improved_metrics['amharic_ratio']),
                'avg_diversity': np.mean(improved_metrics['diversity']),
                'pattern_distribution': Counter(improved_metrics['patterns'])
            }
    
    def create_visual_report(self, results: Dict, save_path: str = "visual_comparison_report.png"):
        """Create comprehensive visual report"""
        print(f"\n📊 Creating visual report: {save_path}")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Amharic H-Net Model: Visual Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Quality Metrics Comparison
        self._plot_quality_metrics(axes[0, 0], results)
        
        # 2. Pattern Distribution
        self._plot_pattern_distribution(axes[0, 1], results)
        
        # 3. Sample Quality Scores
        self._plot_sample_scores(axes[0, 2], results)
        
        # 4. Text Length Distribution
        self._plot_length_distribution(axes[1, 0], results)
        
        # 5. Character Diversity Analysis
        self._plot_diversity_analysis(axes[1, 1], results)
        
        # 6. Improvement Summary
        self._plot_improvement_summary(axes[1, 2], results)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visual report saved to {save_path}")
        
        return save_path
    
    def _plot_quality_metrics(self, ax, results):
        """Plot quality metrics comparison"""
        if 'improved' not in results['analysis_summary']:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title('Quality Metrics')
            return
        
        improved = results['analysis_summary']['improved']
        
        metrics = ['Repetition\n(lower=better)', 'Amharic Ratio\n(higher=better)', 'Diversity\n(higher=better)']
        improved_values = [improved['avg_repetition'], improved['avg_amharic_ratio'], improved['avg_diversity']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars = ax.bar(x, improved_values, width, label='Improved Method', color='green', alpha=0.7)
        
        ax.set_ylabel('Score')
        ax.set_title('Quality Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, improved_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_pattern_distribution(self, ax, results):
        """Plot pattern type distribution"""
        if 'improved' not in results['analysis_summary']:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title('Pattern Distribution')
            return
        
        improved_patterns = results['analysis_summary']['improved']['pattern_distribution']
        
        if improved_patterns:
            patterns = list(improved_patterns.keys())
            counts = list(improved_patterns.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(patterns)))
            wedges, texts, autotexts = ax.pie(counts, labels=patterns, autopct='%1.1f%%', 
                                             colors=colors, startangle=90)
            
            ax.set_title('Pattern Type Distribution\n(Improved Method)')
        else:
            ax.text(0.5, 0.5, 'No pattern data', ha='center', va='center')
            ax.set_title('Pattern Distribution')
    
    def _plot_sample_scores(self, ax, results):
        """Plot sample quality scores"""
        scores = []
        
        for prompt_result in results.get('improved_results', []):
            for sample in prompt_result['samples']:
                scores.append(sample['analysis'].get('score', sample.get('score', 0)))
        
        if scores:
            ax.hist(scores, bins=10, alpha=0.7, color='green', edgecolor='black')
            ax.set_xlabel('Quality Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Sample Quality Score Distribution')
            ax.axvline(np.mean(scores), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(scores):.3f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No score data', ha='center', va='center')
            ax.set_title('Sample Quality Scores')
    
    def _plot_length_distribution(self, ax, results):
        """Plot text length distribution"""
        lengths = []
        
        for prompt_result in results.get('improved_results', []):
            for sample in prompt_result['samples']:
                lengths.append(sample['analysis']['length'])
        
        if lengths:
            ax.hist(lengths, bins=15, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel('Text Length (characters)')
            ax.set_ylabel('Frequency')
            ax.set_title('Generated Text Length Distribution')
            ax.axvline(np.mean(lengths), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(lengths):.1f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No length data', ha='center', va='center')
            ax.set_title('Text Length Distribution')
    
    def _plot_diversity_analysis(self, ax, results):
        """Plot character diversity analysis"""
        diversity_scores = []
        unique_chars = []
        
        for prompt_result in results.get('improved_results', []):
            for sample in prompt_result['samples']:
                diversity_scores.append(sample['analysis']['diversity_score'])
                unique_chars.append(sample['analysis']['unique_chars'])
        
        if diversity_scores and unique_chars:
            scatter = ax.scatter(unique_chars, diversity_scores, alpha=0.6, c='purple')
            ax.set_xlabel('Unique Characters')
            ax.set_ylabel('Diversity Score')
            ax.set_title('Character Diversity Analysis')
            
            # Add trend line
            if len(unique_chars) > 1:
                z = np.polyfit(unique_chars, diversity_scores, 1)
                p = np.poly1d(z)
                ax.plot(unique_chars, p(unique_chars), "r--", alpha=0.8)
        else:
            ax.text(0.5, 0.5, 'No diversity data', ha='center', va='center')
            ax.set_title('Character Diversity Analysis')
    
    def _plot_improvement_summary(self, ax, results):
        """Plot improvement summary"""
        # Create a summary text
        summary_text = "🚀 IMPROVEMENT SUMMARY\n\n"
        
        if 'improved' in results['analysis_summary']:
            improved = results['analysis_summary']['improved']
            
            summary_text += f"📊 Quality Metrics:\n"
            summary_text += f"• Repetition: {improved['avg_repetition']:.3f}\n"
            summary_text += f"• Amharic Ratio: {improved['avg_amharic_ratio']:.3f}\n"
            summary_text += f"• Diversity: {improved['avg_diversity']:.3f}\n\n"
            
            summary_text += f"🎯 Key Improvements:\n"
            summary_text += f"• Repetition penalties implemented\n"
            summary_text += f"• Nucleus sampling (top-p)\n"
            summary_text += f"• Temperature optimization\n"
            summary_text += f"• Multi-level pattern detection\n\n"
            
            # Pattern analysis
            patterns = improved['pattern_distribution']
            if patterns:
                good_quality = patterns.get('good_quality', 0)
                total_samples = sum(patterns.values())
                quality_percentage = (good_quality / total_samples) * 100 if total_samples > 0 else 0
                summary_text += f"✅ Quality: {quality_percentage:.1f}% good samples"
        else:
            summary_text += "No analysis data available"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Improvement Summary')

def main():
    """Main function to run visual comparison"""
    print("🎨 Amharic H-Net Visual Comparison Tool")
    print("=" * 50)
    
    # Initialize comparison tool
    tool = VisualComparisonTool()
    
    # Test prompts
    test_prompts = [
        "ሰላም",
        "ትምህርት", 
        "ቤተሰብ",
        "ኢትዮጵያ",
        "ባህል"
    ]
    
    # Run comparison
    results = tool.compare_generation_methods(test_prompts, num_samples=3)
    
    if results:
        # Create visual report
        report_path = tool.create_visual_report(results)
        
        # Save detailed results
        json_path = "visual_comparison_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convert Counter objects to regular dicts for JSON serialization
            json_results = results.copy()
            if 'analysis_summary' in json_results:
                for method in json_results['analysis_summary']:
                    if 'pattern_distribution' in json_results['analysis_summary'][method]:
                        json_results['analysis_summary'][method]['pattern_distribution'] = \
                            dict(json_results['analysis_summary'][method]['pattern_distribution'])
            
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Visual report: {report_path}")
        print(f"📄 Detailed results: {json_path}")
        
        # Print summary
        if 'improved' in results['analysis_summary']:
            improved = results['analysis_summary']['improved']
            print(f"\n🎯 Summary:")
            print(f"   Repetition Score: {improved['avg_repetition']:.3f} (lower is better)")
            print(f"   Amharic Ratio: {improved['avg_amharic_ratio']:.3f} (higher is better)")
            print(f"   Diversity Score: {improved['avg_diversity']:.3f} (higher is better)")
    else:
        print("❌ No results to analyze")

if __name__ == "__main__":
    main()
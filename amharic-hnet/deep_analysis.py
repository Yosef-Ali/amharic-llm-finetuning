#!/usr/bin/env python3
"""
Deep Analysis Tool for Amharic H-Net Model
Focuses on identifying generation issues and providing actionable insights
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import json
import time
import sys
import os
from typing import List, Dict, Tuple
import re

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_train import EnhancedHNet, EnhancedAmharicTokenizer

class DeepModelAnalyzer:
    def __init__(self):
        """Initialize the deep analyzer"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.load_model()
        
    def load_model(self):
        """Load the trained model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = EnhancedAmharicTokenizer()
            self.tokenizer.load("models/enhanced_tokenizer.pkl")
            
            # Load model
            self.model = EnhancedHNet(
                vocab_size=self.tokenizer.vocab_size,
                embedding_dim=256,
                hidden_dim=512,
                num_layers=3,
                dropout=0.2
            )
            
            checkpoint = torch.load("models/enhanced_hnet/best_model.pt", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("📝 Note: Make sure training has completed and model files exist")
            return False
    
    def analyze_repetition_issues(self):
        """Deep analysis of repetition patterns"""
        print("\n" + "="*60)
        print("🔍 DEEP REPETITION ANALYSIS")
        print("="*60)
        
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded. Cannot perform analysis.")
            return
        
        test_prompts = [
            "ሰላም", "ትምህርት", "ቤተሰብ", "ኢትዮጵያ", "ባህል", 
            "ታሪክ", "ሳይንስ", "ጤና", "ፍቅር", "አዲስ"
        ]
        
        repetition_analysis = {
            'severe_repetition': 0,
            'moderate_repetition': 0,
            'good_generation': 0,
            'pattern_types': Counter(),
            'problematic_sequences': [],
            'temperature_effects': {}
        }
        
        for temp in [0.3, 0.7, 1.0, 1.5]:
            temp_results = []
            print(f"\n🌡️ Testing at temperature {temp}:")
            
            for prompt in test_prompts:
                try:
                    generated = self.model.generate(
                        self.tokenizer,
                        prompt=prompt,
                        max_length=100,
                        temperature=temp,
                        device=self.device
                    )
                    
                    # Analyze this generation
                    analysis = self._analyze_single_text(generated, prompt)
                    temp_results.append(analysis)
                    
                    # Categorize repetition level
                    if analysis['repetition_score'] > 0.8:
                        repetition_analysis['severe_repetition'] += 1
                        repetition_analysis['problematic_sequences'].append({
                            'prompt': prompt,
                            'temperature': temp,
                            'generated': generated[:100],
                            'issue': 'severe_repetition'
                        })
                    elif analysis['repetition_score'] > 0.5:
                        repetition_analysis['moderate_repetition'] += 1
                    else:
                        repetition_analysis['good_generation'] += 1
                    
                    # Track pattern types
                    repetition_analysis['pattern_types'][analysis['pattern_type']] += 1
                    
                    print(f"   '{prompt}' → {analysis['summary']}")
                    
                except Exception as e:
                    print(f"   ❌ Failed for '{prompt}': {e}")
            
            # Store temperature effects
            repetition_analysis['temperature_effects'][temp] = {
                'avg_repetition': np.mean([r['repetition_score'] for r in temp_results]),
                'avg_amharic_ratio': np.mean([r['amharic_ratio'] for r in temp_results]),
                'avg_diversity': np.mean([r['diversity_score'] for r in temp_results])
            }
        
        self._display_repetition_results(repetition_analysis)
        return repetition_analysis
    
    def _analyze_single_text(self, text: str, prompt: str) -> Dict:
        """Analyze a single generated text for various issues"""
        analysis = {
            'length': len(text),
            'prompt': prompt,
            'generated': text
        }
        
        # 1. Repetition Analysis
        words = text.split()
        if len(words) > 1:
            unique_words = len(set(words))
            repetition_score = 1 - (unique_words / len(words))
        else:
            repetition_score = 0
        
        # 2. Character-level repetition
        char_repetition = self._calculate_char_repetition(text)
        
        # 3. Amharic character ratio
        amharic_chars = sum(1 for c in text if '\u1200' <= c <= '\u137F')
        total_chars = len(text) if len(text) > 0 else 1
        amharic_ratio = amharic_chars / total_chars
        
        # 4. Diversity score (unique n-grams)
        diversity_score = self._calculate_diversity(text)
        
        # 5. Pattern type identification
        pattern_type = self._identify_pattern_type(text)
        
        analysis.update({
            'repetition_score': max(repetition_score, char_repetition),
            'amharic_ratio': amharic_ratio,
            'diversity_score': diversity_score,
            'pattern_type': pattern_type,
            'summary': self._generate_summary(repetition_score, amharic_ratio, diversity_score, pattern_type)
        })
        
        return analysis
    
    def _calculate_char_repetition(self, text: str) -> float:
        """Calculate character-level repetition"""
        if len(text) < 3:
            return 0
        
        # Look for repeated sequences
        max_repetition = 0
        for length in range(1, min(10, len(text) // 2)):
            for i in range(len(text) - length):
                pattern = text[i:i+length]
                count = 0
                pos = i
                while pos < len(text) - length and text[pos:pos+length] == pattern:
                    count += 1
                    pos += length
                
                if count > 1:
                    repetition_ratio = (count * length) / len(text)
                    max_repetition = max(max_repetition, repetition_ratio)
        
        return max_repetition
    
    def _calculate_diversity(self, text: str) -> float:
        """Calculate text diversity using n-gram uniqueness"""
        if len(text) < 3:
            return 0
        
        # Calculate 3-gram diversity
        trigrams = [text[i:i+3] for i in range(len(text)-2)]
        if len(trigrams) == 0:
            return 0
        
        unique_trigrams = len(set(trigrams))
        return unique_trigrams / len(trigrams)
    
    def _identify_pattern_type(self, text: str) -> str:
        """Identify the type of repetition pattern"""
        if len(text) < 5:
            return "too_short"
        
        # Check for single character repetition
        if len(set(text)) <= 3:
            return "single_char_repetition"
        
        # Check for word repetition
        words = text.split()
        if len(words) > 1 and len(set(words)) <= len(words) * 0.3:
            return "word_repetition"
        
        # Check for sequence repetition
        for length in range(2, min(20, len(text) // 3)):
            pattern = text[:length]
            if text.count(pattern) >= 3:
                return "sequence_repetition"
        
        # Check for alternating patterns
        if self._has_alternating_pattern(text):
            return "alternating_pattern"
        
        return "normal"
    
    def _has_alternating_pattern(self, text: str) -> bool:
        """Check for alternating character patterns"""
        if len(text) < 6:
            return False
        
        # Check for ABABAB pattern
        for i in range(len(text) - 5):
            if text[i] == text[i+2] == text[i+4] and text[i+1] == text[i+3] == text[i+5]:
                return True
        
        return False
    
    def _generate_summary(self, repetition: float, amharic: float, diversity: float, pattern: str) -> str:
        """Generate a summary of the analysis"""
        if repetition > 0.8:
            quality = "❌ Severe repetition"
        elif repetition > 0.5:
            quality = "⚠️ Moderate repetition"
        elif diversity < 0.3:
            quality = "⚠️ Low diversity"
        elif amharic < 0.5:
            quality = "⚠️ Low Amharic ratio"
        else:
            quality = "✅ Good quality"
        
        return f"{quality} (Rep: {repetition:.2f}, Am: {amharic:.2f}, Div: {diversity:.2f}, Type: {pattern})"
    
    def _display_repetition_results(self, analysis: Dict):
        """Display repetition analysis results"""
        total = analysis['severe_repetition'] + analysis['moderate_repetition'] + analysis['good_generation']
        
        print(f"\n📊 REPETITION ANALYSIS SUMMARY:")
        print(f"   • Total generations analyzed: {total}")
        print(f"   • Severe repetition: {analysis['severe_repetition']} ({analysis['severe_repetition']/total*100:.1f}%)")
        print(f"   • Moderate repetition: {analysis['moderate_repetition']} ({analysis['moderate_repetition']/total*100:.1f}%)")
        print(f"   • Good quality: {analysis['good_generation']} ({analysis['good_generation']/total*100:.1f}%)")
        
        print(f"\n🔄 PATTERN TYPES:")
        for pattern, count in analysis['pattern_types'].most_common():
            print(f"   • {pattern}: {count} occurrences")
        
        print(f"\n🌡️ TEMPERATURE EFFECTS:")
        for temp, effects in analysis['temperature_effects'].items():
            print(f"   • T={temp}: Rep={effects['avg_repetition']:.2f}, Am={effects['avg_amharic_ratio']:.2f}, Div={effects['avg_diversity']:.2f}")
        
        if analysis['problematic_sequences']:
            print(f"\n⚠️ MOST PROBLEMATIC EXAMPLES:")
            for i, example in enumerate(analysis['problematic_sequences'][:3]):
                print(f"   {i+1}. Prompt: '{example['prompt']}' (T={example['temperature']})")
                print(f"      Generated: '{example['generated'][:60]}{'...' if len(example['generated']) > 60 else ''}'")
    
    def generate_improvement_recommendations(self, analysis: Dict) -> List[str]:
        """Generate specific recommendations for improving the model"""
        recommendations = []
        
        total = analysis['severe_repetition'] + analysis['moderate_repetition'] + analysis['good_generation']
        severe_rate = analysis['severe_repetition'] / total if total > 0 else 0
        
        # High repetition issues
        if severe_rate > 0.3:
            recommendations.extend([
                "🔧 CRITICAL: Implement repetition penalty in generation",
                "🔧 Add nucleus sampling (top-p) to reduce repetitive outputs",
                "🔧 Consider implementing diverse beam search"
            ])
        
        # Pattern-specific recommendations
        common_patterns = analysis['pattern_types'].most_common(3)
        for pattern, count in common_patterns:
            if pattern == "single_char_repetition" and count > 5:
                recommendations.append("🔧 Add character-level diversity loss during training")
            elif pattern == "sequence_repetition" and count > 5:
                recommendations.append("🔧 Implement sequence-level repetition penalty")
            elif pattern == "word_repetition" and count > 5:
                recommendations.append("🔧 Add word-level repetition penalty")
        
        # Temperature-specific recommendations
        temp_effects = analysis['temperature_effects']
        best_temp = min(temp_effects.keys(), key=lambda t: temp_effects[t]['avg_repetition'])
        recommendations.append(f"🔧 Use temperature {best_temp} for best results")
        
        # Training recommendations
        if severe_rate > 0.5:
            recommendations.extend([
                "📚 Increase training data diversity",
                "📚 Add data augmentation techniques",
                "📚 Implement curriculum learning"
            ])
        
        return recommendations
    
    def create_visual_report(self, analysis: Dict):
        """Create visual charts of the analysis"""
        print("\n📈 Creating visual analysis report...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Amharic H-Net Deep Analysis Report', fontsize=16, fontweight='bold')
        
        # 1. Repetition severity distribution
        categories = ['Severe', 'Moderate', 'Good']
        values = [analysis['severe_repetition'], analysis['moderate_repetition'], analysis['good_generation']]
        colors = ['red', 'orange', 'green']
        
        axes[0, 0].pie(values, labels=categories, colors=colors, autopct='%1.1f%%')
        axes[0, 0].set_title('Generation Quality Distribution')
        
        # 2. Pattern types
        patterns = list(analysis['pattern_types'].keys())
        pattern_counts = list(analysis['pattern_types'].values())
        
        axes[0, 1].bar(patterns, pattern_counts, color='skyblue')
        axes[0, 1].set_title('Repetition Pattern Types')
        axes[0, 1].set_xlabel('Pattern Type')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Temperature effects on repetition
        temps = list(analysis['temperature_effects'].keys())
        rep_scores = [analysis['temperature_effects'][t]['avg_repetition'] for t in temps]
        
        axes[1, 0].plot(temps, rep_scores, 'ro-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Temperature vs Repetition Score')
        axes[1, 0].set_xlabel('Temperature')
        axes[1, 0].set_ylabel('Average Repetition Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Multi-metric comparison
        metrics = ['Repetition', 'Amharic Ratio', 'Diversity']
        temp_1_0 = analysis['temperature_effects'].get(1.0, {})
        scores = [
            temp_1_0.get('avg_repetition', 0),
            temp_1_0.get('avg_amharic_ratio', 0),
            temp_1_0.get('avg_diversity', 0)
        ]
        
        bars = axes[1, 1].bar(metrics, scores, color=['red', 'blue', 'green'], alpha=0.7)
        axes[1, 1].set_title('Model Performance Metrics (T=1.0)')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('deep_analysis_report.png', dpi=300, bbox_inches='tight')
        print("💾 Visual report saved as 'deep_analysis_report.png'")
        plt.close()
    
    def run_complete_analysis(self):
        """Run the complete deep analysis"""
        print("🚀 Starting Deep Model Analysis...")
        print("This will analyze generation patterns, repetition issues, and provide recommendations.")
        
        if not self.model or not self.tokenizer:
            print("❌ Cannot run analysis - model not loaded")
            return
        
        # 1. Analyze repetition issues
        analysis = self.analyze_repetition_issues()
        
        # 2. Generate recommendations
        recommendations = self.generate_improvement_recommendations(analysis)
        
        # 3. Create visual report
        self.create_visual_report(analysis)
        
        # 4. Save detailed report
        report = {
            'analysis_results': analysis,
            'recommendations': recommendations,
            'model_info': {
                'vocab_size': self.tokenizer.vocab_size if self.tokenizer else 0,
                'device': str(self.device),
                'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('deep_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 5. Display recommendations
        print("\n" + "="*60)
        print("💡 IMPROVEMENT RECOMMENDATIONS")
        print("="*60)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec}")
        
        print("\n" + "="*60)
        print("✅ DEEP ANALYSIS COMPLETE!")
        print("="*60)
        print("📁 Files created:")
        print("   • deep_analysis_report.png")
        print("   • deep_analysis_report.json")
        print("\n🔍 Check these files for detailed insights and visual analysis!")

def main():
    """Main function"""
    analyzer = DeepModelAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
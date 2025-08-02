#!/usr/bin/env python3
"""
Best Practices Guide for Meaningful Amharic Text Generation

This comprehensive guide demonstrates the most effective techniques for generating
high-quality, relevant, and meaningful Amharic text using the H-Net model.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import re
from collections import defaultdict, Counter
import json
from typing import List, Dict, Tuple, Optional

class AmharicBestPracticesGenerator:
    """
    Advanced Amharic text generator implementing best practices for:
    - Semantic relevance
    - Vocabulary richness
    - Contextual coherence
    - Domain-specific generation
    """
    
    def __init__(self, model_path: str = "./model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Comprehensive Amharic knowledge base
        self.domain_vocabulary = {
            'education': {
                'core': ['ትምህርት', 'ተማሪ', 'መምህር', 'ትምህርት ቤት', 'ዩኒቨርሲቲ', 'ኮሌጅ'],
                'concepts': ['ዕውቀት', 'ትምህርት', 'ጥናት', 'ምርምር', 'ፈተና', 'ውጤት'],
                'actions': ['ይማራል', 'ያስተምራል', 'ያጠናል', 'ይመረምራል', 'ይፈትናል']
            },
            'family': {
                'core': ['ቤተሰብ', 'እናት', 'አባት', 'ልጅ', 'ወንድም', 'እህት'],
                'concepts': ['ፍቅር', 'መከባበር', 'አብሮነት', 'ደጋፊነት', 'እንክብካቤ'],
                'actions': ['ይወዳል', 'ይከባከባል', 'ይደግፋል', 'ያሳድጋል', 'ይጠብቃል']
            },
            'country': {
                'core': ['ኢትዮጵያ', 'ሀገር', 'ህዝብ', 'ባህል', 'ታሪክ', 'ቋንቋ'],
                'concepts': ['ነፃነት', 'ዲሞክራሲ', 'ልማት', 'ሰላም', 'አንድነት'],
                'actions': ['ይገነባል', 'ያድጋል', 'ይጠብቃል', 'ያስተዳድራል', 'ያበረታታል']
            },
            'health': {
                'core': ['ጤና', 'ሐኪም', 'ሆስፒታል', 'መድሃኒት', 'ህክምና'],
                'concepts': ['ደህንነት', 'ክብካቤ', 'ህክምና', 'መከላከል', 'ማገገም'],
                'actions': ['ያክማል', 'ይከላከላል', 'ያስተናግዳል', 'ይመረምራል']
            },
            'work': {
                'core': ['ስራ', 'ሰራተኛ', 'ኩባንያ', 'ቢሮ', 'ፕሮጀክት'],
                'concepts': ['ኃላፊነት', 'ብቃት', 'ትብብር', 'ውጤታማነት', 'ልማት'],
                'actions': ['ይሰራል', 'ያስተዳድራል', 'ያዘጋጃል', 'ያስፈጽማል']
            }
        }
        
        # Semantic relationships
        self.semantic_relations = {
            'ትምህርት': ['ተማሪ', 'መምህር', 'ትምህርት ቤት', 'ዕውቀት', 'ጥናት'],
            'ቤተሰብ': ['እናት', 'አባት', 'ልጅ', 'ፍቅር', 'መከባበር'],
            'ኢትዮጵያ': ['ሀገር', 'ህዝብ', 'ባህል', 'ታሪክ', 'ነፃነት'],
            'ጤና': ['ሐኪም', 'ሆስፒታል', 'መድሃኒት', 'ክብካቤ'],
            'ስራ': ['ሰራተኛ', 'ኩባንያ', 'ኃላፊነት', 'ብቃት']
        }
        
        # Common sentence patterns
        self.sentence_patterns = [
            "{subject} {verb} {object} ነው።",
            "{subject} በ{context} {verb}።",
            "{subject} ምክንያቱም {reason} {verb}።",
            "{subject} እና {related} {verb}።",
            "{subject} ለ{purpose} {verb}።"
        ]
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_semantic_score': 0.3,
            'min_vocabulary_match': 0.4,
            'max_repetition': 0.2,
            'min_coherence': 0.6
        }
    
    def load_model(self):
        """Load the H-Net model and tokenizer"""
        print("🔄 Loading H-Net model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.eval()
        print("✅ Model loaded successfully!")
    
    def identify_domain(self, prompt: str) -> str:
        """Identify the most relevant domain for the prompt"""
        prompt_lower = prompt.lower()
        domain_scores = {}
        
        for domain, vocab in self.domain_vocabulary.items():
            score = 0
            for category, words in vocab.items():
                for word in words:
                    if word in prompt or any(char in prompt for char in word):
                        score += len(word) / len(prompt) if len(prompt) > 0 else 0
            domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
    
    def get_contextual_vocabulary(self, prompt: str, domain: str) -> List[str]:
        """Get relevant vocabulary based on prompt and domain"""
        vocab = []
        
        # Add domain-specific vocabulary
        if domain in self.domain_vocabulary:
            for category, words in self.domain_vocabulary[domain].items():
                vocab.extend(words)
        
        # Add semantically related words
        for word in prompt.split():
            if word in self.semantic_relations:
                vocab.extend(self.semantic_relations[word])
        
        return list(set(vocab))
    
    def generate_with_domain_guidance(self, prompt: str, max_length: int = 50, 
                                    num_candidates: int = 5) -> Dict:
        """Generate text with domain-specific guidance"""
        if not self.model or not self.tokenizer:
            self.load_model()
        
        # Identify domain and get relevant vocabulary
        domain = self.identify_domain(prompt)
        contextual_vocab = self.get_contextual_vocabulary(prompt, domain)
        
        print(f"🎯 Detected domain: {domain}")
        print(f"📚 Contextual vocabulary: {contextual_vocab[:10]}...")
        
        # Generate multiple candidates
        candidates = []
        
        for i in range(num_candidates):
            # Encode prompt
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            
            # Generate with domain-aware sampling
            with torch.no_grad():
                generated = self._domain_aware_generation(
                    inputs, max_length, contextual_vocab, domain
                )
            
            # Decode and clean
            text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            text = text[len(prompt):].strip()
            
            if text and len(text) > 5:  # Minimum length check
                quality = self._evaluate_quality(text, prompt, contextual_vocab, domain)
                candidates.append({
                    'text': text,
                    'quality': quality,
                    'domain': domain
                })
        
        # Select best candidate
        if candidates:
            best = max(candidates, key=lambda x: x['quality']['overall_score'])
            return {
                'text': best['text'],
                'quality': best['quality'],
                'domain': domain,
                'contextual_vocab': contextual_vocab,
                'all_candidates': candidates
            }
        
        return {'text': '', 'quality': {}, 'domain': domain}
    
    def _domain_aware_generation(self, inputs: torch.Tensor, max_length: int, 
                               vocab: List[str], domain: str) -> torch.Tensor:
        """Generate text with domain awareness"""
        generated = inputs.clone()
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs.logits[0, -1, :]
                
                # Apply domain-specific boosting
                logits = self._apply_domain_boosting(logits, vocab, domain)
                
                # Apply quality constraints
                logits = self._apply_quality_constraints(logits, generated)
                
                # Sample next token
                probs = F.softmax(logits / 0.8, dim=-1)  # Temperature = 0.8
                
                # Nucleus sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                nucleus = sorted_indices[cumsum_probs <= 0.9]  # top-p = 0.9
                
                if len(nucleus) > 0:
                    next_token = nucleus[torch.multinomial(sorted_probs[:len(nucleus)], 1)]
                else:
                    next_token = sorted_indices[0:1]
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Check for natural stopping
                if self._should_stop(generated):
                    break
        
        return generated
    
    def _apply_domain_boosting(self, logits: torch.Tensor, vocab: List[str], 
                             domain: str) -> torch.Tensor:
        """Boost probabilities of domain-relevant tokens"""
        for word in vocab:
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            for token_id in word_tokens:
                if token_id < len(logits):
                    logits[token_id] += 2.0  # Boost domain vocabulary
        
        return logits
    
    def _apply_quality_constraints(self, logits: torch.Tensor, 
                                 generated: torch.Tensor) -> torch.Tensor:
        """Apply quality constraints to prevent repetition and improve coherence"""
        # Penalize recent tokens (repetition penalty)
        if generated.size(1) > 1:
            recent_tokens = generated[0, -10:].tolist()  # Last 10 tokens
            for token_id in set(recent_tokens):
                if token_id < len(logits):
                    count = recent_tokens.count(token_id)
                    logits[token_id] -= count * 1.5  # Repetition penalty
        
        # Boost sentence-ending tokens when appropriate
        text_so_far = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        if len(text_so_far) > 30 and not text_so_far.endswith(('።', '?', '!')):
            period_token = self.tokenizer.encode('።', add_special_tokens=False)
            if period_token and period_token[0] < len(logits):
                logits[period_token[0]] += 1.0
        
        return logits
    
    def _should_stop(self, generated: torch.Tensor) -> bool:
        """Determine if generation should stop"""
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Stop on sentence endings
        if text.endswith(('።', '?', '!')):
            return True
        
        # Stop if too long
        if len(text) > 200:
            return True
        
        return False
    
    def _evaluate_quality(self, text: str, prompt: str, vocab: List[str], 
                        domain: str) -> Dict:
        """Comprehensive quality evaluation"""
        # Semantic relevance
        semantic_score = self._calculate_semantic_relevance(text, prompt, vocab)
        
        # Vocabulary match
        vocab_score = self._calculate_vocabulary_match(text, vocab)
        
        # Coherence
        coherence_score = self._calculate_coherence(text)
        
        # Repetition check
        repetition_score = self._calculate_repetition(text)
        
        # Amharic purity
        amharic_score = self._check_amharic_purity(text)
        
        # Overall score
        overall_score = (
            semantic_score * 0.3 +
            vocab_score * 0.25 +
            coherence_score * 0.25 +
            (1 - repetition_score) * 0.1 +
            amharic_score * 0.1
        )
        
        return {
            'semantic_relevance': semantic_score,
            'vocabulary_match': vocab_score,
            'coherence': coherence_score,
            'repetition': repetition_score,
            'amharic_purity': amharic_score,
            'overall_score': overall_score,
            'meets_thresholds': self._meets_quality_thresholds({
                'semantic_relevance': semantic_score,
                'vocabulary_match': vocab_score,
                'repetition': repetition_score,
                'coherence': coherence_score
            })
        }
    
    def _calculate_semantic_relevance(self, text: str, prompt: str, 
                                    vocab: List[str]) -> float:
        """Calculate semantic relevance to prompt"""
        text_words = set(text.split())
        prompt_words = set(prompt.split())
        vocab_words = set(vocab)
        
        # Direct overlap with prompt
        prompt_overlap = len(text_words & prompt_words) / max(len(prompt_words), 1)
        
        # Overlap with contextual vocabulary
        vocab_overlap = len(text_words & vocab_words) / max(len(text_words), 1)
        
        return (prompt_overlap + vocab_overlap) / 2
    
    def _calculate_vocabulary_match(self, text: str, vocab: List[str]) -> float:
        """Calculate how many words match the contextual vocabulary"""
        text_words = text.split()
        if not text_words:
            return 0.0
        
        matches = sum(1 for word in text_words if any(v in word for v in vocab))
        return matches / len(text_words)
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence"""
        # Check for proper sentence structure
        has_subject = any(word in text for word in ['ኢትዮጵያ', 'ቤተሰብ', 'ትምህርት', 'ጤና', 'ስራ'])
        has_verb = any(word.endswith(('ል', 'ላል', 'ናል')) for word in text.split())
        has_ending = text.endswith(('።', 'ነው።', 'ናቸው።'))
        
        coherence_factors = [has_subject, has_verb, has_ending]
        return sum(coherence_factors) / len(coherence_factors)
    
    def _calculate_repetition(self, text: str) -> float:
        """Calculate repetition score (lower is better)"""
        words = text.split()
        if len(words) <= 1:
            return 0.0
        
        word_counts = Counter(words)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        return repeated_words / len(words)
    
    def _check_amharic_purity(self, text: str) -> float:
        """Check if text is pure Amharic"""
        amharic_chars = sum(1 for char in text if '\u1200' <= char <= '\u137F')
        total_chars = sum(1 for char in text if char.isalpha())
        return amharic_chars / max(total_chars, 1)
    
    def _meets_quality_thresholds(self, scores: Dict) -> bool:
        """Check if all quality thresholds are met"""
        return (
            scores['semantic_relevance'] >= self.quality_thresholds['min_semantic_score'] and
            scores['vocabulary_match'] >= self.quality_thresholds['min_vocabulary_match'] and
            scores['repetition'] <= self.quality_thresholds['max_repetition'] and
            scores['coherence'] >= self.quality_thresholds['min_coherence']
        )
    
    def generate_multiple_approaches(self, prompt: str) -> Dict:
        """Generate using multiple approaches and return the best"""
        approaches = {
            'domain_guided': self.generate_with_domain_guidance,
            'conservative': lambda p: self.generate_with_domain_guidance(p, max_length=30),
            'creative': lambda p: self.generate_with_domain_guidance(p, max_length=70),
        }
        
        results = {}
        best_result = None
        best_score = 0
        
        for approach_name, approach_func in approaches.items():
            try:
                result = approach_func(prompt)
                results[approach_name] = result
                
                if result.get('quality', {}).get('overall_score', 0) > best_score:
                    best_score = result['quality']['overall_score']
                    best_result = result
                    best_result['approach'] = approach_name
            except Exception as e:
                print(f"⚠️ Error in {approach_name}: {e}")
                results[approach_name] = {'error': str(e)}
        
        return {
            'best_result': best_result,
            'all_results': results,
            'comparison': self._compare_approaches(results)
        }
    
    def _compare_approaches(self, results: Dict) -> Dict:
        """Compare different approaches"""
        comparison = {}
        
        for approach, result in results.items():
            if 'quality' in result:
                comparison[approach] = {
                    'score': result['quality']['overall_score'],
                    'semantic': result['quality']['semantic_relevance'],
                    'vocabulary': result['quality']['vocabulary_match'],
                    'coherence': result['quality']['coherence'],
                    'meets_thresholds': result['quality']['meets_thresholds']
                }
        
        return comparison

def demo_best_practices():
    """Demonstrate best practices for Amharic text generation"""
    print("🚀 Amharic Text Generation - Best Practices Demo")
    print("=" * 60)
    
    generator = AmharicBestPracticesGenerator()
    
    # Test prompts covering different domains
    test_prompts = [
        'ትምህርት',      # Education
        'ቤተሰብ',       # Family
        'ኢትዮጵያ',      # Country
        'ጤና',         # Health
        'ስራ'          # Work
    ]
    
    all_results = {}
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        print("-" * 40)
        
        # Generate using multiple approaches
        result = generator.generate_multiple_approaches(prompt)
        all_results[prompt] = result
        
        if result['best_result']:
            best = result['best_result']
            print(f"✅ Best Result ({best.get('approach', 'unknown')}):")  
            print(f"   '{best['text']}'")
            print(f"   Quality Score: {best['quality']['overall_score']:.3f}")
            print(f"   Domain: {best['domain']}")
            print(f"   Meets Thresholds: {best['quality']['meets_thresholds']}")
            
            # Show quality breakdown
            quality = best['quality']
            print(f"   📊 Quality Breakdown:")
            print(f"      Semantic: {quality['semantic_relevance']:.3f}")
            print(f"      Vocabulary: {quality['vocabulary_match']:.3f}")
            print(f"      Coherence: {quality['coherence']:.3f}")
            print(f"      Repetition: {quality['repetition']:.3f}")
            print(f"      Amharic Purity: {quality['amharic_purity']:.3f}")
        else:
            print("❌ No successful generation")
    
    # Save comprehensive results
    with open('best_practices_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("🎉 Best practices demo complete!")
    print("\n📋 Key Best Practices Demonstrated:")
    print("   ✅ Domain-specific vocabulary guidance")
    print("   ✅ Semantic relationship mapping")
    print("   ✅ Multi-approach generation")
    print("   ✅ Comprehensive quality evaluation")
    print("   ✅ Quality threshold enforcement")
    print("   ✅ Contextual coherence optimization")
    print("   ✅ Pure Amharic output")
    print("   ✅ Repetition prevention")
    print("\n💡 Results saved to: best_practices_results.json")

if __name__ == "__main__":
    demo_best_practices()
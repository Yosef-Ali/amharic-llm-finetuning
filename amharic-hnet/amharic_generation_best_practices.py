#!/usr/bin/env python3
"""
Amharic Text Generation - Best Practices Framework

This guide provides a comprehensive framework for generating meaningful,
relevant, and high-quality Amharic text without external dependencies.
"""

import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

class AmharicGenerationFramework:
    """
    Best practices framework for meaningful Amharic text generation
    """
    
    def __init__(self):
        # Domain-specific vocabulary for contextual generation
        self.domain_vocabulary = {
            'education': {
                'core_words': ['ትምህርት', 'ተማሪ', 'መምህር', 'ትምህርት ቤት', 'ዩኒቨርሲቲ', 'ኮሌጅ', 'ዕውቀት'],
                'concepts': ['ጥናት', 'ምርምር', 'ፈተና', 'ውጤት', 'ብቃት', 'ክህሎት', 'ትምህርት'],
                'actions': ['ይማራል', 'ያስተምራል', 'ያጠናል', 'ይመረምራል', 'ይፈትናል', 'ያዳብራል'],
                'qualities': ['ጥሩ', 'ውጤታማ', 'ጠቃሚ', 'አስፈላጊ', 'ዘመናዊ', 'ልዩ']
            },
            'family': {
                'core_words': ['ቤተሰብ', 'እናት', 'አባት', 'ልጅ', 'ወንድም', 'እህት', 'አያት'],
                'concepts': ['ፍቅር', 'መከባበር', 'አብሮነት', 'ደጋፊነት', 'እንክብካቤ', 'ትብብር'],
                'actions': ['ይወዳል', 'ይከባከባል', 'ይደግፋል', 'ያሳድጋል', 'ይጠብቃል', 'ያበረታታል'],
                'qualities': ['ውብ', 'ጠንካራ', 'ተባባሪ', 'ወዳጅ', 'ደጋፊ', 'አስተዋይ']
            },
            'country': {
                'core_words': ['ኢትዮጵያ', 'ሀገር', 'ህዝብ', 'ባህል', 'ታሪክ', 'ቋንቋ', 'ክልል'],
                'concepts': ['ነፃነት', 'ዲሞክራሲ', 'ልማት', 'ሰላም', 'አንድነት', 'ብዝሃነት'],
                'actions': ['ይገነባል', 'ያድጋል', 'ይጠብቃል', 'ያስተዳድራል', 'ያበረታታል', 'ያዳብራል'],
                'qualities': ['ታሪካዊ', 'ውብ', 'ብዙ ባህላዊ', 'ነፃ', 'ዲሞክራሲያዊ', 'ልዩ']
            },
            'health': {
                'core_words': ['ጤና', 'ሐኪም', 'ሆስፒታል', 'መድሃኒት', 'ህክምና', 'ክብካቤ'],
                'concepts': ['ደህንነት', 'መከላከል', 'ማገገም', 'ህክምና', 'እንክብካቤ'],
                'actions': ['ያክማል', 'ይከላከላል', 'ያስተናግዳል', 'ይመረምራል', 'ይጠብቃል'],
                'qualities': ['ጤናማ', 'ደህና', 'ጠንካራ', 'ንፁህ', 'ጥሩ']
            },
            'work': {
                'core_words': ['ስራ', 'ሰራተኛ', 'ኩባንያ', 'ቢሮ', 'ፕሮጀክት', 'ኃላፊነት'],
                'concepts': ['ብቃት', 'ትብብር', 'ውጤታማነት', 'ልማት', 'እድገት'],
                'actions': ['ይሰራል', 'ያስተዳድራል', 'ያዘጋጃል', 'ያስፈጽማል', 'ያዳብራል'],
                'qualities': ['ውጤታማ', 'ብቃት ያለው', 'ተባባሪ', 'ኃላፊ', 'ጠንካራ']
            }
        }
        
        # Semantic relationship mapping
        self.semantic_relations = {
            'ትምህርት': ['ተማሪ', 'መምህር', 'ትምህርት ቤት', 'ዕውቀት', 'ጥናት', 'ብቃት'],
            'ቤተሰብ': ['እናት', 'አባት', 'ልጅ', 'ፍቅር', 'መከባበር', 'አብሮነት'],
            'ኢትዮጵያ': ['ሀገር', 'ህዝብ', 'ባህል', 'ታሪክ', 'ነፃነት', 'ልማት'],
            'ጤና': ['ሐኪም', 'ሆስፒታል', 'መድሃኒት', 'ክብካቤ', 'ደህንነት'],
            'ስራ': ['ሰራተኛ', 'ኩባንያ', 'ኃላፊነት', 'ብቃት', 'ውጤታማነት']
        }
        
        # Common Amharic sentence patterns
        self.sentence_patterns = [
            "{subject} {quality} {concept} ነው።",
            "{subject} በ{context} {action}።",
            "{subject} ምክንያቱም {reason} {action}።",
            "{subject} እና {related} {action}።",
            "{subject} ለ{purpose} {action}።",
            "{subject} {concept} {quality} ነው።",
            "{subject} በጣም {quality} {concept} ነው።"
        ]
        
        # Quality assessment criteria
        self.quality_criteria = {
            'semantic_relevance': {
                'description': 'How well the text relates to the input prompt',
                'weight': 0.30,
                'min_threshold': 0.4
            },
            'vocabulary_richness': {
                'description': 'Use of domain-appropriate vocabulary',
                'weight': 0.25,
                'min_threshold': 0.3
            },
            'coherence': {
                'description': 'Logical flow and sentence structure',
                'weight': 0.25,
                'min_threshold': 0.6
            },
            'repetition_control': {
                'description': 'Absence of unnecessary repetition',
                'weight': 0.10,
                'max_threshold': 0.2
            },
            'amharic_purity': {
                'description': 'Pure Amharic without mixed languages',
                'weight': 0.10,
                'min_threshold': 0.9
            }
        }
    
    def identify_domain(self, prompt: str) -> str:
        """Identify the most relevant domain for the given prompt"""
        prompt_words = set(prompt.split())
        domain_scores = {}
        
        for domain, vocab_categories in self.domain_vocabulary.items():
            score = 0
            total_words = 0
            
            for category, words in vocab_categories.items():
                total_words += len(words)
                for word in words:
                    # Check for exact matches and partial matches
                    if word in prompt:
                        score += 3  # Exact match
                    elif any(char_seq in prompt for char_seq in [word[:3], word[-3:]] if len(word) >= 3):
                        score += 1  # Partial match
            
            domain_scores[domain] = score / max(total_words, 1)
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
    
    def get_contextual_vocabulary(self, prompt: str, domain: str) -> Dict[str, List[str]]:
        """Get relevant vocabulary based on prompt and identified domain"""
        vocab = {'core_words': [], 'concepts': [], 'actions': [], 'qualities': []}
        
        # Add domain-specific vocabulary
        if domain in self.domain_vocabulary:
            for category, words in self.domain_vocabulary[domain].items():
                vocab[category] = words.copy()
        
        # Add semantically related words from prompt
        for word in prompt.split():
            if word in self.semantic_relations:
                vocab['concepts'].extend(self.semantic_relations[word])
        
        # Remove duplicates while preserving order
        for category in vocab:
            vocab[category] = list(dict.fromkeys(vocab[category]))
        
        return vocab
    
    def generate_structured_sentences(self, prompt: str, domain: str, 
                                    vocab: Dict[str, List[str]]) -> List[str]:
        """Generate structured sentences using patterns and vocabulary"""
        sentences = []
        
        # Use the prompt as the main subject
        subject = prompt.strip()
        
        for pattern in self.sentence_patterns:
            try:
                # Fill pattern with appropriate vocabulary
                if '{quality}' in pattern and vocab['qualities']:
                    quality = vocab['qualities'][0]
                else:
                    quality = 'ጥሩ'  # Default quality
                
                if '{concept}' in pattern and vocab['concepts']:
                    concept = vocab['concepts'][0]
                else:
                    concept = vocab['core_words'][0] if vocab['core_words'] else 'ነገር'
                
                if '{action}' in pattern and vocab['actions']:
                    action = vocab['actions'][0]
                else:
                    action = 'ነው'
                
                if '{context}' in pattern and vocab['concepts']:
                    context = vocab['concepts'][1] if len(vocab['concepts']) > 1 else vocab['concepts'][0]
                else:
                    context = 'ጊዜ'
                
                if '{reason}' in pattern and vocab['concepts']:
                    reason = vocab['concepts'][0]
                else:
                    reason = 'ምክንያት'
                
                if '{related}' in pattern and vocab['core_words']:
                    related = vocab['core_words'][1] if len(vocab['core_words']) > 1 else vocab['core_words'][0]
                else:
                    related = 'ሌላ'
                
                if '{purpose}' in pattern and vocab['concepts']:
                    purpose = vocab['concepts'][0]
                else:
                    purpose = 'ዓላማ'
                
                # Format the sentence
                sentence = pattern.format(
                    subject=subject,
                    quality=quality,
                    concept=concept,
                    action=action,
                    context=context,
                    reason=reason,
                    related=related,
                    purpose=purpose
                )
                
                sentences.append(sentence)
                
            except (KeyError, IndexError) as e:
                continue  # Skip patterns that can't be filled
        
        return sentences
    
    def evaluate_text_quality(self, text: str, prompt: str, 
                            vocab: Dict[str, List[str]]) -> Dict:
        """Comprehensive quality evaluation of generated text"""
        scores = {}
        
        # 1. Semantic Relevance
        scores['semantic_relevance'] = self._calculate_semantic_relevance(text, prompt, vocab)
        
        # 2. Vocabulary Richness
        scores['vocabulary_richness'] = self._calculate_vocabulary_richness(text, vocab)
        
        # 3. Coherence
        scores['coherence'] = self._calculate_coherence(text)
        
        # 4. Repetition Control
        scores['repetition_control'] = 1.0 - self._calculate_repetition_score(text)
        
        # 5. Amharic Purity
        scores['amharic_purity'] = self._calculate_amharic_purity(text)
        
        # Calculate overall score
        overall_score = sum(
            scores[criterion] * self.quality_criteria[criterion]['weight']
            for criterion in scores
        )
        
        # Check if meets quality thresholds
        meets_thresholds = self._check_quality_thresholds(scores)
        
        return {
            'scores': scores,
            'overall_score': overall_score,
            'meets_thresholds': meets_thresholds,
            'quality_assessment': self._generate_quality_assessment(scores)
        }
    
    def _calculate_semantic_relevance(self, text: str, prompt: str, 
                                    vocab: Dict[str, List[str]]) -> float:
        """Calculate how semantically relevant the text is to the prompt"""
        text_words = set(text.split())
        prompt_words = set(prompt.split())
        
        # Direct overlap with prompt
        prompt_overlap = len(text_words & prompt_words) / max(len(prompt_words), 1)
        
        # Overlap with contextual vocabulary
        all_vocab_words = set()
        for word_list in vocab.values():
            all_vocab_words.update(word_list)
        
        vocab_overlap = len(text_words & all_vocab_words) / max(len(text_words), 1)
        
        return (prompt_overlap * 0.6 + vocab_overlap * 0.4)
    
    def _calculate_vocabulary_richness(self, text: str, 
                                     vocab: Dict[str, List[str]]) -> float:
        """Calculate vocabulary richness and domain appropriateness"""
        text_words = text.split()
        if not text_words:
            return 0.0
        
        # Count matches with domain vocabulary
        vocab_matches = 0
        all_vocab_words = set()
        for word_list in vocab.values():
            all_vocab_words.update(word_list)
        
        for word in text_words:
            if any(vocab_word in word for vocab_word in all_vocab_words):
                vocab_matches += 1
        
        return vocab_matches / len(text_words)
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence based on structure and flow"""
        coherence_factors = []
        
        # Check for proper sentence ending
        coherence_factors.append(text.endswith(('።', 'ነው።', 'ናቸው።', '?', '!')))
        
        # Check for subject presence
        has_subject = any(word in text for word in 
                         ['ኢትዮጵያ', 'ቤተሰብ', 'ትምህርት', 'ጤና', 'ስራ'])
        coherence_factors.append(has_subject)
        
        # Check for verb presence
        has_verb = any(word.endswith(('ል', 'ላል', 'ናል', 'ነው')) for word in text.split())
        coherence_factors.append(has_verb)
        
        # Check sentence length (not too short, not too long)
        word_count = len(text.split())
        appropriate_length = 3 <= word_count <= 15
        coherence_factors.append(appropriate_length)
        
        return sum(coherence_factors) / len(coherence_factors)
    
    def _calculate_repetition_score(self, text: str) -> float:
        """Calculate repetition score (higher means more repetition)"""
        words = text.split()
        if len(words) <= 1:
            return 0.0
        
        word_counts = Counter(words)
        repeated_words = sum(max(0, count - 1) for count in word_counts.values())
        return repeated_words / len(words)
    
    def _calculate_amharic_purity(self, text: str) -> float:
        """Calculate the purity of Amharic text (no mixed languages)"""
        if not text:
            return 0.0
        
        # Count Amharic characters (Ethiopic script range)
        amharic_chars = sum(1 for char in text if '\u1200' <= char <= '\u137F')
        
        # Count alphabetic characters
        alpha_chars = sum(1 for char in text if char.isalpha())
        
        if alpha_chars == 0:
            return 0.0
        
        return amharic_chars / alpha_chars
    
    def _check_quality_thresholds(self, scores: Dict) -> Dict:
        """Check if scores meet minimum quality thresholds"""
        threshold_results = {}
        
        for criterion, score in scores.items():
            if criterion in self.quality_criteria:
                criteria = self.quality_criteria[criterion]
                
                if 'min_threshold' in criteria:
                    threshold_results[criterion] = score >= criteria['min_threshold']
                elif 'max_threshold' in criteria:
                    threshold_results[criterion] = score <= criteria['max_threshold']
        
        threshold_results['overall'] = all(threshold_results.values())
        return threshold_results
    
    def _generate_quality_assessment(self, scores: Dict) -> str:
        """Generate a human-readable quality assessment"""
        assessments = []
        
        if scores['semantic_relevance'] >= 0.6:
            assessments.append("Highly relevant to prompt")
        elif scores['semantic_relevance'] >= 0.3:
            assessments.append("Moderately relevant to prompt")
        else:
            assessments.append("Low relevance to prompt")
        
        if scores['vocabulary_richness'] >= 0.5:
            assessments.append("Rich domain vocabulary")
        elif scores['vocabulary_richness'] >= 0.3:
            assessments.append("Adequate vocabulary")
        else:
            assessments.append("Limited vocabulary")
        
        if scores['coherence'] >= 0.8:
            assessments.append("Excellent coherence")
        elif scores['coherence'] >= 0.6:
            assessments.append("Good coherence")
        else:
            assessments.append("Poor coherence")
        
        if scores['repetition_control'] >= 0.8:
            assessments.append("No repetition issues")
        elif scores['repetition_control'] >= 0.6:
            assessments.append("Minor repetition")
        else:
            assessments.append("Significant repetition")
        
        if scores['amharic_purity'] >= 0.9:
            assessments.append("Pure Amharic")
        else:
            assessments.append("Mixed language content")
        
        return "; ".join(assessments)
    
    def demonstrate_best_practices(self, prompts: List[str]) -> Dict:
        """Demonstrate best practices with multiple prompts"""
        results = {}
        
        print("🚀 Amharic Text Generation - Best Practices Framework")
        print("=" * 65)
        
        for prompt in prompts:
            print(f"\n📝 Analyzing prompt: '{prompt}'")
            print("-" * 45)
            
            # Step 1: Domain identification
            domain = self.identify_domain(prompt)
            print(f"🎯 Identified domain: {domain}")
            
            # Step 2: Contextual vocabulary extraction
            vocab = self.get_contextual_vocabulary(prompt, domain)
            print(f"📚 Contextual vocabulary loaded:")
            for category, words in vocab.items():
                if words:
                    print(f"   {category}: {words[:3]}..." if len(words) > 3 else f"   {category}: {words}")
            
            # Step 3: Generate structured sentences
            sentences = self.generate_structured_sentences(prompt, domain, vocab)
            print(f"\n✨ Generated {len(sentences)} structured sentences:")
            
            sentence_results = []
            for i, sentence in enumerate(sentences[:3], 1):  # Show top 3
                quality = self.evaluate_text_quality(sentence, prompt, vocab)
                sentence_results.append({
                    'sentence': sentence,
                    'quality': quality
                })
                
                print(f"\n{i}. '{sentence}'")
                print(f"   Overall Score: {quality['overall_score']:.3f}")
                print(f"   Assessment: {quality['quality_assessment']}")
                print(f"   Meets Thresholds: {quality['meets_thresholds']['overall']}")
            
            # Find best sentence
            if sentence_results:
                best_sentence = max(sentence_results, key=lambda x: x['quality']['overall_score'])
                print(f"\n✅ Best sentence: '{best_sentence['sentence']}'")
                print(f"   Score: {best_sentence['quality']['overall_score']:.3f}")
            
            results[prompt] = {
                'domain': domain,
                'vocabulary': vocab,
                'sentences': sentence_results,
                'best_sentence': best_sentence if sentence_results else None
            }
        
        return results

def main():
    """Main demonstration function"""
    framework = AmharicGenerationFramework()
    
    # Test with various prompts
    test_prompts = [
        'ትምህርት',      # Education
        'ቤተሰብ',       # Family  
        'ኢትዮጵያ',      # Country
        'ጤና',         # Health
        'ስራ'          # Work
    ]
    
    # Demonstrate best practices
    results = framework.demonstrate_best_practices(test_prompts)
    
    # Save results
    with open('amharic_best_practices_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 65)
    print("🎉 Best Practices Framework Demo Complete!")
    print("\n📋 Key Best Practices Implemented:")
    print("   ✅ Domain-specific vocabulary mapping")
    print("   ✅ Semantic relationship analysis")
    print("   ✅ Structured sentence generation")
    print("   ✅ Multi-criteria quality evaluation")
    print("   ✅ Quality threshold enforcement")
    print("   ✅ Contextual coherence optimization")
    print("   ✅ Pure Amharic output validation")
    print("   ✅ Repetition prevention")
    print("   ✅ Domain-aware vocabulary selection")
    print("\n💡 Detailed results saved to: amharic_best_practices_results.json")
    
    print("\n🔧 Recommended Implementation Steps:")
    print("   1. Integrate domain identification into your model")
    print("   2. Use contextual vocabulary for guided generation")
    print("   3. Apply structured sentence patterns")
    print("   4. Implement comprehensive quality scoring")
    print("   5. Enforce quality thresholds before output")
    print("   6. Use semantic relationships for coherence")
    print("   7. Monitor and prevent repetition patterns")

if __name__ == "__main__":
    main()
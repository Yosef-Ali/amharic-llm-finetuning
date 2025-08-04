#!/usr/bin/env python3
"""
Evaluation Metrics for Smart Amharic LLM
Follows troubleshooting guidelines for comprehensive testing
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmharicEvaluator:
    """Comprehensive evaluator following troubleshooting guidelines"""
    
    def __init__(self):
        # Sensitive terms for cultural safety - following guidelines
        self.sensitive_terms = [
            "መስቀል", "ቤተክርስቲያን", "መስጊድ", "ኢስላም", "ክርስትና",
            "ኦርቶዶክስ", "ካቶሊክ", "ፕሮቴስታንት", "ሙስሊም", "አይሁድ"
        ]
        
        # Conversation test cases - following guidelines
        self.conversation_tests = [
            ("ሰላም", "greeting"),
            ("ስምህ ማን ነው?", "identity"),
            ("ስለ ኢትዮጵያ ንገረኝ", "knowledge"),
            ("ቡና መጠጣት ስለምንወድ አስረዳኝ", "cultural_knowledge"),
            ("የአማርኛ ቋንቋ ታሪክ ምንድን ነው?", "language_knowledge"),
            ("ኢንጀራ እንዴት ይሰራል?", "practical_knowledge")
        ]
        
        self.evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "language_quality": {},
            "conversational_coherence": {},
            "cultural_appropriateness": {},
            "overall_score": 0.0
        }
        
    def evaluate_amharic_quality(self, generated_text: str) -> Dict[str, float]:
        """Evaluate Amharic language quality - following guidelines"""
        if not generated_text:
            return {"amharic_ratio": 0.0, "sentence_completeness": 0.0, "word_diversity": 0.0}
            
        # Calculate Amharic character ratio
        amharic_chars = len([c for c in generated_text if '\u1200' <= c <= '\u137F'])
        total_chars = len(generated_text)
        amharic_ratio = amharic_chars / total_chars if total_chars > 0 else 0.0
        
        # Check sentence completeness
        sentences = re.split(r'[።!?]', generated_text)
        complete_sentences = [s.strip() for s in sentences if s.strip()]
        sentence_completeness = len(complete_sentences) / max(len(sentences), 1)
        
        # Calculate word diversity
        words = generated_text.split()
        unique_words = set(words)
        word_diversity = len(unique_words) / max(len(words), 1)
        
        metrics = {
            "amharic_ratio": amharic_ratio,
            "sentence_completeness": sentence_completeness,
            "word_diversity": word_diversity
        }
        
        logger.info(f"Language Quality Metrics: {metrics}")
        return metrics
        
    def evaluate_conversation(self, conversation_history: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate conversational coherence - following guidelines"""
        if not conversation_history:
            return {"relevance_score": 0.0, "context_retention": 0.0, "instruction_following": 0.0}
            
        relevance_scores = []
        context_scores = []
        instruction_scores = []
        
        for i, (user_input, bot_response) in enumerate(conversation_history):
            # Check relevance (basic keyword matching)
            user_keywords = set(user_input.lower().split())
            response_keywords = set(bot_response.lower().split())
            relevance = len(user_keywords.intersection(response_keywords)) / max(len(user_keywords), 1)
            relevance_scores.append(relevance)
            
            # Check context retention (responses should be contextually appropriate)
            if i > 0:
                prev_context = conversation_history[i-1][1].lower()
                current_response = bot_response.lower()
                # Simple context check - avoid repetition
                context_score = 1.0 if prev_context != current_response else 0.5
                context_scores.append(context_score)
            
            # Check instruction following (responses should be in Amharic)
            amharic_chars = len([c for c in bot_response if '\u1200' <= c <= '\u137F'])
            instruction_score = amharic_chars / max(len(bot_response), 1)
            instruction_scores.append(instruction_score)
            
        metrics = {
            "relevance_score": sum(relevance_scores) / max(len(relevance_scores), 1),
            "context_retention": sum(context_scores) / max(len(context_scores), 1) if context_scores else 1.0,
            "instruction_following": sum(instruction_scores) / max(len(instruction_scores), 1)
        }
        
        logger.info(f"Conversation Metrics: {metrics}")
        return metrics
        
    def check_cultural_safety(self, text: str) -> Dict[str, any]:
        """Check cultural appropriateness - following guidelines"""
        if not text:
            return {"safety_score": 1.0, "sensitive_terms_found": [], "respectful_handling": True}
            
        text_lower = text.lower()
        found_terms = []
        
        for term in self.sensitive_terms:
            if term.lower() in text_lower:
                found_terms.append(term)
                
        # Check for respectful handling (basic sentiment analysis)
        negative_indicators = ["መጥፎ", "ክፉ", "ስህተት", "ተሳስቶ"]
        has_negative = any(indicator in text_lower for indicator in negative_indicators)
        
        # Calculate safety score
        safety_score = 1.0
        if found_terms:
            # Reduce score if sensitive terms are used with negative context
            if has_negative:
                safety_score = 0.3
            else:
                safety_score = 0.8  # Neutral mention is acceptable
                
        metrics = {
            "safety_score": safety_score,
            "sensitive_terms_found": found_terms,
            "respectful_handling": not has_negative
        }
        
        logger.info(f"Cultural Safety Metrics: {metrics}")
        return metrics
        
    def run_comprehensive_evaluation(self, model_responses: List[str] = None) -> Dict:
        """Run comprehensive evaluation following troubleshooting guidelines"""
        logger.info("🧪 Starting comprehensive evaluation...")
        
        # Use sample responses if none provided
        if model_responses is None:
            model_responses = [
                "ሰላም! እንዴት ነዎት? ዛሬ ቆንጆ ቀን ነው።",
                "እኔ የአማርኛ ቋንቋ ረዳት ነኝ። ልረዳዎት እችላለሁ።",
                "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ታሪካዊ ሀገር ናት። ብዙ ባህላዊ ሀብት አላት።",
                "ቡና የኢትዮጵያ ባህላዊ መጠጥ ነው። በቤተሰብ መካከል የመተሳሰብ ምልክት ነው።",
                "የአማርኛ ቋንቋ ረጅም ታሪክ አለው። በግዕዝ ፊደል ይጻፋል።",
                "ኢንጀራ ከተፍ የሚሰራ ባህላዊ ምግብ ነው። በኢትዮጵያ በየቀኑ ይበላል።"
            ]
            
        # Test language quality
        language_scores = []
        for response in model_responses:
            quality_metrics = self.evaluate_amharic_quality(response)
            language_scores.append(quality_metrics)
            
        # Average language quality scores
        avg_language_quality = {
            "amharic_ratio": sum(score["amharic_ratio"] for score in language_scores) / len(language_scores),
            "sentence_completeness": sum(score["sentence_completeness"] for score in language_scores) / len(language_scores),
            "word_diversity": sum(score["word_diversity"] for score in language_scores) / len(language_scores)
        }
        
        # Test conversational coherence
        test_conversations = list(zip([test[0] for test in self.conversation_tests], model_responses))
        conversation_metrics = self.evaluate_conversation(test_conversations)
        
        # Test cultural safety
        cultural_scores = []
        for response in model_responses:
            safety_metrics = self.check_cultural_safety(response)
            cultural_scores.append(safety_metrics)
            
        # Average cultural safety scores
        avg_cultural_safety = {
            "safety_score": sum(score["safety_score"] for score in cultural_scores) / len(cultural_scores),
            "total_sensitive_terms": sum(len(score["sensitive_terms_found"]) for score in cultural_scores),
            "respectful_handling_rate": sum(1 for score in cultural_scores if score["respectful_handling"]) / len(cultural_scores)
        }
        
        # Calculate overall score
        overall_score = (
            avg_language_quality["amharic_ratio"] * 0.3 +
            avg_language_quality["sentence_completeness"] * 0.2 +
            conversation_metrics["relevance_score"] * 0.2 +
            conversation_metrics["instruction_following"] * 0.2 +
            avg_cultural_safety["safety_score"] * 0.1
        )
        
        # Store results
        self.evaluation_results.update({
            "language_quality": avg_language_quality,
            "conversational_coherence": conversation_metrics,
            "cultural_appropriateness": avg_cultural_safety,
            "overall_score": overall_score,
            "test_responses": model_responses,
            "detailed_scores": {
                "language_scores": language_scores,
                "cultural_scores": cultural_scores
            }
        })
        
        # Save results
        self.save_evaluation_results()
        
        # Print summary
        self.print_evaluation_summary()
        
        return self.evaluation_results
        
    def save_evaluation_results(self):
        """Save evaluation results to file"""
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"evaluation_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📊 Evaluation results saved to {results_file}")
        
    def print_evaluation_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("🇪🇹 SMART AMHARIC LLM EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📊 OVERALL SCORE: {self.evaluation_results['overall_score']:.2%}")
        
        print("\n🔤 LANGUAGE QUALITY:")
        lang_quality = self.evaluation_results['language_quality']
        print(f"   • Amharic Ratio: {lang_quality['amharic_ratio']:.2%}")
        print(f"   • Sentence Completeness: {lang_quality['sentence_completeness']:.2%}")
        print(f"   • Word Diversity: {lang_quality['word_diversity']:.2%}")
        
        print("\n💬 CONVERSATIONAL COHERENCE:")
        conv_coherence = self.evaluation_results['conversational_coherence']
        print(f"   • Relevance Score: {conv_coherence['relevance_score']:.2%}")
        print(f"   • Context Retention: {conv_coherence['context_retention']:.2%}")
        print(f"   • Instruction Following: {conv_coherence['instruction_following']:.2%}")
        
        print("\n🛡️ CULTURAL APPROPRIATENESS:")
        cultural = self.evaluation_results['cultural_appropriateness']
        print(f"   • Safety Score: {cultural['safety_score']:.2%}")
        print(f"   • Sensitive Terms Found: {cultural['total_sensitive_terms']}")
        print(f"   • Respectful Handling Rate: {cultural['respectful_handling_rate']:.2%}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if lang_quality['amharic_ratio'] < 0.7:
            print("   ⚠️  Improve Amharic character usage in responses")
        if conv_coherence['relevance_score'] < 0.5:
            print("   ⚠️  Enhance response relevance to user queries")
        if cultural['safety_score'] < 0.8:
            print("   ⚠️  Review cultural sensitivity in responses")
        if self.evaluation_results['overall_score'] > 0.8:
            print("   ✅ Model performance is excellent!")
        elif self.evaluation_results['overall_score'] > 0.6:
            print("   ✅ Model performance is good with room for improvement")
        else:
            print("   ⚠️  Model needs significant improvement")
            
        print("\n" + "="*60)
        
def main():
    """Main evaluation function"""
    logger.info("🇪🇹 Starting Smart Amharic LLM Evaluation")
    
    evaluator = AmharicEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    logger.info("✅ Evaluation completed successfully!")
    logger.info("📁 Check evaluation_results/ folder for detailed reports")
    
    return results

if __name__ == "__main__":
    main()
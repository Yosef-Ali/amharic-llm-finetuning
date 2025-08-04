"""Evaluation metrics for Amharic text generation."""

import re
import math
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import statistics


class AmharicTextEvaluator:
    """Comprehensive evaluation metrics for Amharic text quality."""
    
    def __init__(self):
        # Amharic-specific patterns
        self.amharic_chars = set('ሀሁሂሃሄህሆለሉሊላሌልሎሐሑሒሓሔሕሖመሙሚማሜምሞሠሡሢሣሤሥሦረሩሪራሬርሮሰሱሲሳሴስሶሸሹሺሻሼሽሾቀቁቂቃቄቅቆቈቊቋቌቍበቡቢባቤብቦቨቩቪቫቬቭቮተቱቲታቴትቶቸቹቺቻቼችቾኀኁኂኃኄኅኆነኑኒናኔንኖኘኙኚኛኜኝኞአኡኢኣኤእኦከኩኪካኬክኮኸኹኺኻኼኽኾወዉዊዋዌውዎዐዑዒዓዔዕዖዘዙዚዛዜዝዞዠዡዢዣዤዥዦየዩዪያዬይዮደዱዲዳዴድዶዸዹዺዻዼዽዾጀጁጂጃጄጅጆገጉጊጋጌግጎጐጒጓጔጕጠጡጢጣጤጥጦጨጩጪጫጬጭጮፀፁፂፃፄፅፆፈፉፊፋፌፍፎፐፑፒፓፔፕፖ')
        
        # Common Amharic words for validation
        self.common_words = {
            'ነው', 'ናት', 'ናቸው', 'አለ', 'ሆነ', 'መጣ', 'ሄደ', 'ሰራ', 'ተናገረ',
            'እና', 'ወይም', 'ግን', 'ከዚያ', 'በተጨማሪ', 'ስለዚህ', 'ይሁን እንጂ',
            'ኢትዮጵያ', 'አዲስ አበባ', 'ሰላም', 'ሰው', 'ሰዎች', 'ሀገር', 'ህዝብ',
            'መንግሥት', 'ትምህርት', 'ወጣቶች', 'ህጻናት', 'ሴቶች', 'ባህል', 'ታሪክ'
        }
        
        # Sentence ending patterns
        self.sentence_endings = {'.', '።', '!', '?', '፡', '፣', '፤', '፥', '፦', '፧', '፨'}
        
    def evaluate_text(self, text: str) -> Dict[str, float]:
        """Comprehensive evaluation of Amharic text."""
        if not text or not text.strip():
            return self._empty_scores()
        
        text = text.strip()
        
        scores = {}
        
        # Basic metrics
        scores.update(self._basic_metrics(text))
        
        # Language-specific metrics
        scores.update(self._amharic_metrics(text))
        
        # Fluency metrics
        scores.update(self._fluency_metrics(text))
        
        # Coherence metrics
        scores.update(self._coherence_metrics(text))
        
        # Overall quality score
        scores['overall_quality'] = self._calculate_overall_score(scores)
        
        return scores
    
    def _empty_scores(self) -> Dict[str, float]:
        """Return zero scores for empty text."""
        return {
            'length': 0, 'word_count': 0, 'sentence_count': 0,
            'amharic_ratio': 0, 'proper_endings': 0, 'word_variety': 0,
            'avg_word_length': 0, 'sentence_length_variety': 0,
            'fluency_score': 0, 'coherence_score': 0, 'overall_quality': 0
        }
    
    def _basic_metrics(self, text: str) -> Dict[str, float]:
        """Basic text statistics."""
        words = text.split()
        sentences = self._split_sentences(text)
        
        return {
            'length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    
    def _amharic_metrics(self, text: str) -> Dict[str, float]:
        """Amharic-specific quality metrics."""
        # Amharic character ratio
        amharic_chars = sum(1 for c in text if c in self.amharic_chars)
        total_chars = sum(1 for c in text if c.isalpha())
        amharic_ratio = amharic_chars / total_chars if total_chars > 0 else 0
        
        # Proper sentence endings
        sentences = self._split_sentences(text)
        proper_endings = sum(1 for s in sentences if any(s.strip().endswith(e) for e in self.sentence_endings))
        proper_ending_ratio = proper_endings / len(sentences) if sentences else 0
        
        # Common word usage
        words = set(text.split())
        common_word_count = len(words.intersection(self.common_words))
        common_word_ratio = common_word_count / len(words) if words else 0
        
        return {
            'amharic_ratio': amharic_ratio,
            'proper_endings': proper_ending_ratio,
            'common_words': common_word_ratio
        }
    
    def _fluency_metrics(self, text: str) -> Dict[str, float]:
        """Text fluency evaluation."""
        words = text.split()
        sentences = self._split_sentences(text)
        
        # Word variety (unique words / total words)
        unique_words = len(set(words))
        word_variety = unique_words / len(words) if words else 0
        
        # Sentence length variety (standard deviation of sentence lengths)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        length_variety = 0
        if len(sentence_lengths) > 1:
            length_variety = statistics.stdev(sentence_lengths) / statistics.mean(sentence_lengths)
        
        # Repetition penalty
        word_counts = Counter(words)
        max_repetitions = max(word_counts.values()) if word_counts else 1
        repetition_penalty = 1.0 / max_repetitions if max_repetitions > 0 else 0
        
        fluency_score = (word_variety + length_variety + repetition_penalty) / 3
        
        return {
            'word_variety': word_variety,
            'sentence_length_variety': length_variety,
            'repetition_penalty': repetition_penalty,
            'fluency_score': fluency_score
        }
    
    def _coherence_metrics(self, text: str) -> Dict[str, float]:
        """Text coherence evaluation."""
        sentences = self._split_sentences(text)
        
        if len(sentences) < 2:
            return {'coherence_score': 0.8}  # Single sentence gets decent score
        
        # Topic consistency (word overlap between sentences)
        word_overlaps = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].split())
            words2 = set(sentences[i + 1].split())
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            overlap_ratio = overlap / total if total > 0 else 0
            word_overlaps.append(overlap_ratio)
        
        avg_overlap = statistics.mean(word_overlaps) if word_overlaps else 0
        
        # Conjunction usage
        conjunctions = {'እና', 'ወይም', 'ግን', 'ከዚያ', 'በተጨማሪ', 'ስለዚህ', 'ይሁን እንጂ', 'በመሆኑም'}
        conjunction_count = sum(1 for word in text.split() if word in conjunctions)
        conjunction_ratio = conjunction_count / len(sentences) if sentences else 0
        
        coherence_score = (avg_overlap + min(conjunction_ratio, 0.3)) / 1.3
        
        return {'coherence_score': coherence_score}
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'amharic_ratio': 0.2,
            'proper_endings': 0.15,
            'fluency_score': 0.3,
            'coherence_score': 0.25,
            'common_words': 0.1
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in scores:
                weighted_sum += scores[metric] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using Amharic punctuation."""
        # Split on various Amharic sentence endings
        pattern = r'[።!?፡]+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Dict[str, float]]:
        """Compare two texts and return evaluation scores."""
        scores1 = self.evaluate_text(text1)
        scores2 = self.evaluate_text(text2)
        
        return {
            'text1': scores1,
            'text2': scores2,
            'differences': {k: scores2[k] - scores1[k] for k in scores1.keys()}
        }
    
    def evaluate_multiple(self, texts: List[str]) -> Dict[str, any]:
        """Evaluate multiple texts and return statistics."""
        if not texts:
            return {}
        
        all_scores = [self.evaluate_text(text) for text in texts]
        
        # Calculate statistics for each metric
        metrics = all_scores[0].keys()
        stats = {}
        
        for metric in metrics:
            values = [scores[metric] for scores in all_scores]
            stats[metric] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values)
            }
        
        return {
            'individual_scores': all_scores,
            'statistics': stats,
            'best_text_index': max(range(len(all_scores)), key=lambda i: all_scores[i]['overall_quality']),
            'worst_text_index': min(range(len(all_scores)), key=lambda i: all_scores[i]['overall_quality'])
        }


def evaluate_generation_quality(generated_texts: List[str], reference_texts: List[str] = None) -> Dict:
    """High-level function to evaluate generation quality."""
    evaluator = AmharicTextEvaluator()
    
    results = {
        'generated': evaluator.evaluate_multiple(generated_texts)
    }
    
    if reference_texts:
        results['reference'] = evaluator.evaluate_multiple(reference_texts)
        
        # Compare generated vs reference
        gen_scores = results['generated']['statistics']
        ref_scores = results['reference']['statistics']
        
        results['comparison'] = {}
        for metric in gen_scores.keys():
            gen_mean = gen_scores[metric]['mean']
            ref_mean = ref_scores[metric]['mean']
            improvement = (gen_mean - ref_mean) / ref_mean if ref_mean != 0 else 0
            results['comparison'][metric] = {
                'generated': gen_mean,
                'reference': ref_mean,
                'improvement': improvement
            }
    
    return results
import re
import os
import sys
import json
import numpy as np
from collections import Counter
from pathlib import Path

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AmharicLinguisticEvaluator:
    """Evaluator for Amharic text linguistic quality."""
    
    def __init__(self, grammar_rules_file=None, cultural_terms_file=None, discourse_markers_file=None):
        """Initialize the evaluator with optional rule files.
        
        Args:
            grammar_rules_file: Path to grammar rules file (optional)
            cultural_terms_file: Path to cultural terms file (optional)
            discourse_markers_file: Path to discourse markers file (optional)
        """
        # Load grammar rules if provided
        self.grammar_rules = {}
        if grammar_rules_file and os.path.exists(grammar_rules_file):
            self.grammar_rules = self._load_grammar_rules(grammar_rules_file)
        else:
            print("No grammar rules file provided or file not found. Using default rules.")
            self.grammar_rules = self._get_default_grammar_rules()
        
        # Load cultural terms if provided
        self.cultural_terms = {}
        if cultural_terms_file and os.path.exists(cultural_terms_file):
            self.cultural_terms = self._load_cultural_terms(cultural_terms_file)
        else:
            print("No cultural terms file provided or file not found. Using default terms.")
            self.cultural_terms = self._get_default_cultural_terms()
        
        # Load discourse markers if provided
        self.discourse_markers = []
        if discourse_markers_file and os.path.exists(discourse_markers_file):
            self.discourse_markers = self._load_discourse_markers(discourse_markers_file)
        else:
            print("No discourse markers file provided or file not found. Using default markers.")
            self.discourse_markers = self._get_default_discourse_markers()
        
        # Amharic character range (Ethiopic Unicode block)
        self.amharic_char_range = (0x1200, 0x137F)
    
    def _load_grammar_rules(self, file_path):
        """Load grammar rules from file.
        
        Args:
            file_path: Path to grammar rules file
            
        Returns:
            Dictionary of grammar rules
        """
        rules = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            pattern, correction = parts[0], parts[1]
                            rules[pattern] = correction
            print(f"Loaded {len(rules)} grammar rules from {file_path}")
        except Exception as e:
            print(f"Error loading grammar rules: {e}")
        return rules
    
    def _load_cultural_terms(self, file_path):
        """Load cultural terms from file.
        
        Args:
            file_path: Path to cultural terms file
            
        Returns:
            Dictionary of cultural terms and contexts
        """
        terms = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            term, context = parts[0], parts[1]
                            terms[term] = context
            print(f"Loaded {len(terms)} cultural terms from {file_path}")
        except Exception as e:
            print(f"Error loading cultural terms: {e}")
        return terms
    
    def _load_discourse_markers(self, file_path):
        """Load discourse markers from file.
        
        Args:
            file_path: Path to discourse markers file
            
        Returns:
            List of discourse markers
        """
        markers = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        markers.append(line.strip())
            print(f"Loaded {len(markers)} discourse markers from {file_path}")
        except Exception as e:
            print(f"Error loading discourse markers: {e}")
        return markers
    
    def _get_default_grammar_rules(self):
        """Get default grammar rules.
        
        Returns:
            Dictionary of default grammar rules
        """
        # These are simplified examples - a real implementation would have more comprehensive rules
        return {
            r'\b(\w+) \1\b': "Repeated word",  # Detect repeated words
            r'[^።፡፣፤፥፦፧\s\w]': "Invalid punctuation",  # Non-Amharic punctuation
            r'\b\w{1,2}\b': "Very short word",  # Very short words (potential errors)
            r'[A-Za-z]': "Latin character in Amharic text",  # Latin characters
        }
    
    def _get_default_cultural_terms(self):
        """Get default cultural terms.
        
        Returns:
            Dictionary of default cultural terms and contexts
        """
        # These are simplified examples - a real implementation would have more terms
        return {
            "ኢትዮጵያ": "country",
            "አዲስ አበባ": "capital city",
            "አማርኛ": "language",
            "ሀበሻ": "cultural identity",
            "ጥምቀት": "religious holiday",
            "መስቀል": "religious holiday",
            "እንጀራ": "food",
            "ዶሮ ወጥ": "food",
            "ሸማኔ": "clothing",
            "ጠላ": "drink",
            "ታቦት": "religious artifact",
            "አክሱም": "historical city",
            "ላሊበላ": "historical site",
        }
    
    def _get_default_discourse_markers(self):
        """Get default discourse markers.
        
        Returns:
            List of default discourse markers
        """
        # These are simplified examples - a real implementation would have more markers
        return [
            "ስለዚህ",  # therefore
            "እንዲሁም",  # also
            "ነገር ግን",  # but
            "በመሆኑም",  # consequently
            "ስለሆነም",  # thus
            "በተጨማሪም",  # in addition
            "ለምሳሌ",  # for example
            "እንደ",  # like/as
            "ቢሆንም",  # although
            "ከዚህ በተጨማሪ",  # furthermore
            "በአጭሩ",  # in short
            "በመጨረሻም",  # finally
        ]
    
    def calculate_amharic_ratio(self, text):
        """Calculate the ratio of Amharic characters in the text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Ratio of Amharic characters (0-1)
        """
        if not text:
            return 0.0
        
        amharic_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isspace():
                continue  # Skip whitespace
            
            total_chars += 1
            code_point = ord(char)
            if self.amharic_char_range[0] <= code_point <= self.amharic_char_range[1]:
                amharic_chars += 1
        
        return amharic_chars / max(1, total_chars)
    
    def evaluate_grammar(self, text):
        """Evaluate grammar quality of the text.
        
        Args:
            text: The text to evaluate
            
        Returns:
            Tuple of (grammar_score, issues)
        """
        if not text:
            return 1.0, []
        
        issues = []
        for pattern, description in self.grammar_rules.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                issues.append({
                    'pattern': pattern,
                    'description': description,
                    'match': match.group(0),
                    'position': match.span()
                })
        
        # Calculate score based on number of issues relative to text length
        # More sophisticated scoring would consider issue severity and text structure
        words = len(text.split())
        grammar_score = 1.0 - (len(issues) / max(10, words))
        grammar_score = max(0.0, min(1.0, grammar_score))  # Clamp to 0-1
        
        return grammar_score, issues
    
    def evaluate_cultural_relevance(self, text):
        """Evaluate cultural relevance of the text.
        
        Args:
            text: The text to evaluate
            
        Returns:
            Tuple of (relevance_score, relevant_terms)
        """
        if not text:
            return 0.0, []
        
        relevant_terms = []
        for term, context in self.cultural_terms.items():
            if term in text:
                relevant_terms.append({
                    'term': term,
                    'context': context,
                    'count': text.count(term)
                })
        
        # Calculate score based on number and diversity of cultural terms
        # More sophisticated scoring would consider term importance and context
        term_count = sum(item['count'] for item in relevant_terms)
        unique_terms = len(relevant_terms)
        
        # Balance between term count and diversity
        relevance_score = 0.7 * min(1.0, unique_terms / 10) + 0.3 * min(1.0, term_count / 20)
        
        return relevance_score, relevant_terms
    
    def evaluate_coherence(self, text):
        """Evaluate coherence of the text.
        
        Args:
            text: The text to evaluate
            
        Returns:
            Tuple of (coherence_score, coherence_details)
        """
        if not text:
            return 0.0, {}
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[።፡፣]', text) if s.strip()]
        if len(sentences) <= 1:
            return 0.5, {'reason': 'Too few sentences for coherence analysis'}
        
        # Check for discourse markers
        marker_count = 0
        marker_positions = []
        
        for i, sentence in enumerate(sentences):
            if i == 0:  # Skip first sentence
                continue
                
            for marker in self.discourse_markers:
                if sentence.startswith(marker) or f" {marker} " in sentence:
                    marker_count += 1
                    marker_positions.append((i, marker))
                    break
        
        # Check for topic consistency using word overlap between sentences
        word_sets = [set(re.findall(r'\b\w+\b', s.lower())) for s in sentences]
        overlap_scores = []
        
        for i in range(1, len(word_sets)):
            prev_words = word_sets[i-1]
            curr_words = word_sets[i]
            
            if not prev_words or not curr_words:
                continue
                
            # Jaccard similarity between consecutive sentences
            overlap = len(prev_words.intersection(curr_words)) / len(prev_words.union(curr_words))
            overlap_scores.append(overlap)
        
        # Calculate coherence score
        avg_overlap = sum(overlap_scores) / max(1, len(overlap_scores))
        marker_ratio = marker_count / max(1, len(sentences) - 1)
        
        # Combine discourse markers and word overlap for coherence score
        coherence_score = 0.6 * avg_overlap + 0.4 * marker_ratio
        coherence_score = max(0.0, min(1.0, coherence_score))  # Clamp to 0-1
        
        coherence_details = {
            'sentence_count': len(sentences),
            'discourse_marker_count': marker_count,
            'discourse_marker_ratio': marker_ratio,
            'avg_sentence_overlap': avg_overlap,
            'marker_positions': marker_positions
        }
        
        return coherence_score, coherence_details
    
    def evaluate_repetition(self, text):
        """Evaluate repetition in the text.
        
        Args:
            text: The text to evaluate
            
        Returns:
            Tuple of (repetition_score, repetition_details)
        """
        if not text:
            return 1.0, {}
        
        # Extract words (excluding punctuation and whitespace)
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 1.0, {'reason': 'No words found for repetition analysis'}
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Calculate repetition metrics
        total_words = len(words)
        unique_words = len(word_counts)
        vocabulary_diversity = unique_words / total_words
        
        # Find repeated n-grams (phrases)
        repeated_phrases = []
        for n in range(2, min(6, total_words // 2)):  # Check 2 to 5-grams
            ngrams = [' '.join(words[i:i+n]) for i in range(total_words - n + 1)]
            ngram_counts = Counter(ngrams)
            
            # Record phrases that repeat more than twice
            for phrase, count in ngram_counts.items():
                if count > 2:
                    repeated_phrases.append({
                        'phrase': phrase,
                        'count': count,
                        'length': n
                    })
        
        # Calculate repetition penalty based on repeated phrases
        phrase_penalty = sum(p['count'] * p['length'] for p in repeated_phrases) / max(1, total_words)
        phrase_penalty = min(1.0, phrase_penalty)  # Cap at 1.0
        
        # Final repetition score (higher is better - less repetition)
        repetition_score = 0.7 * vocabulary_diversity + 0.3 * (1.0 - phrase_penalty)
        repetition_score = max(0.0, min(1.0, repetition_score))  # Clamp to 0-1
        
        repetition_details = {
            'total_words': total_words,
            'unique_words': unique_words,
            'vocabulary_diversity': vocabulary_diversity,
            'most_common_words': word_counts.most_common(5),
            'repeated_phrases': sorted(repeated_phrases, key=lambda x: x['count'], reverse=True)[:5]
        }
        
        return repetition_score, repetition_details
    
    def evaluate_text(self, text):
        """Comprehensive evaluation of text quality.
        
        Args:
            text: The text to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        if not text:
            return {
                'error': 'Empty text provided',
                'overall_score': 0.0
            }
        
        # Calculate individual metrics
        amharic_ratio = self.calculate_amharic_ratio(text)
        grammar_score, grammar_issues = self.evaluate_grammar(text)
        relevance_score, relevant_terms = self.evaluate_cultural_relevance(text)
        coherence_score, coherence_details = self.evaluate_coherence(text)
        repetition_score, repetition_details = self.evaluate_repetition(text)
        
        # Calculate overall score with weighted components
        overall_score = (
            0.25 * amharic_ratio +
            0.25 * grammar_score +
            0.15 * relevance_score +
            0.20 * coherence_score +
            0.15 * repetition_score
        )
        
        # Prepare evaluation results
        evaluation = {
            'overall_score': overall_score,
            'metrics': {
                'amharic_ratio': amharic_ratio,
                'grammar_score': grammar_score,
                'cultural_relevance': relevance_score,
                'coherence': coherence_score,
                'repetition': repetition_score
            },
            'details': {
                'grammar_issues': grammar_issues[:10],  # Limit to top 10 issues
                'cultural_terms': relevant_terms,
                'coherence': coherence_details,
                'repetition': repetition_details
            }
        }
        
        return evaluation
    
    def compare_texts(self, original_text, generated_text):
        """Compare original and generated texts.
        
        Args:
            original_text: The original text
            generated_text: The generated text to compare
            
        Returns:
            Dictionary with comparison results
        """
        # Evaluate both texts
        original_eval = self.evaluate_text(original_text)
        generated_eval = self.evaluate_text(generated_text)
        
        # Calculate differences
        metric_diffs = {}
        for metric, score in generated_eval['metrics'].items():
            original_score = original_eval['metrics'].get(metric, 0)
            metric_diffs[metric] = score - original_score
        
        # Calculate overall difference
        overall_diff = generated_eval['overall_score'] - original_eval['overall_score']
        
        comparison = {
            'original_score': original_eval['overall_score'],
            'generated_score': generated_eval['overall_score'],
            'overall_difference': overall_diff,
            'metric_differences': metric_diffs,
            'original_evaluation': original_eval,
            'generated_evaluation': generated_eval
        }
        
        return comparison


def main():
    """Main function to demonstrate linguistic evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Amharic text linguistic quality')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to evaluate')
    parser.add_argument('--file', type=str, default=None,
                        help='File containing text to evaluate')
    parser.add_argument('--compare', type=str, default=None,
                        help='File containing text to compare with')
    parser.add_argument('--grammar-rules', type=str, default=None,
                        help='Path to grammar rules file')
    parser.add_argument('--cultural-terms', type=str, default=None,
                        help='Path to cultural terms file')
    parser.add_argument('--discourse-markers', type=str, default=None,
                        help='Path to discourse markers file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for evaluation results (JSON)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = AmharicLinguisticEvaluator(
        grammar_rules_file=args.grammar_rules,
        cultural_terms_file=args.cultural_terms,
        discourse_markers_file=args.discourse_markers
    )
    
    # Get text to evaluate
    text_to_evaluate = None
    if args.text:
        text_to_evaluate = args.text
    elif args.file and os.path.exists(args.file):
        with open(args.file, 'r', encoding='utf-8') as f:
            text_to_evaluate = f.read()
    
    if not text_to_evaluate:
        print("No text provided for evaluation. Use --text or --file.")
        return
    
    # Evaluate text
    if args.compare and os.path.exists(args.compare):
        # Compare with another text
        with open(args.compare, 'r', encoding='utf-8') as f:
            comparison_text = f.read()
        
        results = evaluator.compare_texts(comparison_text, text_to_evaluate)
        print(f"\nComparison Results:")
        print(f"Original Score: {results['original_score']:.4f}")
        print(f"Generated Score: {results['generated_score']:.4f}")
        print(f"Overall Difference: {results['overall_difference']:.4f}")
        
        print("\nMetric Differences:")
        for metric, diff in results['metric_differences'].items():
            print(f"  {metric}: {diff:+.4f}")
    else:
        # Single text evaluation
        results = evaluator.evaluate_text(text_to_evaluate)
        print(f"\nEvaluation Results:")
        print(f"Overall Score: {results['overall_score']:.4f}")
        
        print("\nMetric Scores:")
        for metric, score in results['metrics'].items():
            print(f"  {metric}: {score:.4f}")
        
        print("\nGrammar Issues:")
        for issue in results['details']['grammar_issues'][:5]:  # Show top 5
            print(f"  - {issue['description']}: '{issue['match']}'")
        
        print("\nCultural Terms:")
        for term in results['details']['cultural_terms'][:5]:  # Show top 5
            print(f"  - {term['term']} ({term['context']}): {term['count']} occurrences")
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
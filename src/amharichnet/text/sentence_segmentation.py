"""
Advanced Sentence Boundary Detection for Amharic Text
Enhanced segmentation with linguistic patterns and neural approaches
"""

import re
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import unicodedata


@dataclass 
class SegmentationConfig:
    """Configuration for sentence segmentation."""
    min_sentence_length: int = 5
    max_sentence_length: int = 200
    confidence_threshold: float = 0.7
    use_neural_segmentation: bool = True
    use_rule_based_fallback: bool = True


class AmharicSentenceSegmenter:
    """Advanced sentence boundary detection for Amharic text."""
    
    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()
        
        # Amharic-specific punctuation marks
        self.sentence_endings = {
            '።': 1.0,  # Primary sentence ending
            '!': 0.9,  # Exclamation
            '?': 0.9,  # Question
            '፡': 0.7,  # Colon/pause
            '፣': 0.3,  # Comma (weaker boundary)
            '፤': 0.6,  # Semicolon
            '፥': 0.4,  # Minor pause
            '፦': 0.5,  # List marker
            '፧': 0.8,  # Question mark
            '፨': 0.7,  # Paragraph separator
        }
        
        # Western punctuation (common in mixed text)
        self.western_endings = {
            '.': 0.8,
            '!': 0.9,
            '?': 0.9,
            ';': 0.6,
            ':': 0.4,
        }
        
        # Combine all punctuation
        self.all_endings = {**self.sentence_endings, **self.western_endings}
        
        # Amharic characters for context analysis
        self.amharic_chars = set('ሀሁሂሃሄህሆለሉሊላሌልሎሐሑሒሓሔሕሖመሙሚማሜምሞሠሡሢሣሤሥሦረሩሪራሬርሮሰሱሲሳሴስሶሸሹሺሻሼሽሾቀቁቂቃቄቅቆቈቊቋቌቍበቡቢባቤብቦቨቩቪቫቬቭቮተቱቲታቴትቶቸቹቺቻቼችቾኀኁኂኃኄኅኆነኑኒናኔንኖኘኙኚኛኜኝኞአኡኢኣኤእኦከኩኪካኬክኮኸኹኺኻኼኽኾወዉዊዋዌውዎዐዑዒዓዔዕዖዘዙዚዛዜዝዞዠዡዢዣዤዥዦየዩዪያዬይዮደዱዲዳዴድዶዸዹዺዻዼዽዾጀጁጂጃጄጅጆገጉጊጋጌግጎጐጒጓጔጕጠጡጢጣጤጥጦጨጩጪጫጬጭጮፀፁፂፃፄፅፆፈፉፊፋፌፍፎፐፑፒፓፔፕፖ')
        
        # Common sentence starters in Amharic
        self.sentence_starters = {
            'እና', 'ወይም', 'ነገር ግን', 'ስለዚህ', 'በተጨማሪ', 'ከዚያ', 'አንደኛ', 'ሁለተኛ', 'ሦስተኛ',
            'በመጀመሪያ', 'በመጨረሻ', 'በተቃራኒው', 'ይሁን እንጂ', 'በመሆኑም', 'አሁን', 'ከዚህ በፊት'
        }
        
        # Common abbreviations that shouldn't end sentences
        self.abbreviations = {
            'ዓ.ም', 'ዓ.ዓ', 'ክ.ክ', 'ወ.ዘ.ተ', 'ወ.ዘ', 'ሰ.ዓ', 'ት.ም', 'ዶ/ር', 'ፕ/ር', 'ወ/ሮ', 'ወ/ሪት'
        }
        
        # Initialize neural segmenter if enabled
        self.neural_segmenter = None
        if self.config.use_neural_segmentation:
            self.neural_segmenter = self._init_neural_segmenter()
    
    def _init_neural_segmenter(self) -> Optional[nn.Module]:
        """Initialize neural sentence segmentation model."""
        try:
            return NeuralSentenceSegmenter()
        except Exception as e:
            print(f"⚠️  Neural segmenter initialization failed: {e}")
            return None
    
    def _is_amharic_char(self, char: str) -> bool:
        """Check if character is Amharic."""
        return char in self.amharic_chars
    
    def _get_character_context(self, text: str, position: int, window: int = 5) -> Dict[str, float]:
        """Analyze character context around a position."""
        start = max(0, position - window)
        end = min(len(text), position + window + 1)
        context = text[start:end]
        
        amharic_count = sum(1 for c in context if self._is_amharic_char(c))
        total_chars = len([c for c in context if c.isalpha()])
        
        return {
            'amharic_ratio': amharic_count / max(total_chars, 1),
            'context_length': len(context),
            'has_digits': any(c.isdigit() for c in context),
            'has_punctuation': any(c in self.all_endings for c in context)
        }
    
    def _calculate_boundary_score(self, text: str, position: int, punct_char: str) -> float:
        """Calculate probability that position is a sentence boundary."""
        base_score = self.all_endings.get(punct_char, 0.0)
        
        if base_score == 0.0:
            return 0.0
        
        # Context analysis
        context = self._get_character_context(text, position)
        
        # Look at next few characters
        next_text = text[position + 1:position + 10].strip()
        prev_text = text[max(0, position - 10):position].strip()
        
        # Modifiers for score
        modifiers = []
        
        # Boost score if next word starts with capital or sentence starter
        if next_text:
            first_word = next_text.split()[0] if next_text.split() else ""
            if first_word in self.sentence_starters:
                modifiers.append(0.3)
            elif first_word and (first_word[0].isupper() or self._is_amharic_char(first_word[0])):
                modifiers.append(0.2)
        
        # Reduce score if looks like abbreviation
        words_before = prev_text.split()
        if words_before:
            last_word = words_before[-1]
            if last_word in self.abbreviations:
                modifiers.append(-0.4)
            elif len(last_word) <= 3 and '.' in punct_char:  # Likely abbreviation
                modifiers.append(-0.3)
        
        # Adjust based on Amharic context
        if context['amharic_ratio'] > 0.8:
            modifiers.append(0.1)  # Boost for pure Amharic text
        
        # Length-based adjustments
        if position < 10:  # Too early in text
            modifiers.append(-0.2)
        
        # Calculate final score
        final_score = base_score + sum(modifiers)
        return max(0.0, min(1.0, final_score))
    
    def _rule_based_segment(self, text: str) -> List[Tuple[int, int, float]]:
        """Rule-based sentence segmentation."""
        boundaries = []
        
        # Find all potential boundary positions
        for i, char in enumerate(text):
            if char in self.all_endings:
                score = self._calculate_boundary_score(text, i, char)
                if score >= self.config.confidence_threshold:
                    boundaries.append((i, i + 1, score))
        
        return boundaries
    
    def _neural_segment(self, text: str) -> List[Tuple[int, int, float]]:
        """Neural-based sentence segmentation."""
        if self.neural_segmenter is None:
            return []
        
        try:
            return self.neural_segmenter.segment(text)
        except Exception as e:
            print(f"⚠️  Neural segmentation failed: {e}")
            return []
    
    def _merge_boundaries(self, 
                         rule_boundaries: List[Tuple[int, int, float]], 
                         neural_boundaries: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Merge rule-based and neural boundaries."""
        all_boundaries = rule_boundaries + neural_boundaries
        
        # Sort by position
        all_boundaries.sort(key=lambda x: x[0])
        
        # Merge nearby boundaries
        merged = []
        for start, end, score in all_boundaries:
            if merged and abs(merged[-1][0] - start) <= 2:
                # Merge with previous boundary (take higher score)
                prev_start, prev_end, prev_score = merged[-1]
                if score > prev_score:
                    merged[-1] = (start, end, score)
            else:
                merged.append((start, end, score))
        
        return merged
    
    def _extract_sentences(self, text: str, boundaries: List[Tuple[int, int, float]]) -> List[str]:
        """Extract sentences based on boundaries."""
        if not boundaries:
            return [text.strip()] if text.strip() else []
        
        sentences = []
        last_end = 0
        
        for start, end, score in boundaries:
            # Extract sentence
            sentence = text[last_end:end].strip()
            if sentence and len(sentence) >= self.config.min_sentence_length:
                sentences.append(sentence)
            last_end = end
        
        # Add remaining text
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining and len(remaining) >= self.config.min_sentence_length:
                sentences.append(remaining)
        
        return sentences
    
    def segment(self, text: str) -> List[str]:
        """Main segmentation method."""
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # Get boundaries from different methods
        rule_boundaries = self._rule_based_segment(text)
        
        neural_boundaries = []
        if self.config.use_neural_segmentation and self.neural_segmenter:
            neural_boundaries = self._neural_segment(text)
        
        # Merge boundaries
        if neural_boundaries and self.config.use_neural_segmentation:
            final_boundaries = self._merge_boundaries(rule_boundaries, neural_boundaries)
        else:
            final_boundaries = rule_boundaries
        
        # Extract sentences
        sentences = self._extract_sentences(text, final_boundaries)
        
        # Filter by length constraints
        filtered_sentences = []
        for sentence in sentences:
            if (self.config.min_sentence_length <= len(sentence) <= self.config.max_sentence_length):
                filtered_sentences.append(sentence)
            elif len(sentence) > self.config.max_sentence_length:
                # Split long sentences further
                long_parts = self._split_long_sentence(sentence)
                filtered_sentences.extend(long_parts)
        
        return filtered_sentences
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split overly long sentences at weaker boundaries."""
        if len(sentence) <= self.config.max_sentence_length:
            return [sentence]
        
        # Try splitting at weaker punctuation
        weak_punctuation = ['፣', '፤', '፥', ',', ';']
        
        for punct in weak_punctuation:
            if punct in sentence:
                parts = sentence.split(punct)
                if all(len(part.strip()) >= self.config.min_sentence_length for part in parts):
                    return [part.strip() + punct for part in parts[:-1]] + [parts[-1].strip()]
        
        # Fallback: split at word boundaries
        words = sentence.split()
        target_length = self.config.max_sentence_length // 2
        
        parts = []
        current_part = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > target_length and current_part:
                parts.append(' '.join(current_part))
                current_part = [word]
                current_length = len(word)
            else:
                current_part.append(word)
                current_length += len(word) + 1
        
        if current_part:
            parts.append(' '.join(current_part))
        
        return parts


class NeuralSentenceSegmenter(nn.Module):
    """Neural network for sentence boundary detection."""
    
    def __init__(self, char_vocab_size: int = 1000, hidden_dim: int = 128):
        super().__init__()
        self.char_vocab_size = char_vocab_size
        self.hidden_dim = hidden_dim
        
        # Character embedding
        self.char_embedding = nn.Embedding(char_vocab_size, 64)
        
        # BiLSTM for sequence modeling
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True, bidirectional=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Binary classification: boundary or not
        )
        
        # Character to index mapping
        self.char_to_idx = {}
        self._build_char_vocab()
    
    def _build_char_vocab(self):
        """Build character vocabulary for Amharic."""
        # Amharic characters
        amharic_chars = 'ሀሁሂሃሄህሆለሉሊላሌልሎሐሑሒሓሔሕሖመሙሚማሜምሞሠሡሢሣሤሥሦረሩሪራሬርሮሰሱሲሳሴስሶሸሹሺሻሼሽሾቀቁቂቃቄቅቆቈቊቋቌቍበቡቢባቤብቦቨቩቪቫቬቭቮተቱቲታቴትቶቸቹቺቻቼችቾኀኁኂኃኄኅኆነኑኒናኔንኖኘኙኚኛኜኝኞአኡኢኣኤእኦከኩኪካኬክኮኸኹኺኻኼኽኾወዉዊዋዌውዎዐዑዒዓዔዕዖዘዙዚዛዜዝዞዠዡዢዣዤዥዦየዩዪያዬይዮደዱዲዳዴድዶዸዹዺዻዼዽዾጀጁጂጃጄጅጆገጉጊጋጌግጎጐጒጓጔጕጠጡጢጣጤጥጦጨጩጪጫጬጭጮፀፁፂፃፄፅፆፈፉፊፋፌፍፎፐፑፒፓፔፕፖ'
        
        # Add common punctuation and space
        all_chars = amharic_chars + '።!?፡፣፤፥፦፧፨.!?;:, \n\t'
        
        # Build mapping
        self.char_to_idx = {'<UNK>': 0, '<PAD>': 1}
        for i, char in enumerate(set(all_chars), 2):
            self.char_to_idx[char] = i
    
    def encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text to character indices."""
        indices = []
        for char in text[:max_length]:
            indices.append(self.char_to_idx.get(char, 0))  # 0 = <UNK>
        
        # Pad sequence
        while len(indices) < max_length:
            indices.append(1)  # 1 = <PAD>
        
        return torch.tensor(indices, dtype=torch.long)
    
    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Character embedding
        embedded = self.char_embedding(char_ids)  # [batch, seq_len, embed_dim]
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim * 2]
        
        # Classification
        logits = self.classifier(lstm_out)  # [batch, seq_len, 2]
        
        return logits
    
    def segment(self, text: str) -> List[Tuple[int, int, float]]:
        """Segment text using neural model."""
        self.eval()
        
        # Encode text
        char_ids = self.encode_text(text).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            logits = self.forward(char_ids)
            probs = torch.softmax(logits, dim=-1)
            boundary_probs = probs[0, :, 1]  # Probability of boundary
        
        # Find boundaries above threshold
        boundaries = []
        for i, prob in enumerate(boundary_probs):
            if i < len(text) and prob > 0.7:  # Threshold
                boundaries.append((i, i + 1, prob.item()))
        
        return boundaries


def create_segmenter(config_path: Optional[str] = None) -> AmharicSentenceSegmenter:
    """Create Amharic sentence segmenter."""
    config = SegmentationConfig()
    
    if config_path:
        # Load config from file if provided
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Update config with loaded values
            for key, value in config_dict.get('segmentation', {}).items():
                if hasattr(config, key):
                    setattr(config, key, value)
        except Exception as e:
            print(f"⚠️  Config loading failed: {e}, using defaults")
    
    return AmharicSentenceSegmenter(config)


def main():
    """Demo of sentence segmentation."""
    print("🔤 Amharic Sentence Segmentation Demo")
    print("=" * 40)
    
    # Create segmenter
    segmenter = create_segmenter()
    
    # Test texts
    test_texts = [
        "ኢትዮጵያ ውብ ሀገር ናት። በአፍሪካ ቀንድ ትገኛለች። ብዙ ብሔሮች እና ብሔረሰቦች በሰላም የሚኖሩባት ሀገር ናት።",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት፣ በተጨማሪም የአፍሪካ ዲፕሎማሲያዊ ዋና ከተማ ነች። ብዙ ዓለም አቀፍ ድርጅቶች ዋና መሥሪያ ቤት ናት።",
        "ትምህርት በጣም አስፈላጊ ነው! ለሁሉም ህጻናት መብት ነው? በጥራት መሰጠት አለበት፡ ወጣቶችን ለወደፊት ያዘጋጃል።"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}:")
        print(f"Input: {text}")
        print("Sentences:")
        
        sentences = segmenter.segment(text)
        for j, sentence in enumerate(sentences, 1):
            print(f"  {j}. {sentence}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
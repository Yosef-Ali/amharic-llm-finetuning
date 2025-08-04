import re
from config import AmharicConfig

class AmharicTokenizer:
    """Proper Amharic tokenizer that respects SPACES and PUNCTUATION"""
    
    def __init__(self):
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4
        }
        self.vocab = self.special_tokens.copy()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # CRITICAL: PRESERVE PUNCTUATION AS SEPARATE TOKENS
        self.punctuation = {
            "።": "FULL_STOP",
            "፤": "SEMICOLON",
            "፡": "COMMA",
            "፣": "COMMA",
            "፥": "COLON",
            "፦": "QUESTION_MARK"
        }
        
        for punct in self.punctuation:
            if punct not in self.vocab:
                idx = len(self.vocab)
                self.vocab[punct] = idx
                self.inv_vocab[idx] = punct
        
        # Build vocabulary from corpus
        self._build_vocab_from_corpus()
        print(f"Tokenizer vocabulary size: {len(self.vocab)}")
    
    def _build_vocab_from_corpus(self):
        """Build vocabulary by reading the Amharic corpus and adding common words."""
        import os # Import os here for local scope
        corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "amharic_corpus.txt")
        
        # Add a comprehensive list of common Amharic words and numbers
        common_words = [
            # Common Nouns
            "ኢትዮጵያ", "አማርኛ", "ሀገር", "ቋንቋ", "ቡና", "ህዝብ", "ሰላም", "ውሃ", "ምግብ",
            "ሰው", "ቤት", "ትምህርት", "ጤና", "ስራ", "ገንዘብ", "ጊዜ", "አለም", "ችግር",
            "አዲስ", "አንድ", "ሁለት", "ሶስት", "አራት", "አምስት", "ስድስት", "ሰባት", "ስምንት", "ዘጠኝ", "አስር",
            "ሃያ", "ሰላሳ", "አርባ", "ሃምሳ", "መቶ", "ሺህ", "ሚሊዮን", "ቢሊዮን",
            "ቀን", "ሳምንት", "ወር", "ዓመት", "ሰዓት", "ደቂቃ", "ሰከንድ",
            "ትናንት", "ዛሬ", "ነገ", "ጧት", "ከሰዓት", "ማታ",
            "ትልቅ", "ትንሽ", "ረጅም", "አጭር", "ቀጭን", "ወፍራም", "ፈጣን", "ቀስ",
            "ጥሩ", "መጥፎ", "ቆንጆ", "አስቀያሚ", "ደስተኛ", "አዝኖ", "ብሩህ", "ጨለማ",
            "ቀይ", "ሰማያዊ", "አረንጓዴ", "ቢጫ", "ጥቁር", "ነጭ",
            "አዎ", "አይደለም", "እባክዎ", "አመሰግናለሁ", "ይቅርታ", "እንኳን ደህና መጡ",
            "እንዴት ነህ", "ደህና ነኝ", "ስምህ ማን ነው", "ስሜ", "ነው",
            "አለ", "ሆነ", "ሄደ", "መጣ", "ሰራ", "ተናገረ", "አየ", "ሰማ", "በላ", "ጠጣ",
            "ጻፈ", "አነበበ", "ተማረ", "አስተማረ", "ሰጠ", "ወሰደ", "ከፈተ", "ዘጋ",
            "ገባ", "ወጣ", "ተቀመጠ", "ቆመ", "ተኛ", "ነቃ", "ሳቀ", "አለቀሰ",
            "በ", "ለ", "ከ", "ላይ", "ታች", "ውስጥ", "ውጭ", "ፊት", "ኋላ", "ጎን", "መካከል",
            "እና", "ወይም", "ግን", "ስለ", "ምክንያቱም", "ስለዚህ", "ምንም", "ሁሉም",
            "እኔ", "አንተ", "አንቺ", "እሱ", "እሷ", "እኛ", "እናንተ", "እነሱ",
            "ይህ", "ያ", "እነዚህ", "እነዚያ", "ማን", "ምን", "የት", "መቼ", "እንዴት", "ለምን",
            "2019", "2023", "2024", "2025", "30", "23.6", "17.2", "2", "3", "5"
        ]
        
        for word in common_words:
            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.inv_vocab[idx] = word

        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    # Clean line similar to dataset.py to ensure consistency
                    clean_line = re.sub(r'[^\u1200-\u137F\s\u1361-\u1368]', '', line)
                    clean_line = re.sub(r'\s+', ' ', clean_line).strip()
                    
                    if clean_line:
                        tokens = self.tokenize(clean_line)
                        for token in tokens:
                            if token not in self.vocab:
                                idx = len(self.vocab)
                                self.vocab[token] = idx
                                self.inv_vocab[idx] = token
        except FileNotFoundError:
            print(f"⚠️ Corpus file not found at {corpus_path}. Using the default common words vocabulary.")
    
    def tokenize(self, text):
        """Tokenize Amharic text with PROPER WORD BOUNDARIES"""
        tokens = []
        
        # CRITICAL: SPLIT ON SPACES (NOT CHARACTER-LEVEL)
        words = text.split()
        
        for word in words:
            # Handle punctuation attached to words
            for punct in self.punctuation:
                if word.endswith(punct):
                    # Split punctuation from word
                    word_part = word[:-len(punct)]
                    if word_part:
                        tokens.append(word_part)
                    tokens.append(punct)
                    break
            else:
                tokens.append(word)
        
        return tokens
    
    def encode(self, text):
        """Convert text to token IDs"""
        tokens = self.tokenize(text)
        return [self.convert_token_to_id(token) for token in tokens]
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        tokens = [self.inv_vocab.get(tid, "[UNK]") for tid in token_ids if tid != self.pad_token_id]
        
        # CRITICAL: RECONSTRUCT WITH PROPER SPACING
        text = ""
        for i, token in enumerate(tokens):
            if token in self.punctuation:
                # No space BEFORE punctuation
                text += token
            else:
                # Space BEFORE word (except first token)
                if text and not text.endswith(" "):
                    text += " "
                text += token
        
        return text.strip()
    
    def convert_token_to_id(self, token):
        """Convert token to ID, handling unknowns"""
        return self.vocab.get(token, self.unk_token_id)
    
    @property
    def pad_token_id(self):
        return self.special_tokens["[PAD]"]
    
    @property
    def unk_token_id(self):
        return self.special_tokens["[UNK]"]

import re
from hybrid_tokenizer import HybridAmharicTokenizer

class AmharicValidator:
    """Validator for proper Amharic output"""
    
    def __init__(self):
        self.tokenizer = HybridAmharicTokenizer()
        self.punctuation = {
            "።": "FULL_STOP",
            "፤": "SEMICOLON",
            "፡": "COMMA",
            "፣": "COMMA",
            "፥": "COLON",
            "፦": "QUESTION_MARK"
        }
        self.cultural_rules = self._load_cultural_rules()
    
    def _load_cultural_rules(self):
        """Load Ethiopian-validated cultural rules"""
        return {
            "ቡና": {
                "sacred_context": "ቡና የኢትዮጵያ ባህላዊ ማህበራዊ ሥርዓት እና የእግዚአብሔር ስጦታ ነው።",
                "taboo_associations": ["addictive", "drug", "harmful", "የሰው ልጅ አለመለወጥ"]
            },
            "ኢትዮጵያ": {
                "required_context": "በአፍሪካ ውስጥ የምትገኝ",
                "forbidden_phrases": ["colony", "tribal"]
            }
        }
    
    def validate_spacing(self, text):
        """Validate proper Amharic spacing"""
        # Check for double spaces
        if "  " in text:
            return False, "DOUBLE_SPACE"
        
        # Check for space before punctuation
        for punct in self.punctuation:
            if f" {punct}" in text:
                return False, f"SPACE_BEFORE_PUNCTUATION: {punct}"
        
        # Check for no space after punctuation
        if any(punct in text and text.index(punct) + 1 < len(text) and not text[text.index(punct)+1] == " " 
               for punct in self.punctuation if punct in text):
            return False, "MISSING_SPACE_AFTER_PUNCTUATION"
        
        return True, "VALID_SPACING"
    
    def validate_punctuation(self, text):
        """Validate proper Ge'ez punctuation usage"""
        # Check for English punctuation
        if any(c in text for c in [".", ",", "!", "?", ":", ";"]):
            return False, "ENGLISH_PUNCTUATION"
        
        # Check for proper full stop at end
        if not text.strip().endswith("።") and not text.strip().endswith("። "):
            return False, "MISSING_FULL_STOP"
        
        return True, "VALID_PUNCTUATION"
    
    def validate_cultural_safety(self, text, context=""):
        """Validate cultural appropriateness"""
        for term, rules in self.cultural_rules.items():
            if term in text:
                # Check for taboo associations
                if "taboo_associations" in rules:
                    for taboo in rules["taboo_associations"]:
                        if taboo in text.lower():
                            return False, f"CULTURAL_TABOO: {term} + {taboo}"
        
        return True, "CULTURALLY_SAFE"
    
    def validate(self, text, context=""):
        """Run all validation checks"""
        # Check spacing
        valid, reason = self.validate_spacing(text)
        if not valid:
            return False, reason
        
        # Check punctuation
        valid, reason = self.validate_punctuation(text)
        if not valid:
            return False, reason
        
        # Check cultural safety
        valid, reason = self.validate_cultural_safety(text, context)
        if not valid:
            return False, reason
        
        return True, "VALID_AMHARIC"
    
    def apply_guardrails(self, text, context=""):
        """Apply cultural guardrails to output"""
        for term, rules in self.cultural_rules.items():
            if term in text:
                # Replace taboo outputs
                if "taboo_associations" in rules:
                    for taboo in rules["taboo_associations"]:
                        if taboo in text.lower():
                            return rules["sacred_context"]
        
        return text

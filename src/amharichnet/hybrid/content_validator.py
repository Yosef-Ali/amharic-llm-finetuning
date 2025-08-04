"""
Content Validator for Amharic Text
Real-time validation of generated content using extraction
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..extraction.schemas import get_schema_by_domain


class ValidationLevel(Enum):
    """Validation strictness levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"


class IssueType(Enum):
    """Types of validation issues."""
    LENGTH = "length"
    ENTITIES = "entities"
    RELATIONSHIPS = "relationships"
    DOMAIN_COMPLIANCE = "domain_compliance"
    QUALITY = "quality"
    COHERENCE = "coherence"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    issue_type: IssueType
    severity: str  # "error", "warning", "info"
    message: str
    suggestion: str = ""
    character_position: Optional[Tuple[int, int]] = None


@dataclass
class ValidationResult:
    """Result of content validation."""
    is_valid: bool
    overall_score: float
    issues: List[ValidationIssue]
    suggestions: List[str]
    validation_time: float
    
    # Detailed scores
    length_score: float = 0.0
    entity_score: float = 0.0
    relationship_score: float = 0.0
    domain_score: float = 0.0
    quality_score: float = 0.0
    coherence_score: float = 0.0


class ContentValidator:
    """Validates Amharic content against extraction schemas and quality metrics."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.amharic_patterns = self._initialize_amharic_patterns()
        self.domain_keywords = self._initialize_domain_keywords()
    
    def _initialize_amharic_patterns(self) -> Dict[str, List[str]]:
        """Initialize Amharic language patterns."""
        return {
            "formal_titles": [r"ዶ/ር\s+\w+", r"ፕ/ር\s+\w+", r"ወ/ሮ\s+\w+", r"አቶ\s+\w+"],
            "organizations": [r"\w+\s+ሚኒስቴር", r"\w+\s+ዩኒቨርስቲ", r"\w+\s+ድርጅት", r"\w+\s+ኩባንያ"],
            "locations": [r"አዲስ አበባ", r"ባህር ዳር", r"መቀሌ", r"ጎንደር", r"\w+\s+ክልል"],
            "dates": [r"\d+\s+ዓ\.ም", r"\w+\s+\d+", r"\d+/\d+/\d+", r"በ\w+\s+\d+"],
            "sentence_endings": [r"።", r"!", r"?", r"፡"],
            "conjunctions": [r"እና", r"ወይም", r"ነገር ግን", r"ስለዚህ", r"ከዚያም"]
        }
    
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize domain-specific keywords."""
        return {
            "news": ["ዜና", "ሪፖርት", "መረጃ", "ክንውን", "ስብሰባ", "ውይይት"],
            "government": ["መንግሥት", "ፖሊሲ", "ሕግ", "አዋጅ", "ሚኒስቴር", "ክልል"],
            "education": ["ትምህርት", "ዩኒቨርስቲ", "ተማሪ", "መምህር", "ምርምር", "ዲግሪ"],
            "healthcare": ["ጤና", "ሕክምና", "ሆስፒታል", "ዶክተር", "በሽታ", "ሕክምና"],
            "culture": ["ባህል", "ወግ", "በዓል", "ስርዓት", "ሙዚቃ", "ጥበብ"]
        }
    
    def validate_content(self, 
                        text: str,
                        domain: str = "news",
                        extraction_result: Optional[Dict[str, Any]] = None,
                        target_length: Optional[Tuple[int, int]] = None) -> ValidationResult:
        """Validate content comprehensively."""
        
        start_time = time.time()
        issues = []
        suggestions = []
        
        # Set target length if not provided
        if target_length is None:
            target_length = self._get_default_length_range(domain)
        
        # Perform different validation checks
        length_score = self._validate_length(text, target_length, issues, suggestions)
        entity_score = self._validate_entities(text, domain, extraction_result, issues, suggestions)
        relationship_score = self._validate_relationships(text, domain, extraction_result, issues, suggestions)
        domain_score = self._validate_domain_compliance(text, domain, issues, suggestions)
        quality_score = self._validate_quality(text, issues, suggestions)
        coherence_score = self._validate_coherence(text, issues, suggestions)
        
        # Calculate overall score
        scores = [length_score, entity_score, relationship_score, domain_score, quality_score, coherence_score]
        weights = [0.15, 0.25, 0.15, 0.20, 0.15, 0.10]  # Entity and domain most important
        
        overall_score = sum(score * weight for score, weight in zip(scores, weights))
        
        # Determine validity based on validation level
        validity_threshold = {
            ValidationLevel.MINIMAL: 0.6,
            ValidationLevel.STANDARD: 0.7,
            ValidationLevel.STRICT: 0.8
        }
        
        is_valid = overall_score >= validity_threshold[self.validation_level]
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=is_valid,
            overall_score=overall_score,
            issues=issues,
            suggestions=suggestions,
            validation_time=validation_time,
            length_score=length_score,
            entity_score=entity_score,
            relationship_score=relationship_score,
            domain_score=domain_score,
            quality_score=quality_score,
            coherence_score=coherence_score
        )
    
    def _validate_length(self, 
                        text: str, 
                        target_length: Tuple[int, int],
                        issues: List[ValidationIssue],
                        suggestions: List[str]) -> float:
        """Validate text length."""
        
        word_count = len(text.split())
        min_length, max_length = target_length
        
        if word_count < min_length:
            issues.append(ValidationIssue(
                issue_type=IssueType.LENGTH,
                severity="warning",
                message=f"ጽሑፉ በጣም አጭር ነው ({word_count} ቃላት, ቢያንስ {min_length} ያስፈልጋል)",
                suggestion="ተጨማሪ ዝርዝር መረጃ እና ምሳሌዎች ያክሉ"
            ))
            suggestions.append("ጽሑፉን በተጨማሪ ዝርዝር መረጃ ያሰፉ")
            score = max(0.3, word_count / min_length)
            
        elif word_count > max_length:
            issues.append(ValidationIssue(
                issue_type=IssueType.LENGTH,
                severity="info",
                message=f"ጽሑፉ ረጅም ነው ({word_count} ቃላት, ከ{max_length} በላይ)",
                suggestion="ዋናዎቹን ነጥቦች ላይ ተመስርተው ያሳጥሩ"
            ))
            suggestions.append("ጽሑፉን ዋና ዋና ነጥኦች ላይ በማተኮር ያሳጥሩ")
            score = max(0.7, 1.0 - (word_count - max_length) / max_length * 0.3)
            
        else:
            score = 1.0
        
        return score
    
    def _validate_entities(self, 
                          text: str,
                          domain: str,
                          extraction_result: Optional[Dict[str, Any]],
                          issues: List[ValidationIssue],
                          suggestions: List[str]) -> float:
        """Validate entity presence and quality."""
        
        score = 1.0
        
        if extraction_result:
            entities = extraction_result.get("entities", {})
            total_entities = sum(len(v) for v in entities.values())
            
            # Check minimum entity count
            min_entities = self._get_minimum_entity_count(domain)
            if total_entities < min_entities:
                issues.append(ValidationIssue(
                    issue_type=IssueType.ENTITIES,
                    severity="warning",
                    message=f"በቂ አካላት አልተገኙም ({total_entities} ከ{min_entities} ቢያንስ)",
                    suggestion="ተጨማሪ ስሞች፣ ቦታዎች እና ቀኖች ያክሉ"
                ))
                suggestions.append("በጽሑፉ ውስጥ ተጨማሪ አካላት (ስሞች፣ ቦታዎች፣ ቀኖች) ያክሉ")
                score *= max(0.4, total_entities / min_entities)
            
            # Check domain-specific entities
            schema = get_schema_by_domain(domain)
            expected_entity_types = set(schema.get("entities", {}).keys())
            found_entity_types = set(entities.keys())
            
            missing_types = expected_entity_types - found_entity_types
            if missing_types:
                missing_amharic = [self._entity_type_to_amharic(et) for et in missing_types]
                issues.append(ValidationIssue(
                    issue_type=IssueType.ENTITIES,
                    severity="warning",
                    message=f"የሚከተሉት የሚጠበቁ አካላት ስለታሳቃሁ: {', '.join(missing_amharic)}",
                    suggestion=f"እባክዎ {', '.join(missing_amharic)} ያክሉ"
                ))
                suggestions.append(f"የሚከተሉት አካላት ያክሉ: {', '.join(missing_amharic)}")
                coverage_score = len(found_entity_types) / len(expected_entity_types)
                score *= max(0.5, coverage_score)
        
        else:
            # Basic pattern-based validation
            entity_patterns_found = 0
            for pattern_type, patterns in self.amharic_patterns.items():
                if pattern_type in ["formal_titles", "organizations", "locations", "dates"]:
                    for pattern in patterns:
                        if re.search(pattern, text):
                            entity_patterns_found += 1
                            break
            
            if entity_patterns_found < 2:
                issues.append(ValidationIssue(
                    issue_type=IssueType.ENTITIES,
                    severity="warning",
                    message="በጽሑፉ ውስጥ በቂ አካላት አልተገኙም",
                    suggestion="ስሞች፣ ተቋማት፣ ቦታዎች እና ቀኖች ያክሉ"
                ))
                suggestions.append("በጽሑፉ ውስጥ ተጨማሪ አካላት ያክሉ")
                score *= max(0.5, entity_patterns_found / 4)
        
        return score
    
    def _validate_relationships(self, 
                               text: str,
                               domain: str,
                               extraction_result: Optional[Dict[str, Any]],
                               issues: List[ValidationIssue],
                               suggestions: List[str]) -> float:
        """Validate relationships between entities."""
        
        score = 1.0
        
        if extraction_result:
            relationships = extraction_result.get("relationships", [])
            
            # Check for basic relationships
            if len(relationships) == 0:
                # Look for relationship indicators in text
                relationship_indicators = [
                    r"\w+\s+በ\w+\s+ይሰራል",  # works at
                    r"\w+\s+በ\w+\s+ይገኛል",  # located in
                    r"በ\w+\s+\w+\s+ተከስቷል", # happened at
                    r"\w+\s+እና\s+\w+",      # basic connections
                ]
                
                indicators_found = 0
                for pattern in relationship_indicators:
                    if re.search(pattern, text):
                        indicators_found += 1
                
                if indicators_found == 0:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.RELATIONSHIPS,
                        severity="info",
                        message="በአካላት መካከል ግንኙነት አልተገኘም",
                        suggestion="የአካላት መካከል ያለውን ግንኙነት ይግለጹ"
                    ))
                    suggestions.append("በሰዎች፣ ቦታዎች እና ተቋማት መካከል ያለውን ግንኙነት ይግለጹ")
                    score *= 0.7
        
        return score
    
    def _validate_domain_compliance(self, 
                                   text: str,
                                   domain: str,
                                   issues: List[ValidationIssue],
                                   suggestions: List[str]) -> float:
        """Validate compliance with domain expectations."""
        
        domain_keywords = self.domain_keywords.get(domain, [])
        if not domain_keywords:
            return 1.0
        
        # Check for domain-specific keywords
        keywords_found = 0
        for keyword in domain_keywords:
            if keyword in text:
                keywords_found += 1
        
        keyword_score = min(1.0, keywords_found / len(domain_keywords) * 2)  # Allow 50% coverage for full score
        
        if keyword_score < 0.5:
            domain_amharic = {
                "news": "ዜና",
                "government": "መንግሥት", 
                "education": "ትምህርት",
                "healthcare": "ጤና",
                "culture": "ባህል"
            }.get(domain, domain)
            
            issues.append(ValidationIssue(
                issue_type=IssueType.DOMAIN_COMPLIANCE,
                severity="warning",
                message=f"ጽሑፉ የ{domain_amharic} ዓይነት አይመስልም",
                suggestion=f"ከ{domain_amharic} ጋር የሚገናኙ ቃላት እና ጭብጦች ያክሉ"
            ))
            suggestions.append(f"ከ{domain_amharic} ዘርፍ ጋር የሚገናኙ ቃላት ያክሉ")
        
        return keyword_score
    
    def _validate_quality(self, 
                         text: str,
                         issues: List[ValidationIssue],
                         suggestions: List[str]) -> float:
        """Validate general text quality."""
        
        score = 1.0
        
        # Check sentence structure
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            issues.append(ValidationIssue(
                issue_type=IssueType.QUALITY,
                severity="warning",  
                message="ጽሑፉ በጣም ጥቂት ዓረፍተ ነገሮች አሉት",
                suggestion="ተጨማሪ ዓረፍተ ነገሮች ያክሉ"
            ))
            suggestions.append("ጽሑፉን በበርካታ ዓረፍተ ነገሮች ያሰፉ")
            score *= 0.7
        
        # Check for proper sentence endings
        sentences_with_endings = 0
        ending_patterns = self.amharic_patterns["sentence_endings"]
        
        for sentence in sentences:
            for pattern in ending_patterns:
                if sentence.strip().endswith(pattern):
                    sentences_with_endings += 1
                    break
        
        if len(sentences) > 0:
            ending_score = sentences_with_endings / len(sentences)
            if ending_score < 0.8:
                issues.append(ValidationIssue(
                    issue_type=IssueType.QUALITY,
                    severity="info",
                    message="አንዳንድ ዓረፍተ ነገሮች ትክክለኛ መደምደሚያ የላቸውም",
                    suggestion="ዓረፍተ ነገሮችን በ፣ ። ወይም ! ያጠናቀቁ"
                ))
                score *= max(0.8, ending_score)
        
        # Check for repetitive content
        words = text.split()
        unique_words = set(words)
        if len(words) > 20:  # Only check for longer texts
            diversity_ratio = len(unique_words) / len(words)
            if diversity_ratio < 0.5:
                issues.append(ValidationIssue(
                    issue_type=IssueType.QUALITY,
                    severity="info",
                    message="ጽሑፉ ተደጋጋሚ ቃላት ይዟል",
                    suggestion="የተለያዩ ቃላት እና አገላለጾች ይጠቀሙ"
                ))
                suggestions.append("የተለያዩ ቃላት እና አገላለጾች በመጠቀም ጽሑፉን ያብዛዙ")
                score *= max(0.7, diversity_ratio / 0.5)  # Scale to 0.5 = 1.0 score
        
        return score
    
    def _validate_coherence(self, 
                           text: str,
                           issues: List[ValidationIssue],
                           suggestions: List[str]) -> float:
        """Validate text coherence and flow."""
        
        score = 1.0
        
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return score
        
        # Check for conjunctions and transitions
        conjunctions = self.amharic_patterns["conjunctions"]
        conjunction_count = 0
        
        for conjunction in conjunctions:
            conjunction_count += len(re.findall(conjunction, text))
        
        # Expect at least one conjunction per 3 sentences
        expected_conjunctions = max(1, len(sentences) // 3)
        if conjunction_count < expected_conjunctions:
            issues.append(ValidationIssue(
                issue_type=IssueType.COHERENCE,
                severity="info",
                message="ዓረፍተ ነገሮች በበቂ ሁኔታ አልተገናኙም",
                suggestion="የሚከተሉትን ማገናኛ ቃላት ይጠቀሙ: እና፣ ነገር ግን፣ ስለዚህ"
            ))
            suggestions.append("በዓረፍተ ነገሮች መካከል ማገናኛ ቃላት ይጠቀሙ")
            score *= max(0.7, conjunction_count / expected_conjunctions)
        
        return score
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split on Amharic sentence endings
        sentences = re.split(r'[።!?፡]\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_default_length_range(self, domain: str) -> Tuple[int, int]:
        """Get default length range for domain."""
        ranges = {
            "news": (100, 300),
            "government": (150, 400),
            "education": (120, 350),
            "healthcare": (100, 300),
            "culture": (80, 250)
        }
        return ranges.get(domain, (100, 300))
    
    def _get_minimum_entity_count(self, domain: str) -> int:
        """Get minimum expected entity count for domain."""
        counts = {
            "news": 4,
            "government": 3,
            "education": 3,
            "healthcare": 3,
            "culture": 2
        }
        return counts.get(domain, 3)
    
    def _entity_type_to_amharic(self, entity_type: str) -> str:
        """Convert entity type to Amharic name."""
        mappings = {
            "people": "ሰዎች",
            "organizations": "ድርጅቶች",
            "locations": "ቦታዎች",
            "dates": "ቀኖች",
            "officials": "ሃላፊዎች",
            "regions": "ክልሎች",
            "policies": "ፖሊሲዎች",
            "laws": "ሕጎች"
        }
        return mappings.get(entity_type, entity_type)
    
    def generate_improvement_report(self, validation_result: ValidationResult) -> str:
        """Generate a detailed improvement report."""
        
        report_lines = []
        report_lines.append("📊 የጽሑፍ ግምገማ ሪፖርት")
        report_lines.append("=" * 50)
        
        # Overall assessment
        if validation_result.is_valid:
            report_lines.append("✅ ጽሑፉ ጥራት ያለው ነው")
        else:
            report_lines.append("⚠️  ጽሑፉ ማሻሻያ ያስፈልገዋል")
        
        report_lines.append(f"🎯 አጠቃላይ ውጤት: {validation_result.overall_score:.2f}")
        report_lines.append("")
        
        # Detailed scores
        report_lines.append("📋 ዝርዝር ውጤቶች:")
        report_lines.append(f"   ርዝመት: {validation_result.length_score:.2f}")
        report_lines.append(f"   አካላት: {validation_result.entity_score:.2f}")
        report_lines.append(f"   ግንኙነቶች: {validation_result.relationship_score:.2f}")
        report_lines.append(f"   ዓይነት ተዛማጅነት: {validation_result.domain_score:.2f}")
        report_lines.append(f"   ጥራት: {validation_result.quality_score:.2f}")
        report_lines.append(f"   ቅንጅት: {validation_result.coherence_score:.2f}")
        report_lines.append("")
        
        # Issues
        if validation_result.issues:
            report_lines.append("⚠️  የተገኙ ችግሮች:")
            for issue in validation_result.issues:
                severity_icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "•")
                report_lines.append(f"   {severity_icon} {issue.message}")
                if issue.suggestion:
                    report_lines.append(f"      💡 {issue.suggestion}")
            report_lines.append("")
        
        # Suggestions
        if validation_result.suggestions:
            report_lines.append("💡 ማሻሻያ ሀሳቦች:")
            for suggestion in validation_result.suggestions:
                report_lines.append(f"   • {suggestion}")
            report_lines.append("")
        
        report_lines.append(f"⏱️  ግምገማ ጊዜ: {validation_result.validation_time:.3f} ሰከንድ")
        
        return "\n".join(report_lines)
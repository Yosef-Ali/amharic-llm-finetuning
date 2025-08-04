"""
Schema-Aware Text Generation
Generate Amharic text that conforms to information extraction schemas
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..extraction.schemas import get_schema_by_domain, AMHARIC_ENTITY_TYPES


@dataclass
class GenerationConstraints:
    """Constraints for schema-aware generation."""
    required_entities: List[str] = None
    minimum_entity_count: Dict[str, int] = None
    required_relationships: List[str] = None
    target_text_length: Tuple[int, int] = (100, 500)  # min, max
    domain_keywords: List[str] = None
    
    def __post_init__(self):
        if self.required_entities is None:
            self.required_entities = []
        if self.minimum_entity_count is None:
            self.minimum_entity_count = {}
        if self.required_relationships is None:
            self.required_relationships = []
        if self.domain_keywords is None:
            self.domain_keywords = []


class SchemaAwareGenerator:
    """Generates text that conforms to extraction schemas."""
    
    def __init__(self):
        self.entity_patterns = AMHARIC_ENTITY_TYPES
        self.template_cache = {}
    
    def create_guided_prompt(self, 
                           original_prompt: str,
                           domain: str,
                           constraints: Optional[GenerationConstraints] = None) -> str:
        """Create a prompt that guides generation based on schema constraints."""
        
        schema = get_schema_by_domain(domain)
        constraints = constraints or GenerationConstraints()
        
        # Build domain-specific guidance
        domain_info = schema.get("description", f"{domain} domain content")
        
        # Entity guidance
        entity_guidance = self._build_entity_guidance(schema, constraints)
        
        # Relationship guidance  
        relationship_guidance = self._build_relationship_guidance(schema, constraints)
        
        # Length guidance
        min_len, max_len = constraints.target_text_length
        length_guidance = f"በ{min_len} እና {max_len} ቃላት መካከል"
        
        # Construct guided prompt
        guided_prompt = f"""
{original_prompt}

📋 የጽሑፍ መመሪያዎች:
• ዓይነት: {domain_info}
• ርዝመት: {length_guidance}
{entity_guidance}
{relationship_guidance}

እባክዎ ዝርዝር እና ትክክለኛ መረጃ ያለው ጽሑፍ ይፍጠሩ።
"""
        
        return guided_prompt.strip()
    
    def _build_entity_guidance(self, 
                              schema: Dict[str, Any], 
                              constraints: GenerationConstraints) -> str:
        """Build entity-specific guidance text."""
        
        guidance_lines = []
        
        # Required entities
        if constraints.required_entities:
            entity_names = [self._get_entity_amharic_name(ent) for ent in constraints.required_entities]
            guidance_lines.append(f"• አስፈላጊ አካላት: {', '.join(entity_names)}")
        
        # Schema entities
        schema_entities = schema.get("entities", {})
        if schema_entities and not constraints.required_entities:
            entity_descriptions = []
            for entity_type, entity_config in list(schema_entities.items())[:3]:  # Top 3
                amharic_desc = entity_config.get("amharic_desc", entity_config.get("description", entity_type))
                entity_descriptions.append(amharic_desc)
            
            if entity_descriptions:
                guidance_lines.append(f"• ያካትቱ: {', '.join(entity_descriptions)}")
        
        # Minimum counts
        if constraints.minimum_entity_count:
            count_guidance = []
            for entity_type, min_count in constraints.minimum_entity_count.items():
                entity_name = self._get_entity_amharic_name(entity_type)
                count_guidance.append(f"{entity_name} (ቢያንስ {min_count})")
            
            if count_guidance:
                guidance_lines.append(f"• ቁጥር: {', '.join(count_guidance)}")
        
        return '\n'.join(guidance_lines) if guidance_lines else ""
    
    def _build_relationship_guidance(self, 
                                   schema: Dict[str, Any], 
                                   constraints: GenerationConstraints) -> str:
        """Build relationship guidance text."""
        
        guidance_lines = []
        
        # Required relationships
        if constraints.required_relationships:
            rel_names = [self._get_relationship_amharic_name(schema, rel) for rel in constraints.required_relationships]
            guidance_lines.append(f"• ግንኙነቶች: {', '.join(rel_names)}")
        
        # Schema relationships  
        schema_relationships = schema.get("relationships", {})
        if schema_relationships and not constraints.required_relationships:
            rel_descriptions = []
            for rel_type, rel_config in list(schema_relationships.items())[:2]:  # Top 2
                amharic_desc = rel_config.get("amharic_desc", rel_config.get("description", rel_type))
                rel_descriptions.append(amharic_desc)
            
            if rel_descriptions:
                guidance_lines.append(f"• ግንኙነቶች: {', '.join(rel_descriptions)}")
        
        return '\n'.join(guidance_lines) if guidance_lines else ""
    
    def _get_entity_amharic_name(self, entity_type: str) -> str:
        """Get Amharic name for entity type."""
        
        # Check in AMHARIC_ENTITY_TYPES
        if entity_type.upper() in self.entity_patterns:
            return self.entity_patterns[entity_type.upper()].get("amharic", entity_type)
        
        # Common mappings
        mappings = {
            "people": "ሰዎች",
            "persons": "ሰዎች", 
            "organizations": "ድርጅቶች",
            "locations": "ቦታዎች",
            "places": "ቦታዎች",
            "dates": "ቀኖች",
            "times": "ሰዓቶች",
            "officials": "ሃላፊዎች",
            "regions": "ክልሎች",
            "policies": "ፖሊሲዎች",
            "laws": "ሕጎች"
        }
        
        return mappings.get(entity_type.lower(), entity_type)
    
    def _get_relationship_amharic_name(self, schema: Dict[str, Any], rel_type: str) -> str:
        """Get Amharic name for relationship type."""
        
        schema_relationships = schema.get("relationships", {})
        if rel_type in schema_relationships:
            return schema_relationships[rel_type].get("amharic_desc", rel_type)
        
        # Common mappings
        mappings = {
            "works_at": "የሚሰራበት",
            "located_in": "የሚገኝበት", 
            "happened_on": "የተከሰተበት",
            "member_of": "አባል የሆነበት",
            "leads": "የሚመራው"
        }
        
        return mappings.get(rel_type, rel_type)
    
    def generate_template_variations(self, 
                                   domain: str,
                                   base_prompt: str,
                                   variation_count: int = 3) -> List[str]:
        """Generate multiple prompt variations for the same domain."""
        
        schema = get_schema_by_domain(domain)
        variations = []
        
        # Create different constraint sets
        constraint_sets = self._create_constraint_variations(schema, variation_count)
        
        for i, constraints in enumerate(constraint_sets):
            variation_prompt = self.create_guided_prompt(
                f"{base_prompt} (ዋሪያሽን {i+1})",
                domain,
                constraints
            )
            variations.append(variation_prompt)
        
        return variations
    
    def _create_constraint_variations(self, 
                                    schema: Dict[str, Any], 
                                    count: int) -> List[GenerationConstraints]:
        """Create different constraint variations for a schema."""
        
        entities = list(schema.get("entities", {}).keys())
        relationships = list(schema.get("relationships", {}).keys())
        
        variations = []
        
        for i in range(count):
            if i == 0:
                # Minimal constraints
                constraints = GenerationConstraints(
                    target_text_length=(150, 300)
                )
            elif i == 1:
                # Medium constraints
                constraints = GenerationConstraints(
                    required_entities=entities[:2] if len(entities) >= 2 else entities,
                    target_text_length=(200, 400),
                    minimum_entity_count={entities[0]: 2} if entities else {}
                )
            else:
                # Maximum constraints
                constraints = GenerationConstraints(
                    required_entities=entities[:3] if len(entities) >= 3 else entities,
                    required_relationships=relationships[:1] if relationships else [],
                    target_text_length=(300, 500),
                    minimum_entity_count={
                        entities[0]: 2,
                        entities[1]: 1 if len(entities) > 1 else None
                    } if entities else {}
                )
            
            variations.append(constraints)
        
        return variations
    
    def validate_generated_text(self, 
                               text: str,
                               domain: str,
                               constraints: Optional[GenerationConstraints] = None) -> Dict[str, Any]:
        """Validate generated text against schema constraints."""
        
        validation_result = {
            "valid": True,
            "issues": [],
            "suggestions": [],
            "score": 1.0
        }
        
        if not constraints:
            return validation_result
        
        score_deductions = 0.0
        
        # Check text length
        text_length = len(text.split())
        min_len, max_len = constraints.target_text_length
        
        if text_length < min_len:
            validation_result["issues"].append(f"ጽሑፉ በጣም አጭር ነው ({text_length} ቃላት, ቢያንስ {min_len} ያስፈልጋል)")
            validation_result["suggestions"].append("ተጨማሪ ዝርዝር መረጃ ያክሉ")
            score_deductions += 0.2
        elif text_length > max_len:
            validation_result["issues"].append(f"ጽሑፉ በጣም ረጅም ነው ({text_length} ቃላት, ከ{max_len} በላይ)")
            validation_result["suggestions"].append("ጽሑፉን ያሳጥሩ")
            score_deductions += 0.1
        
        # Check required entities (basic pattern matching)
        if constraints.required_entities:
            missing_entities = []
            for entity_type in constraints.required_entities:
                if not self._text_contains_entity_pattern(text, entity_type):
                    missing_entities.append(self._get_entity_amharic_name(entity_type))
            
            if missing_entities:
                validation_result["issues"].append(f"የሚከተሉት አካላት ስለታሳቃሁ: {', '.join(missing_entities)}")
                validation_result["suggestions"].append(f"እባክዎ {', '.join(missing_entities)} ያክሉ")
                score_deductions += 0.3 * len(missing_entities) / len(constraints.required_entities)
        
        # Check minimum entity counts
        if constraints.minimum_entity_count:
            for entity_type, min_count in constraints.minimum_entity_count.items():
                actual_count = self._count_entity_occurrences(text, entity_type)
                if actual_count < min_count:
                    entity_name = self._get_entity_amharic_name(entity_type)
                    validation_result["issues"].append(
                        f"{entity_name} በቂ አይደለም ({actual_count} ከ{min_count} ቢያንስ)"
                    )
                    validation_result["suggestions"].append(f"ተጨማሪ {entity_name} ያክሉ")
                    score_deductions += 0.2
        
        # Calculate final score
        validation_result["score"] = max(0.0, 1.0 - score_deductions)
        validation_result["valid"] = validation_result["score"] >= 0.7
        
        return validation_result
    
    def _text_contains_entity_pattern(self, text: str, entity_type: str) -> bool:
        """Check if text contains patterns for specific entity type."""
        
        # Get patterns for entity type
        patterns = []
        
        if entity_type.upper() in self.entity_patterns:
            patterns = self.entity_patterns[entity_type.upper()].get("patterns", [])
        
        # Add common patterns based on entity type
        common_patterns = {
            "people": [r"ዶ/ር\s+\w+", r"ፕ/ር\s+\w+", r"ወ/ሮ\s+\w+", r"አቶ\s+\w+"],
            "organizations": [r"\w+\s+ሚኒስቴር", r"\w+\s+ዩኒቨርስቲ", r"\w+\s+ድርጅት"],
            "locations": [r"አዲስ አበባ", r"ባህር ዳር", r"መቀሌ", r"\w+\s+ክልል"],
            "dates": [r"\d+\s+ዓ\.ም", r"\w+\s+\d+", r"\d+/\d+/\d+"]
        }
        
        if entity_type.lower() in common_patterns:
            patterns.extend(common_patterns[entity_type.lower()])
        
        # Check if any pattern matches
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _count_entity_occurrences(self, text: str, entity_type: str) -> int:
        """Count occurrences of entity patterns in text."""
        
        count = 0
        
        # Get patterns for entity type
        patterns = []
        
        if entity_type.upper() in self.entity_patterns:
            patterns = self.entity_patterns[entity_type.upper()].get("patterns", [])
        
        # Add common patterns
        common_patterns = {
            "people": [r"ዶ/ር\s+\w+", r"ፕ/ር\s+\w+", r"ወ/ሮ\s+\w+", r"አቶ\s+\w+"],
            "organizations": [r"\w+\s+ሚኒስቴር", r"\w+\s+ዩኒቨርስቲ", r"\w+\s+ድርጅት"],
            "locations": [r"አዲስ አበባ", r"ባህር ዳር", r"መቀሌ", r"\w+\s+ክልል"],
            "dates": [r"\d+\s+ዓ\.ም", r"\w+\s+\d+", r"\d+/\d+/\d+"]
        }
        
        if entity_type.lower() in common_patterns:
            patterns.extend(common_patterns[entity_type.lower()])
        
        # Count matches for all patterns
        for pattern in patterns:
            matches = re.findall(pattern, text)
            count += len(matches)
        
        return count
    
    def create_domain_specific_templates(self, domain: str) -> Dict[str, str]:
        """Create template prompts for specific domain."""
        
        schema = get_schema_by_domain(domain)
        templates = {}
        
        if domain == "news":
            templates.update({
                "breaking_news": "በዛሬው ቀን በ{location} ውስጥ {event} ተከስቷል። {official} ይህንን ዜና አስታውቀዋል።",
                "government_meeting": "{official} በ{location} ወዳለ {organization} ገብተው ከ{people} ጋር ስብሰባ አካሂደዋል።",
                "announcement": "{organization} በ{date} {policy} እንደሚተገበር አስታውቋል። ይህ ዕቅድ {region} ውስጥ ይሰራል።"
            })
        
        elif domain == "government":
            templates.update({
                "policy_announcement": "{official} አዲስ {policy} አስታውቀዋል። ይህ ፖሊሲ በ{region} ውስጥ ይተገበራል።",
                "law_proclamation": "ህዝብ ተወካዮች ምክር ቤት {law} አወጣ። አዋጁ በ{date} ይሰራል።",
                "administrative_decision": "{region} ከፍተኛ ሃላፊዎች {decision} ወስነዋል። ይህ ውሳኔ {people} ላይ ተጽእኖ ያሳድራል።"
            })
        
        elif domain == "education":
            templates.update({
                "academic_news": "{institution} በዚህ ዓመት {program} ጀምሯል। {official} ፕሮግራሙን ይመሩታል።",
                "research_announcement": "{researcher} በ{institution} ውስጥ {research_topic} ላይ ጥናት አካሂደዋል።",
                "graduation_news": "{institution} በ{date} የ{degree} ተመራቂዎችን አስመርቋል።"
            })
        
        return templates
    
    def apply_template(self, 
                      template: str, 
                      values: Dict[str, str]) -> str:
        """Apply values to a template."""
        
        result = template
        
        for key, value in values.items():
            placeholder = "{" + key + "}"
            result = result.replace(placeholder, value)
        
        return result
    
    def suggest_improvements(self, 
                           text: str,
                           domain: str,
                           extraction_result: Optional[Dict[str, Any]] = None) -> List[str]:
        """Suggest improvements based on extraction results."""
        
        suggestions = []
        
        # Basic length check
        word_count = len(text.split())
        if word_count < 100:
            suggestions.append("ጽሑፉን ያሰፋ - በበለጠ ዝርዝር መረጃ ያሙሉ")
        elif word_count > 400:
            suggestions.append("ጽሑፉን ያሳጥሩ - ዋናዎቹን ነጥቦች ብቻ ያተኩሩ")
        
        # Entity-based suggestions
        if extraction_result:
            entities = extraction_result.get("entities", {})
            total_entities = sum(len(v) for v in entities.values())
            
            if total_entities < 3:
                suggestions.append("ተጨማሪ ስሞች፣ ቦታዎች እና ቀኖች ያክሉ")
            
            # Domain-specific suggestions
            if domain == "news":
                if "people" not in entities or len(entities.get("people", [])) == 0:
                    suggestions.append("በዜናው ውስጥ ሰዎችን ይጥቀሱ")
                if "locations" not in entities:
                    suggestions.append("ክንውኑ የተከሰተበትን ቦታ ይግለጹ")
                if "dates" not in entities:  
                    suggestions.append("ክንውኑ የተከሰተበትን ቀን ይጥቀሱ")
            
            elif domain == "government":
                if "officials" not in entities:
                    suggestions.append("የመንግሥት ሃላፊዎችን ይጥቀሱ")
                if "regions" not in entities:
                    suggestions.append("ተዛማጁ ክልሎችን ይጥቀሱ")
        
        return suggestions
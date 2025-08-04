"""
Amharic Information Extractor
LangExtract integration for Amharic text processing
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import asyncio
from pathlib import Path

try:
    import langextract
    from langextract import LangExtract
    import google.generativeai as genai
except ImportError as e:
    print(f"⚠️  LangExtract dependencies not available: {e}")
    langextract = None
    LangExtract = None
    genai = None

from .schemas import AMHARIC_SCHEMAS, AMHARIC_EXAMPLES, get_schema_by_domain, get_examples_by_domain


@dataclass
class ExtractionResult:
    """Result of Amharic information extraction."""
    text: str
    domain: str
    entities: Dict[str, List[str]]
    relationships: List[Dict[str, str]]
    events: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    character_spans: Dict[str, List[tuple]]  # Source grounding
    metadata: Dict[str, Any]


@dataclass
class AmharicExtractionConfig:
    """Configuration for Amharic extraction."""
    api_key: Optional[str] = None
    model_name: str = "gemini-1.5-flash"
    max_context_length: int = 32000
    temperature: float = 0.1
    use_few_shot: bool = True
    enable_source_grounding: bool = True
    batch_size: int = 5
    timeout_seconds: int = 30


class AmharicExtractionSchemas:
    """Manages Amharic extraction schemas and examples."""
    
    def __init__(self):
        self.schemas = AMHARIC_SCHEMAS
        self.examples = AMHARIC_EXAMPLES
    
    def get_schema(self, domain: str) -> Dict[str, Any]:
        """Get schema for specific domain."""
        return get_schema_by_domain(domain)
    
    def get_examples(self, domain: str) -> Dict[str, Any]:
        """Get few-shot examples for domain."""
        return get_examples_by_domain(domain)
    
    def create_extraction_prompt(self, domain: str, text: str) -> str:
        """Create extraction prompt for LangExtract."""
        schema = self.get_schema(domain)
        examples = self.get_examples(domain)
        
        prompt = f"""
Extract structured information from the following Amharic text in the {domain} domain.

Schema for {domain}:
{json.dumps(schema, indent=2, ensure_ascii=False)}

Example extraction:
Input: {examples['input_text']}
Output: {json.dumps(examples['expected_output'], indent=2, ensure_ascii=False)}

Now extract information from this text:
{text}

Return the results in the same JSON format as the example, ensuring all Amharic text is preserved correctly.
Include character span information for source grounding.
"""
        return prompt
    
    def validate_extraction(self, result: Dict[str, Any], domain: str) -> bool:
        """Validate extraction result against schema."""
        schema = self.get_schema(domain)
        
        # Check if required entity types are present
        required_entities = schema.get("entities", {}).keys()
        extracted_entities = result.get("entities", {}).keys() if isinstance(result.get("entities"), dict) else []
        
        # At least some entities should be extracted
        return len(set(required_entities).intersection(set(extracted_entities))) > 0


class AmharicExtractor:
    """Main Amharic information extraction class."""
    
    def __init__(self, config: Optional[AmharicExtractionConfig] = None):
        self.config = config or AmharicExtractionConfig()
        self.schemas = AmharicExtractionSchemas()
        self.extractor = None
        self._setup_api()
    
    def _setup_api(self):
        """Setup Google Gemini API and LangExtract."""
        if not langextract or not genai:
            print("⚠️  LangExtract not available - using mock extraction")
            return
        
        # Get API key from config or environment
        api_key = self.config.api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            print("⚠️  No Gemini API key found - using mock extraction")
            print("   Set GEMINI_API_KEY environment variable or pass api_key to config")
            return
        
        try:
            # Configure Gemini API
            genai.configure(api_key=api_key)
            
            # Initialize LangExtract
            self.extractor = LangExtract(
                model_name=self.config.model_name,
                temperature=self.config.temperature
            )
            
            print(f"✅ AmharicExtractor initialized with {self.config.model_name}")
            
        except Exception as e:
            print(f"⚠️  Failed to initialize LangExtract: {e}")
            self.extractor = None
    
    def extract(self, 
                text: str, 
                domain: str = "news", 
                **kwargs) -> ExtractionResult:
        """Extract information from Amharic text."""
        
        if not self.extractor:
            return self._mock_extraction(text, domain)
        
        try:
            # Create extraction prompt
            prompt = self.schemas.create_extraction_prompt(domain, text)
            
            # Perform extraction
            extraction_result = self.extractor.extract(
                text=text,
                schema=self.schemas.get_schema(domain),
                examples=self.schemas.get_examples(domain) if self.config.use_few_shot else None,
                **kwargs
            )
            
            # Process and validate results
            processed_result = self._process_extraction_result(
                extraction_result, text, domain
            )
            
            return processed_result
            
        except Exception as e:
            print(f"⚠️  Extraction failed: {e}")
            return self._mock_extraction(text, domain)
    
    def extract_batch(self, 
                     texts: List[str], 
                     domain: str = "news",
                     **kwargs) -> List[ExtractionResult]:
        """Extract information from multiple texts."""
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            batch_results = []
            for text in batch:
                result = self.extract(text, domain, **kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def extract_from_file(self, 
                         file_path: str, 
                         domain: str = "news",
                         **kwargs) -> ExtractionResult:
        """Extract information from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            return self.extract(text, domain, **kwargs)
            
        except Exception as e:
            print(f"❌ Failed to read file {file_path}: {e}")
            return self._mock_extraction("", domain)
    
    def _process_extraction_result(self, 
                                 raw_result: Any, 
                                 original_text: str, 
                                 domain: str) -> ExtractionResult:
        """Process and structure extraction result."""
        
        # Handle different result formats from LangExtract
        if isinstance(raw_result, dict):
            entities = raw_result.get("entities", {})
            relationships = raw_result.get("relationships", [])
            events = raw_result.get("events", [])
        else:
            # Parse from string if necessary
            try:
                parsed = json.loads(str(raw_result))
                entities = parsed.get("entities", {})
                relationships = parsed.get("relationships", [])
                events = parsed.get("events", [])
            except:
                entities = {}
                relationships = []
                events = []
        
        # Calculate simple confidence scores based on entity count and text length
        confidence_scores = self._calculate_confidence_scores(entities, original_text)
        
        # Extract character spans for source grounding
        character_spans = self._extract_character_spans(entities, original_text)
        
        return ExtractionResult(
            text=original_text,
            domain=domain,
            entities=entities,
            relationships=relationships,
            events=events,
            confidence_scores=confidence_scores,
            character_spans=character_spans,
            metadata={
                "extraction_method": "langextract",
                "model": self.config.model_name,
                "schema_version": "1.0",
                "text_length": len(original_text),
                "entity_count": sum(len(v) if isinstance(v, list) else 1 for v in entities.values())
            }
        )
    
    def _mock_extraction(self, text: str, domain: str) -> ExtractionResult:
        """Mock extraction for when LangExtract is not available."""
        
        # Simple rule-based extraction for demonstration
        mock_entities = {}
        mock_relationships = []
        mock_events = []
        
        # Basic entity extraction using patterns
        schema = self.schemas.get_schema(domain)
        
        for entity_type, entity_config in schema.get("entities", {}).items():
            entity_list = []
            
            # Look for basic Amharic patterns
            if entity_type == "people":
                # Look for title patterns
                import re
                patterns = [r'ዶ/ር\s+\w+', r'ፕ/ር\s+\w+', r'ወ/ሮ\s+\w+', r'አቶ\s+\w+']
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    entity_list.extend(matches)
            
            elif entity_type == "locations":
                # Look for common Ethiopian place names
                places = ["አዲስ አበባ", "ባህር ዳር", "መቀሌ", "ጅማ", "ጎንደር", "ሃረር", "አክሱም"]
                for place in places:
                    if place in text:
                        entity_list.append(place)
            
            elif entity_type == "dates":
                # Look for Amharic date patterns
                import re
                date_patterns = [r'\d+\s+ዓ\.ም', r'\w+\s+\d+', r'\d+/\d+/\d+']
                for pattern in date_patterns:
                    matches = re.findall(pattern, text)
                    entity_list.extend(matches)
            
            if entity_list:
                mock_entities[entity_type] = list(set(entity_list))
        
        # Simple confidence calculation
        confidence_scores = self._calculate_confidence_scores(mock_entities, text)
        character_spans = self._extract_character_spans(mock_entities, text)
        
        return ExtractionResult(
            text=text,
            domain=domain,
            entities=mock_entities,
            relationships=mock_relationships,
            events=mock_events,
            confidence_scores=confidence_scores,
            character_spans=character_spans,
            metadata={
                "extraction_method": "mock",
                "model": "rule_based",
                "schema_version": "1.0",
                "text_length": len(text),
                "entity_count": sum(len(v) for v in mock_entities.values())
            }
        )
    
    def _calculate_confidence_scores(self, entities: Dict[str, List[str]], text: str) -> Dict[str, float]:
        """Calculate confidence scores for extracted entities."""
        scores = {}
        
        for entity_type, entity_list in entities.items():
            if not entity_list:
                scores[entity_type] = 0.0
                continue
            
            # Base confidence on entity count and text coverage
            entity_chars = sum(len(entity) for entity in entity_list)
            text_coverage = min(1.0, entity_chars / len(text)) if text else 0.0
            entity_density = min(1.0, len(entity_list) / 10)  # Normalize to 10 entities max
            
            scores[entity_type] = (text_coverage + entity_density) / 2
        
        return scores
    
    def _extract_character_spans(self, entities: Dict[str, List[str]], text: str) -> Dict[str, List[tuple]]:
        """Extract character span positions for source grounding."""
        spans = {}
        
        for entity_type, entity_list in entities.items():
            entity_spans = []
            
            for entity in entity_list:
                start_pos = text.find(entity)
                if start_pos != -1:
                    end_pos = start_pos + len(entity)
                    entity_spans.append((start_pos, end_pos, entity))
            
            spans[entity_type] = entity_spans
        
        return spans
    
    def evaluate_extraction_quality(self, result: ExtractionResult) -> Dict[str, float]:
        """Evaluate the quality of extraction results."""
        
        metrics = {}
        
        # Entity coverage
        total_entities = sum(len(v) for v in result.entities.values())
        metrics["entity_count"] = total_entities
        metrics["entity_density"] = total_entities / len(result.text) * 1000  # per 1000 chars
        
        # Confidence scores
        avg_confidence = sum(result.confidence_scores.values()) / len(result.confidence_scores) if result.confidence_scores else 0.0
        metrics["average_confidence"] = avg_confidence
        
        # Schema compliance
        schema = self.schemas.get_schema(result.domain)
        expected_entity_types = set(schema.get("entities", {}).keys())
        found_entity_types = set(result.entities.keys())
        
        metrics["schema_coverage"] = len(found_entity_types.intersection(expected_entity_types)) / len(expected_entity_types) if expected_entity_types else 0.0
        
        # Relationship extraction
        metrics["relationship_count"] = len(result.relationships)
        
        # Overall quality score
        metrics["overall_quality"] = (
            min(1.0, metrics["entity_density"] / 50) * 0.3 +  # Entity density (normalized)
            metrics["average_confidence"] * 0.4 +  # Confidence
            metrics["schema_coverage"] * 0.3  # Schema compliance
        )
        
        return metrics
    
    def export_results(self, 
                      results: Union[ExtractionResult, List[ExtractionResult]], 
                      output_path: str, 
                      format: str = "json"):
        """Export extraction results to file."""
        
        if isinstance(results, ExtractionResult):
            results = [results]
        
        # Convert to serializable format
        export_data = []
        for result in results:
            export_data.append({
                "text": result.text,
                "domain": result.domain,
                "entities": result.entities,
                "relationships": result.relationships,
                "events": result.events,
                "confidence_scores": result.confidence_scores,
                "character_spans": result.character_spans,
                "metadata": result.metadata
            })
        
        # Export to file
        output_path = Path(output_path)
        
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "csv":
            import pandas as pd
            
            # Flatten data for CSV
            flat_data = []
            for item in export_data:
                base_row = {
                    "text": item["text"][:100] + "..." if len(item["text"]) > 100 else item["text"],
                    "domain": item["domain"],
                    "entity_count": item["metadata"]["entity_count"],
                    "text_length": item["metadata"]["text_length"]
                }
                
                # Add entity counts by type
                for entity_type, entities in item["entities"].items():
                    base_row[f"{entity_type}_count"] = len(entities)
                
                flat_data.append(base_row)
            
            df = pd.DataFrame(flat_data)
            df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"✅ Exported {len(results)} extraction results to {output_path}")


def create_amharic_extractor(api_key: Optional[str] = None, 
                           model_name: str = "gemini-1.5-flash") -> AmharicExtractor:
    """Factory function to create Amharic extractor."""
    config = AmharicExtractionConfig(
        api_key=api_key,
        model_name=model_name,
        use_few_shot=True,
        enable_source_grounding=True
    )
    
    return AmharicExtractor(config)
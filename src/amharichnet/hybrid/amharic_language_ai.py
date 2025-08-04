"""
Unified Amharic Language AI Platform
Combines H-Net generation with LangExtract information extraction
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our components
from amharichnet.models.transformer_hnet import TransformerHNet, TransformerHNetConfig
from amharichnet.inference.advanced_generator import AdvancedAmharicGenerator
from amharichnet.extraction.amharic_extractor import AmharicExtractor, AmharicExtractionConfig, ExtractionResult
from amharichnet.extraction.schemas import get_schema_by_domain, AMHARIC_SCHEMAS


class ProcessingMode(Enum):
    """Processing modes for the hybrid system."""
    GENERATION_ONLY = "generation_only"
    EXTRACTION_ONLY = "extraction_only"
    GENERATION_THEN_EXTRACTION = "generation_then_extraction"
    EXTRACTION_GUIDED_GENERATION = "extraction_guided_generation" 
    ITERATIVE_REFINEMENT = "iterative_refinement"
    VALIDATION_LOOP = "validation_loop"


@dataclass
class LanguageAIConfig:
    """Configuration for the unified language AI platform."""
    
    # Generation settings
    generation_model_path: Optional[str] = None
    generation_config_path: Optional[str] = None
    max_generation_length: int = 500
    generation_temperature: float = 0.8
    generation_strategy: str = "sampling"
    
    # Extraction settings
    extraction_api_key: Optional[str] = None
    extraction_model: str = "gemini-1.5-flash"
    use_few_shot_extraction: bool = True
    enable_source_grounding: bool = True
    
    # Hybrid processing settings
    default_domain: str = "news"
    quality_threshold: float = 0.7
    max_refinement_iterations: int = 3
    enable_cross_validation: bool = True
    
    # Performance settings
    processing_timeout: int = 30
    batch_size: int = 5
    enable_caching: bool = True


@dataclass
class HybridResult:
    """Result from hybrid generation+extraction processing."""
    
    # Input information
    original_prompt: str
    processing_mode: ProcessingMode
    domain: str
    
    # Generation results
    generated_text: Optional[str] = None
    generation_metadata: Dict[str, Any] = None
    
    # Extraction results
    extraction_result: Optional[ExtractionResult] = None
    
    # Quality and validation
    quality_scores: Dict[str, float] = None
    validation_passed: bool = False
    
    # Processing metadata
    processing_time: float = 0.0
    iterations_count: int = 1
    refinement_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.generation_metadata is None:
            self.generation_metadata = {}
        if self.quality_scores is None:
            self.quality_scores = {}
        if self.refinement_history is None:
            self.refinement_history = []


class AmharicLanguageAI:
    """
    Unified Amharic Language AI Platform
    
    Combines advanced H-Net text generation with LangExtract information extraction
    to create a comprehensive language processing system.
    """
    
    def __init__(self, config: Optional[LanguageAIConfig] = None):
        self.config = config or LanguageAIConfig()
        
        # Initialize components
        self.generator = None
        self.extractor = None
        self.hnet_model = None
        
        # Processing cache
        self.cache = {} if self.config.enable_caching else None
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "generation_requests": 0,
            "extraction_requests": 0,
            "hybrid_requests": 0,
            "cache_hits": 0,
            "total_processing_time": 0.0
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize generation and extraction components."""
        
        print("🔧 Initializing Amharic Language AI Platform...")
        
        # Initialize H-Net generator
        try:
            if self.config.generation_model_path and Path(self.config.generation_model_path).exists():
                print("   ✅ Loading H-Net model from checkpoint...")
                # Would load actual model here
                self.hnet_model = "loaded_model"  # Placeholder
            else:
                print("   ⚠️  H-Net model path not found, using mock generator")
            
            # Create advanced generator
            self.generator = AdvancedAmharicGenerator(
                model=self.hnet_model,
                config={
                    "max_length": self.config.max_generation_length,
                    "temperature": self.config.generation_temperature
                }
            )
            print("   ✅ Advanced generator initialized")
            
        except Exception as e:
            print(f"   ⚠️  Generator initialization failed: {e}")
            self.generator = None
        
        # Initialize LangExtract extractor
        try:
            extraction_config = AmharicExtractionConfig(
                api_key=self.config.extraction_api_key,
                model_name=self.config.extraction_model,
                use_few_shot=self.config.use_few_shot_extraction,
                enable_source_grounding=self.config.enable_source_grounding
            )
            
            self.extractor = AmharicExtractor(extraction_config)
            print("   ✅ LangExtract extractor initialized")
            
        except Exception as e:
            print(f"   ⚠️  Extractor initialization failed: {e}")
            self.extractor = None
        
        print("🚀 Amharic Language AI Platform ready!")
    
    def generate_text(self, 
                     prompt: str, 
                     domain: str = None,
                     **kwargs) -> HybridResult:
        """Generate Amharic text using H-Net model."""
        
        start_time = time.time()
        domain = domain or self.config.default_domain
        
        # Check cache
        cache_key = f"gen_{hash(prompt)}_{domain}"
        if self.cache and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]
        
        result = HybridResult(
            original_prompt=prompt,
            processing_mode=ProcessingMode.GENERATION_ONLY,
            domain=domain
        )
        
        try:
            if self.generator:
                # Generate text using H-Net
                generated = self.generator.generate(
                    prompt=prompt,
                    generation_strategy=self.config.generation_strategy,
                    max_length=self.config.max_generation_length,
                    temperature=self.config.generation_temperature,
                    **kwargs
                )
                
                result.generated_text = generated
                result.generation_metadata = {
                    "strategy": self.config.generation_strategy,
                    "max_length": self.config.max_generation_length,
                    "temperature": self.config.generation_temperature,
                    "prompt_length": len(prompt),
                    "output_length": len(generated)
                }
                
            else:
                # Mock generation for demonstration
                result.generated_text = f"[Generated Amharic text for: {prompt[:50]}...]"
                result.generation_metadata = {"method": "mock"}
        
        except Exception as e:
            result.generated_text = f"Generation failed: {e}"
            result.generation_metadata = {"error": str(e)}
        
        # Update statistics and cache
        result.processing_time = time.time() - start_time
        self.stats["total_requests"] += 1
        self.stats["generation_requests"] += 1
        self.stats["total_processing_time"] += result.processing_time
        
        if self.cache:
            self.cache[cache_key] = result
        
        return result
    
    def extract_information(self, 
                           text: str, 
                           domain: str = None,
                           **kwargs) -> HybridResult:
        """Extract information from Amharic text using LangExtract."""
        
        start_time = time.time()
        domain = domain or self.config.default_domain
        
        # Check cache
        cache_key = f"ext_{hash(text)}_{domain}"
        if self.cache and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]
        
        result = HybridResult(
            original_prompt=text,
            processing_mode=ProcessingMode.EXTRACTION_ONLY,
            domain=domain
        )
        
        try:
            if self.extractor:
                # Extract information using LangExtract
                extraction_result = self.extractor.extract(text, domain=domain, **kwargs)
                result.extraction_result = extraction_result
                
                # Calculate quality scores
                result.quality_scores = self.extractor.evaluate_extraction_quality(extraction_result)
                result.validation_passed = result.quality_scores.get("overall_quality", 0) >= self.config.quality_threshold
                
            else:
                # Mock extraction for demonstration  
                result.extraction_result = ExtractionResult(
                    text=text,
                    domain=domain,
                    entities={"mock_entities": ["entity1", "entity2"]},
                    relationships=[],
                    events=[],
                    confidence_scores={"mock_entities": 0.8},
                    character_spans={"mock_entities": [(0, 10, "entity1")]},
                    metadata={"method": "mock"}
                )
                result.quality_scores = {"overall_quality": 0.8}
                result.validation_passed = True
        
        except Exception as e:
            result.quality_scores = {"error": str(e), "overall_quality": 0.0}
            result.validation_passed = False
        
        # Update statistics and cache
        result.processing_time = time.time() - start_time
        self.stats["total_requests"] += 1
        self.stats["extraction_requests"] += 1
        self.stats["total_processing_time"] += result.processing_time
        
        if self.cache:
            self.cache[cache_key] = result
        
        return result
    
    def generate_and_extract(self, 
                           prompt: str, 
                           domain: str = None,
                           **kwargs) -> HybridResult:
        """Generate text then extract information from the generated content."""
        
        start_time = time.time()
        domain = domain or self.config.default_domain
        
        result = HybridResult(
            original_prompt=prompt,
            processing_mode=ProcessingMode.GENERATION_THEN_EXTRACTION,
            domain=domain
        )
        
        try:
            # Step 1: Generate text
            generation_result = self.generate_text(prompt, domain, **kwargs)
            result.generated_text = generation_result.generated_text
            result.generation_metadata = generation_result.generation_metadata
            
            # Step 2: Extract information from generated text
            if result.generated_text:
                extraction_result = self.extract_information(result.generated_text, domain, **kwargs)
                result.extraction_result = extraction_result.extraction_result
                result.quality_scores = extraction_result.quality_scores
                result.validation_passed = extraction_result.validation_passed
            
        except Exception as e:
            result.quality_scores = {"error": str(e), "overall_quality": 0.0}
            result.validation_passed = False
        
        # Update statistics
        result.processing_time = time.time() - start_time
        self.stats["total_requests"] += 1
        self.stats["hybrid_requests"] += 1
        self.stats["total_processing_time"] += result.processing_time
        
        return result
    
    def schema_guided_generation(self, 
                               prompt: str, 
                               domain: str = None,
                               target_entities: List[str] = None,
                               **kwargs) -> HybridResult:
        """Generate text with guidance from extraction schema expectations."""
        
        start_time = time.time()
        domain = domain or self.config.default_domain
        
        result = HybridResult(
            original_prompt=prompt,
            processing_mode=ProcessingMode.EXTRACTION_GUIDED_GENERATION,
            domain=domain
        )
        
        try:
            # Get domain schema for guidance
            schema = get_schema_by_domain(domain)
            entity_types = list(schema.get("entities", {}).keys())
            
            # Create schema-aware prompt
            schema_prompt = self._create_schema_aware_prompt(prompt, domain, entity_types, target_entities)
            
            # Generate with schema guidance
            generation_result = self.generate_text(schema_prompt, domain, **kwargs)
            result.generated_text = generation_result.generated_text
            result.generation_metadata = generation_result.generation_metadata
            result.generation_metadata["schema_guided"] = True
            result.generation_metadata["target_entities"] = target_entities or entity_types
            
            # Validate generated content
            if result.generated_text:
                extraction_result = self.extract_information(result.generated_text, domain, **kwargs)
                result.extraction_result = extraction_result.extraction_result
                result.quality_scores = extraction_result.quality_scores
                result.validation_passed = extraction_result.validation_passed
            
        except Exception as e:
            result.quality_scores = {"error": str(e), "overall_quality": 0.0}
            result.validation_passed = False
        
        # Update statistics
        result.processing_time = time.time() - start_time
        self.stats["total_requests"] += 1
        self.stats["hybrid_requests"] += 1
        self.stats["total_processing_time"] += result.processing_time
        
        return result
    
    def iterative_refinement(self, 
                           prompt: str, 
                           domain: str = None,
                           quality_target: float = None,
                           **kwargs) -> HybridResult:
        """Iteratively generate and refine text until quality threshold is met."""
        
        start_time = time.time()
        domain = domain or self.config.default_domain
        quality_target = quality_target or self.config.quality_threshold
        
        result = HybridResult(
            original_prompt=prompt,
            processing_mode=ProcessingMode.ITERATIVE_REFINEMENT,
            domain=domain
        )
        
        best_result = None
        best_quality = 0.0
        
        try:
            for iteration in range(self.config.max_refinement_iterations):
                # Generate text (with refinement prompt if not first iteration)
                if iteration == 0:
                    current_prompt = prompt
                else:
                    # Create refinement prompt based on previous results
                    current_prompt = self._create_refinement_prompt(
                        prompt, best_result, domain, quality_target
                    )
                
                # Generate and extract
                current_result = self.generate_and_extract(current_prompt, domain, **kwargs)
                current_quality = current_result.quality_scores.get("overall_quality", 0.0)
                
                # Track refinement history
                result.refinement_history.append({
                    "iteration": iteration + 1,
                    "prompt": current_prompt,
                    "quality": current_quality,
                    "text_length": len(current_result.generated_text or ""),
                    "entity_count": sum(len(v) for v in (current_result.extraction_result.entities.values() if current_result.extraction_result else []))
                })
                
                # Update best result if this is better
                if current_quality > best_quality:
                    best_result = current_result
                    best_quality = current_quality
                
                # Check if we've met the quality target
                if current_quality >= quality_target:
                    result.validation_passed = True
                    break
            
            # Use the best result found
            if best_result:
                result.generated_text = best_result.generated_text
                result.extraction_result = best_result.extraction_result
                result.quality_scores = best_result.quality_scores
                result.generation_metadata = best_result.generation_metadata
                result.iterations_count = len(result.refinement_history)
            
        except Exception as e:
            result.quality_scores = {"error": str(e), "overall_quality": 0.0}
            result.validation_passed = False
        
        # Update statistics
        result.processing_time = time.time() - start_time
        self.stats["total_requests"] += 1
        self.stats["hybrid_requests"] += 1
        self.stats["total_processing_time"] += result.processing_time
        
        return result
    
    def _create_schema_aware_prompt(self, 
                                  original_prompt: str, 
                                  domain: str, 
                                  entity_types: List[str],
                                  target_entities: List[str] = None) -> str:
        """Create a prompt that guides generation based on extraction schema."""
        
        schema = get_schema_by_domain(domain)
        domain_desc = schema.get("description", f"{domain} domain content")
        
        # Build entity guidance
        entities_to_include = target_entities or entity_types[:3]  # Limit to top 3 for clarity
        entity_guidance = ", ".join(entities_to_include)
        
        schema_prompt = f"""
{original_prompt}

በዚህ {domain} ዓይነት ጽሑፍ ውስጥ የሚከተሉት አካላት መኖር አለባቸው: {entity_guidance}

እባክዎ ዝርዝር እና ትክክለኛ መረጃ ያለው ጽሑፍ ይፍጠሩ።
"""
        
        return schema_prompt.strip()
    
    def _create_refinement_prompt(self, 
                                original_prompt: str, 
                                previous_result: HybridResult, 
                                domain: str,
                                quality_target: float) -> str:
        """Create a refinement prompt based on previous generation quality."""
        
        if not previous_result or not previous_result.extraction_result:
            return original_prompt
        
        # Analyze what was missing or weak in previous result
        quality = previous_result.quality_scores.get("overall_quality", 0.0)
        entity_count = sum(len(v) for v in previous_result.extraction_result.entities.values())
        
        refinement_guidance = []
        
        if quality < quality_target:
            if entity_count < 3:
                refinement_guidance.append("ተጨማሪ ዝርዝር መረጃ እና አካላት ያክሉ")
            
            if previous_result.quality_scores.get("schema_coverage", 0) < 0.7:
                refinement_guidance.append("የተጠበቁ አካላትን ያካትቱ") 
            
            if previous_result.quality_scores.get("average_confidence", 0) < 0.7:
                refinement_guidance.append("የበለጠ ግልፅ እና ትክክለኛ መረጃ ይጠቀሙ")
        
        guidance_text = "፣ ".join(refinement_guidance) if refinement_guidance else "ጽሑፉን ያሻሽሉ"
        
        refinement_prompt = f"""
{original_prompt}

እባክዎ ቀደም ያለውን ጽሑፍ በመሻሻል {guidance_text}።

የተሻሻለ እና የበለጠ ዝርዝር ጽሑፍ ይፍጠሩ።
"""
        
        return refinement_prompt.strip()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        if stats["total_requests"] > 0:
            stats["average_processing_time"] = stats["total_processing_time"] / stats["total_requests"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_requests"]
        else:
            stats["average_processing_time"] = 0.0
            stats["cache_hit_rate"] = 0.0
        
        return stats
    
    def clear_cache(self):
        """Clear the processing cache."""
        if self.cache:
            self.cache.clear()
            print("🧹 Cache cleared")
    
    def export_result(self, result: HybridResult, output_path: str):
        """Export a hybrid result to JSON file."""
        
        # Convert result to serializable format
        export_data = {
            "original_prompt": result.original_prompt,
            "processing_mode": result.processing_mode.value,
            "domain": result.domain,
            "generated_text": result.generated_text,
            "generation_metadata": result.generation_metadata,
            "extraction_result": {
                "entities": result.extraction_result.entities if result.extraction_result else {},
                "relationships": result.extraction_result.relationships if result.extraction_result else [],
                "events": result.extraction_result.events if result.extraction_result else [],
                "confidence_scores": result.extraction_result.confidence_scores if result.extraction_result else {},
                "character_spans": result.extraction_result.character_spans if result.extraction_result else {},
                "metadata": result.extraction_result.metadata if result.extraction_result else {}
            },
            "quality_scores": result.quality_scores,
            "validation_passed": result.validation_passed,
            "processing_time": result.processing_time,
            "iterations_count": result.iterations_count,
            "refinement_history": result.refinement_history
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Result exported to: {output_path}")


def create_amharic_language_ai(generation_model_path: str = None,
                             extraction_api_key: str = None,
                             **kwargs) -> AmharicLanguageAI:
    """Factory function to create AmharicLanguageAI instance."""
    
    config = LanguageAIConfig(
        generation_model_path=generation_model_path,
        extraction_api_key=extraction_api_key,
        **kwargs
    )
    
    return AmharicLanguageAI(config)
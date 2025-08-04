"""
Amharic Information Extraction Pipeline
Process large collections of Amharic documents efficiently
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

from .amharic_extractor import AmharicExtractor, ExtractionResult, AmharicExtractionConfig


@dataclass
class PipelineConfig:
    """Configuration for extraction pipeline."""
    batch_size: int = 10
    max_workers: int = 4
    output_dir: str = "outputs/extraction_results"
    save_interval: int = 100
    include_metadata: bool = True
    domains: List[str] = None
    file_patterns: List[str] = None
    
    def __post_init__(self):
        if self.domains is None:
            self.domains = ["news", "government", "education", "culture"]
        if self.file_patterns is None:
            self.file_patterns = ["*.json", "*.jsonl", "*.txt"]


@dataclass
class PipelineStats:
    """Statistics for extraction pipeline."""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    processing_time: float = 0.0
    average_processing_time: float = 0.0
    domains_processed: Dict[str, int] = None
    
    def __post_init__(self):
        if self.domains_processed is None:
            self.domains_processed = {}


class ExtractionPipeline:
    """High-performance pipeline for processing Amharic document collections."""
    
    def __init__(self, 
                 extractor: AmharicExtractor,
                 config: Optional[PipelineConfig] = None):
        self.extractor = extractor
        self.config = config or PipelineConfig()
        self.stats = PipelineStats()
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: List[ExtractionResult] = []
        self.results_buffer: List[Dict[str, Any]] = []
    
    def process_directory(self, 
                         input_dir: str, 
                         domain: str = "news",
                         recursive: bool = True) -> List[ExtractionResult]:
        """Process all documents in a directory."""
        
        print(f"🔍 Processing directory: {input_dir}")
        print(f"📊 Domain: {domain}")
        print(f"🔄 Recursive: {recursive}")
        
        # Find all matching files
        input_path = Path(input_dir)
        files = []
        
        for pattern in self.config.file_patterns:
            if recursive:
                files.extend(input_path.rglob(pattern))
            else:
                files.extend(input_path.glob(pattern))
        
        print(f"📁 Found {len(files)} files to process")
        
        if not files:
            print("⚠️  No files found matching patterns")
            return []
        
        # Process files
        return self.process_files(files, domain)
    
    def process_files(self, 
                     file_paths: List[Path], 
                     domain: str = "news") -> List[ExtractionResult]:
        """Process a list of files."""
        
        self.stats.total_documents = len(file_paths)
        self.stats.domains_processed[domain] = 0
        
        start_time = time.time()
        
        print(f"⚡ Starting extraction pipeline...")
        print(f"   Files: {len(file_paths)}")
        print(f"   Domain: {domain}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Max workers: {self.config.max_workers}")
        
        # Process files in batches with progress bar
        with tqdm(total=len(file_paths), desc="Processing files") as pbar:
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                
                # Submit all tasks
                future_to_file = {}
                for file_path in file_paths:
                    future = executor.submit(self._process_single_file, file_path, domain)
                    future_to_file[future] = file_path
                
                # Process completed tasks
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    
                    try:
                        result = future.result()
                        if result:
                            self.results.append(result)
                            self.stats.processed_documents += 1
                            self.stats.domains_processed[domain] += 1
                            
                            # Update entity/relationship counts
                            self.stats.total_entities += sum(len(v) for v in result.entities.values())
                            self.stats.total_relationships += len(result.relationships)
                            
                            # Add to buffer for periodic saving
                            self.results_buffer.append(self._result_to_dict(result))
                            
                            # Save periodically
                            if len(self.results_buffer) >= self.config.save_interval:
                                self._save_batch()
                        
                    except Exception as e:
                        print(f"❌ Failed to process {file_path}: {e}")
                        self.stats.failed_documents += 1
                    
                    pbar.update(1)
        
        # Save remaining results
        if self.results_buffer:
            self._save_batch()
        
        # Update final statistics
        self.stats.processing_time = time.time() - start_time
        self.stats.average_processing_time = self.stats.processing_time / max(1, self.stats.processed_documents)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def process_collection_data(self) -> List[ExtractionResult]:
        """Process our existing 30K+ article collection."""
        
        print("📚 Processing Amharic Article Collection")
        print("=" * 50)
        
        # Paths to our collected data
        data_paths = [
            "data/collected",
            "data/processed/processed_articles", 
            "data/training"
        ]
        
        all_results = []
        
        for data_path in data_paths:
            if not Path(data_path).exists():
                print(f"⚠️  Path not found: {data_path}")
                continue
            
            print(f"\n🔄 Processing: {data_path}")
            
            # Determine domain based on path
            if "government" in data_path.lower():
                domain = "government"
            elif "education" in data_path.lower():
                domain = "education"
            elif "culture" in data_path.lower():
                domain = "culture"
            else:
                domain = "news"  # Default
            
            # Process directory
            results = self.process_directory(data_path, domain=domain, recursive=True)
            all_results.extend(results)
        
        return all_results
    
    def _process_single_file(self, file_path: Path, domain: str) -> Optional[ExtractionResult]:
        """Process a single file and extract information."""
        
        try:
            # Read file content
            content = self._read_file(file_path)
            if not content:
                return None
            
            # Extract information
            result = self.extractor.extract(content, domain=domain)
            
            # Add file metadata
            result.metadata.update({
                "source_file": str(file_path),
                "file_size": file_path.stat().st_size,
                "processing_timestamp": time.time()
            })
            
            return result
            
        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            return None
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read content from various file formats."""
        
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract text from different JSON structures
                if isinstance(data, dict):
                    # Try common text fields
                    for field in ['text', 'content', 'body', 'article', 'title']:
                        if field in data and data[field]:
                            return str(data[field])
                    
                    # If no standard field, concatenate all string values
                    text_parts = []
                    for value in data.values():
                        if isinstance(value, str) and len(value) > 10:
                            text_parts.append(value)
                    
                    return ' '.join(text_parts) if text_parts else None
                
                elif isinstance(data, list):
                    # Handle arrays of text
                    return ' '.join(str(item) for item in data if isinstance(item, str))
            
            elif file_path.suffix.lower() == '.jsonl':
                texts = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if isinstance(data, dict):
                                for field in ['text', 'content', 'body']:
                                    if field in data:
                                        texts.append(str(data[field]))
                                        break
                        except json.JSONDecodeError:
                            continue
                
                return ' '.join(texts) if texts else None
            
            else:  # .txt and other text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            return None
    
    def _result_to_dict(self, result: ExtractionResult) -> Dict[str, Any]:
        """Convert ExtractionResult to dictionary for serialization."""
        return {
            "text": result.text[:500] + "..." if len(result.text) > 500 else result.text,  # Truncate for storage
            "domain": result.domain,
            "entities": result.entities,
            "relationships": result.relationships,
            "events": result.events,
            "confidence_scores": result.confidence_scores,
            "character_spans": result.character_spans,
            "metadata": result.metadata
        }
    
    def _save_batch(self):
        """Save current batch of results."""
        
        timestamp = int(time.time())
        batch_file = Path(self.config.output_dir) / f"extraction_batch_{timestamp}.json"
        
        try:
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump(self.results_buffer, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved batch: {len(self.results_buffer)} results → {batch_file}")
            self.results_buffer.clear()
            
        except Exception as e:
            print(f"❌ Failed to save batch: {e}")
    
    def _print_summary(self):
        """Print processing summary."""
        
        print(f"\n📊 Extraction Pipeline Summary")
        print("=" * 50)
        print(f"📁 Total documents: {self.stats.total_documents}")
        print(f"✅ Processed: {self.stats.processed_documents}")
        print(f"❌ Failed: {self.stats.failed_documents}")
        print(f"📈 Success rate: {self.stats.processed_documents/max(1, self.stats.total_documents)*100:.1f}%")
        print(f"⏱️  Total time: {self.stats.processing_time:.2f}s")
        print(f"⚡ Average per document: {self.stats.average_processing_time:.3f}s")
        print(f"🏷️  Total entities extracted: {self.stats.total_entities:,}")
        print(f"🔗 Total relationships: {self.stats.total_relationships:,}")
        
        print(f"\n📂 By Domain:")
        for domain, count in self.stats.domains_processed.items():
            print(f"   {domain}: {count} documents")
    
    def export_consolidated_results(self, output_file: str = "consolidated_extraction_results.json"):
        """Export all results to a single consolidated file."""
        
        output_path = Path(self.config.output_dir) / output_file
        
        # Collect all results
        all_results = []
        
        # Add buffered results
        all_results.extend(self.results_buffer)
        
        # Add saved batch files
        for batch_file in Path(self.config.output_dir).glob("extraction_batch_*.json"):
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    all_results.extend(batch_data)
            except Exception as e:
                print(f"⚠️  Failed to read batch file {batch_file}: {e}")
        
        # Save consolidated results
        consolidated_data = {
            "metadata": {
                "total_documents": len(all_results),
                "extraction_timestamp": time.time(),
                "pipeline_stats": asdict(self.stats),
                "domains": list(set(r["domain"] for r in all_results))
            },
            "results": all_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Consolidated results exported: {output_path}")
        print(f"   Total documents: {len(all_results)}")
        
        return str(output_path)
    
    def generate_analytics_report(self) -> Dict[str, Any]:
        """Generate analytics report from extraction results."""
        
        # Collect all results for analysis
        all_results = []
        
        # Load from batch files
        for batch_file in Path(self.config.output_dir).glob("extraction_batch_*.json"):
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    all_results.extend(batch_data)
            except Exception as e:
                print(f"⚠️  Failed to read batch file {batch_file}: {e}")
        
        if not all_results:
            return {"error": "No results available for analysis"}
        
        # Analytics
        analytics = {
            "overview": {
                "total_documents": len(all_results),
                "total_entities": sum(sum(len(v) for v in r["entities"].values()) for r in all_results),
                "total_relationships": sum(len(r["relationships"]) for r in all_results),
                "domains": list(set(r["domain"] for r in all_results))
            },
            "entity_analysis": {},
            "domain_analysis": {},
            "quality_metrics": {},
            "processing_stats": asdict(self.stats)
        }
        
        # Entity type analysis
        entity_counts = {}
        for result in all_results:
            for entity_type, entities in result["entities"].items():
                if entity_type not in entity_counts:
                    entity_counts[entity_type] = []
                entity_counts[entity_type].extend(entities)
        
        for entity_type, entities in entity_counts.items():
            analytics["entity_analysis"][entity_type] = {
                "total_count": len(entities),
                "unique_count": len(set(entities)),
                "most_common": list(set(entities))[:10] if entities else []
            }
        
        # Domain analysis
        domain_stats = {}
        for result in all_results:
            domain = result["domain"]
            if domain not in domain_stats:
                domain_stats[domain] = {
                    "document_count": 0,
                    "total_entities": 0,
                    "avg_confidence": 0.0
                }
            
            domain_stats[domain]["document_count"] += 1
            domain_stats[domain]["total_entities"] += sum(len(v) for v in result["entities"].values())
            
            # Average confidence
            if result.get("confidence_scores"):
                avg_conf = sum(result["confidence_scores"].values()) / len(result["confidence_scores"])
                domain_stats[domain]["avg_confidence"] += avg_conf
        
        # Finalize domain averages
        for domain, stats in domain_stats.items():
            if stats["document_count"] > 0:
                stats["avg_confidence"] /= stats["document_count"]
                stats["entities_per_document"] = stats["total_entities"] / stats["document_count"]
        
        analytics["domain_analysis"] = domain_stats
        
        # Quality metrics
        confidences = []
        entity_densities = []
        
        for result in all_results:
            if result.get("confidence_scores"):
                confidences.extend(result["confidence_scores"].values())
            
            text_length = result.get("metadata", {}).get("text_length", 1)
            entity_count = sum(len(v) for v in result["entities"].values())
            entity_densities.append(entity_count / text_length * 1000)  # per 1000 chars
        
        analytics["quality_metrics"] = {
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "avg_entity_density": sum(entity_densities) / len(entity_densities) if entity_densities else 0.0,
            "confidence_distribution": {
                "min": min(confidences) if confidences else 0.0,
                "max": max(confidences) if confidences else 0.0,
                "median": sorted(confidences)[len(confidences)//2] if confidences else 0.0
            }
        }
        
        return analytics


def create_extraction_pipeline(api_key: Optional[str] = None,
                             batch_size: int = 10,
                             max_workers: int = 4) -> ExtractionPipeline:
    """Factory function to create extraction pipeline."""
    
    # Create extractor
    extraction_config = AmharicExtractionConfig(
        api_key=api_key,
        model_name="gemini-1.5-flash",
        use_few_shot=True,
        enable_source_grounding=True
    )
    
    extractor = AmharicExtractor(extraction_config)
    
    # Create pipeline config
    pipeline_config = PipelineConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        save_interval=50,  # Save every 50 documents
        include_metadata=True
    )
    
    return ExtractionPipeline(extractor, pipeline_config)


async def async_process_collection():
    """Async wrapper for processing the article collection."""
    
    print("🚀 Starting Async Amharic Collection Processing")
    
    # Create pipeline
    pipeline = create_extraction_pipeline(
        batch_size=20,
        max_workers=6
    )
    
    # Process collection
    results = pipeline.process_collection_data()
    
    # Export results
    output_file = pipeline.export_consolidated_results()
    
    # Generate analytics
    analytics = pipeline.generate_analytics_report()
    
    # Save analytics
    analytics_file = Path(pipeline.config.output_dir) / "extraction_analytics.json"
    with open(analytics_file, 'w', encoding='utf-8') as f:
        json.dump(analytics, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Analytics saved: {analytics_file}")
    
    return {
        "results_file": output_file,
        "analytics_file": str(analytics_file),
        "total_processed": len(results),
        "analytics": analytics
    }
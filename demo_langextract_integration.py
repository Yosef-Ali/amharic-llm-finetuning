#!/usr/bin/env python3
"""
🚀 Amharic H-Net + LangExtract Integration Demonstration
Complete showcase of advanced information extraction capabilities
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from amharichnet.extraction import AmharicExtractor, ExtractionPipeline
from amharichnet.extraction.amharic_extractor import AmharicExtractionConfig, create_amharic_extractor
from amharichnet.extraction.extraction_pipeline import create_extraction_pipeline

def print_banner():
    """Print demo banner."""
    print("🔥" * 80)
    print("🚀 AMHARIC H-NET + LANGEXTRACT INTEGRATION DEMO")
    print("   Advanced Information Extraction for Amharic Text")
    print("🔥" * 80)
    print()

def demo_basic_extraction():
    """Demonstrate basic Amharic text extraction."""
    print("📝 DEMO 1: Basic Amharic Information Extraction")
    print("=" * 60)
    
    # Sample Amharic news text
    sample_text = """
    ጠቅላይ ሚኒስትር ዶ/ር አብይ አሕመድ ዛረ በመনግሥት ቤት አዲስ አበባ ውስጥ ከተለያዩ ሚኒስትሮች ጋር ስብሰባ ተካሂዷል። 
    ስብሰባው በጥር 15 ቀን 2016 ዓ.ም ነው የተካሄደው። በስብሰባው የኢትዮጵያ ኢኮኖሚ ልማት ተወያይተዋል።
    የትምህርት ሚኒስትር ፕ/ር ቢሩክ መስፍን እና የጤና ሚኒስትር ዶ/ር ልያ ታደሰ በስብሰባው ተሳትፈዋል።
    """
    
    print("📖 Input text:")
    print(sample_text.strip())
    print()
    
    # Create extractor
    extractor = create_amharic_extractor()
    
    # Perform extraction
    print("🔍 Extracting information...")
    result = extractor.extract(sample_text, domain="news")
    
    # Display results
    print("✅ Extraction Results:")
    print("-" * 40)
    
    print(f"🏷️  Domain: {result.domain}")
    print(f"📄 Text length: {len(result.text)} characters")
    print(f"🎯 Extraction method: {result.metadata.get('extraction_method', 'unknown')}")
    print()
    
    print("👥 People found:")
    for person in result.entities.get("people", []):
        print(f"   • {person}")
    print()
    
    print("🏢 Organizations found:")
    for org in result.entities.get("organizations", []):
        print(f"   • {org}")
    print()
    
    print("📍 Locations found:")
    for loc in result.entities.get("locations", []):
        print(f"   • {loc}")
    print()
    
    print("📅 Dates found:")
    for date in result.entities.get("dates", []):
        print(f"   • {date}")
    print()
    
    print("🔗 Relationships found:")
    for rel in result.relationships:
        print(f"   • {rel}")
    print()
    
    # Quality evaluation
    quality_metrics = extractor.evaluate_extraction_quality(result)
    print("📊 Quality Metrics:")
    for metric, value in quality_metrics.items():
        if isinstance(value, float):
            print(f"   • {metric}: {value:.3f}")
        else:
            print(f"   • {metric}: {value}")
    print()
    
    return result

def demo_batch_extraction():
    """Demonstrate batch processing of multiple texts."""
    print("📚 DEMO 2: Batch Processing Multiple Texts")
    print("=" * 60)
    
    # Sample texts in different domains
    texts = [
        {
            "text": "የኢትዮጵያ ፌዴራላዊ ዲሞክራሲያዊ ሪፐብሊክ ህዝብ ተወካዮች ምክር ቤት አዲስ አዋጅ አወጣ። አዋጁ ሁ.ቁ 1250/2016 ዓ.ም የሚል ነው።",
            "domain": "government"
        },
        {
            "text": "አዲስ አበባ ዩኒቨርስቲ የህክምና ፋኩልቲ በዚህ ዓመት 500 ተማሪዎችን ይቀበላል። ዲ/ር መሐሪ ተስፋዬ የፋኩልቲው ዲን ናቸው።",
            "domain": "education"
        },
        {
            "text": "በኢትዮጵያ የጥምቀት በዓል ውብ ባህላዊ ስርዓት አለው። በየዓመቱ ጥር 11 ይከበራል። በአዲስ አበባ ጃን ሜዳ በዚህ በዓል ከፍተኛ አከባበር ይደረጋል።",
            "domain": "culture"
        }
    ]
    
    # Create extractor
    extractor = create_amharic_extractor()
    
    print(f"🔄 Processing {len(texts)} texts in different domains...")
    print()
    
    results = []
    for i, item in enumerate(texts, 1):
        print(f"📄 Text {i} ({item['domain']} domain):")
        print(f"   Input: {item['text'][:100]}...")
        
        result = extractor.extract(item['text'], domain=item['domain'])
        results.append(result)
        
        print(f"   📊 Entities: {sum(len(v) for v in result.entities.values())}")
        print(f"   🔗 Relationships: {len(result.relationships)}")
        print(f"   ⭐ Quality: {extractor.evaluate_extraction_quality(result)['overall_quality']:.3f}")
        print()
    
    # Summary statistics
    total_entities = sum(sum(len(v) for v in r.entities.values()) for r in results)
    total_relationships = sum(len(r.relationships) for r in results)
    avg_quality = sum(extractor.evaluate_extraction_quality(r)['overall_quality'] for r in results) / len(results)
    
    print("📈 Batch Processing Summary:")
    print(f"   • Total texts processed: {len(results)}")
    print(f"   • Total entities extracted: {total_entities}")
    print(f"   • Total relationships found: {total_relationships}")
    print(f"   • Average quality score: {avg_quality:.3f}")
    print()
    
    return results

def demo_collection_processing():
    """Demonstrate processing our article collection."""
    print("🗂️  DEMO 3: Article Collection Processing")
    print("=" * 60)
    
    # Check for data directories
    data_paths = [
        "data/collected",
        "data/processed/processed_articles", 
        "data/training"
    ]
    
    available_paths = []
    for path in data_paths:
        if Path(path).exists():
            available_paths.append(path)
            file_count = len(list(Path(path).rglob("*.json"))) + len(list(Path(path).rglob("*.txt")))
            print(f"✅ Found: {path} ({file_count} files)")
        else:
            print(f"⚠️  Missing: {path}")
    
    if not available_paths:
        print("❌ No data directories found. Creating sample data...")
        
        # Create sample data for demonstration
        sample_dir = Path("demo_data")
        sample_dir.mkdir(exist_ok=True)
        
        sample_articles = [
            {
                "title": "የመንግሥት ስብሰባ",
                "content": "ጠቅላይ ሚኒስትር ዶ/ር አብይ አሕመድ በመንግሥት ቤት ከሚኒስትሮች ጋር ስብሰባ አካሂደዋል።",
                "date": "2016-01-15",
                "domain": "news"
            },
            {
                "title": "የኢትዮጵያ አዲስ ሕግ",
                "content": "ህዝብ ተወካዮች ምክር ቤት አዲስ አዋጅ ሁ.ቁ 1250/2016 ዓ.ም አወጣ።",
                "date": "2016-01-10", 
                "domain": "government"
            },
            {
                "title": "የዩኒቨርስቲ ዜና",
                "content": "አዲስ አበባ ዩኒቨርስቲ የጥናት ፕሮግራም ጀምሯል። ፕ/ር ተስፋዬ መሪነታቸውን ይይዛሉ።",
                "date": "2016-01-12",
                "domain": "education"
            }
        ]
        
        for i, article in enumerate(sample_articles):
            with open(sample_dir / f"article_{i+1}.json", 'w', encoding='utf-8') as f:
                json.dump(article, f, indent=2, ensure_ascii=False)
        
        available_paths = [str(sample_dir)]
        print(f"✅ Created sample data in: {sample_dir}")
    
    print()
    
    # Create extraction pipeline
    print("🔧 Setting up extraction pipeline...")
    pipeline = create_extraction_pipeline(
        batch_size=5,
        max_workers=2  # Reduced for demo
    )
    
    print("🚀 Starting collection processing...")
    print()
    
    # Process first available path as demonstration
    demo_path = available_paths[0]
    results = pipeline.process_directory(demo_path, domain="news", recursive=True)
    
    # Show detailed results for first few items
    print("📋 Sample Extraction Results:")
    print("-" * 50)
    
    for i, result in enumerate(results[:3]):  # Show first 3 results
        print(f"📄 Document {i+1}:")
        print(f"   Domain: {result.domain}")
        print(f"   Text preview: {result.text[:100]}...")
        print(f"   Entities found: {dict((k, len(v)) for k, v in result.entities.items())}")
        print(f"   Relationships: {len(result.relationships)}")
        print(f"   Confidence: {result.confidence_scores}")
        print()
    
    # Export results
    if results:
        output_file = pipeline.export_consolidated_results("demo_extraction_results.json")
        print(f"💾 Results exported to: {output_file}")
        
        # Generate analytics
        analytics = pipeline.generate_analytics_report()
        print("📊 Analytics Summary:")
        print(f"   • Documents processed: {analytics['overview']['total_documents']}")
        print(f"   • Total entities: {analytics['overview']['total_entities']}")
        print(f"   • Total relationships: {analytics['overview']['total_relationships']}")
        print(f"   • Domains covered: {analytics['overview']['domains']}")
    
    print()
    return results

def demo_schema_capabilities():
    """Demonstrate different domain schemas."""
    print("🎯 DEMO 4: Domain-Specific Schema Capabilities")
    print("=" * 60)
    
    from amharichnet.extraction.schemas import AMHARIC_SCHEMAS, get_schema_by_domain
    
    # Show available domains
    domains = list(AMHARIC_SCHEMAS.keys())
    print(f"🗂️  Available domains: {', '.join(domains)}")
    print()
    
    # Show schema details for each domain
    for domain in domains[:3]:  # Show first 3 domains
        schema = get_schema_by_domain(domain)
        print(f"📋 {domain.title()} Domain Schema:")
        print(f"   Description: {schema.get('description', 'N/A')}")
        
        entities = schema.get('entities', {})
        print(f"   Entity types ({len(entities)}):")
        for entity_name, entity_config in list(entities.items())[:3]:
            print(f"      • {entity_name}: {entity_config.get('description', 'N/A')}")
        
        relationships = schema.get('relationships', {})
        if relationships:
            print(f"   Relationship types ({len(relationships)}):")
            for rel_name, rel_config in list(relationships.items())[:2]:
                print(f"      • {rel_name}: {rel_config.get('description', 'N/A')}")
        
        print()

def demo_api_simulation():
    """Simulate API endpoint functionality."""
    print("🌐 DEMO 5: API Endpoint Simulation")
    print("=" * 60)
    
    def simulate_extraction_api(text: str, domain: str = "news") -> Dict[str, Any]:
        """Simulate extraction API endpoint."""
        extractor = create_amharic_extractor()
        result = extractor.extract(text, domain=domain)
        quality = extractor.evaluate_extraction_quality(result)
        
        return {
            "status": "success",
            "processing_time_ms": 150,  # Simulated
            "input": {
                "text_length": len(text),
                "domain": domain
            },
            "output": {
                "entities": result.entities,
                "relationships": result.relationships,
                "events": result.events,
                "confidence_scores": result.confidence_scores,
                "character_spans": result.character_spans
            },
            "quality_metrics": quality,
            "metadata": result.metadata
        }
    
    # Test API with sample text
    test_text = "የኢትዮጵያ ዋና ሚኒስትር ዶ/ር አብይ አሕመድ በአዲስ አበባ የተካሄደ ስብሰባ ላይ ተሳትፈዋል።"
    
    print("📨 API Request:")
    print(f"   Text: {test_text}")
    print(f"   Domain: news")
    print()
    
    print("⏳ Processing...")
    api_response = simulate_extraction_api(test_text, "news")
    
    print("📨 API Response:")
    print(f"   Status: {api_response['status']}")
    print(f"   Processing time: {api_response['processing_time_ms']}ms")
    print(f"   Entities found: {sum(len(v) for v in api_response['output']['entities'].values())}")
    print(f"   Overall quality: {api_response['quality_metrics']['overall_quality']:.3f}")
    print()
    
    # Show JSON structure
    print("📄 Full JSON Response Structure:")
    response_keys = list(api_response.keys())
    for key in response_keys:
        if key == "output":
            output_keys = list(api_response[key].keys())
            print(f"   {key}: {{{', '.join(output_keys)}}}")
        else:
            print(f"   {key}: {type(api_response[key]).__name__}")
    print()

async def run_full_demo():
    """Run the complete demonstration."""
    print_banner()
    
    print("🎯 This demo showcases our Amharic H-Net + LangExtract integration")
    print("   combining advanced text generation with information extraction")
    print()
    
    # Check dependencies
    print("🔍 Checking system requirements...")
    
    try:
        import langextract
        langextract_available = True
        print("   ✅ LangExtract library: Available")
    except ImportError:
        langextract_available = False
        print("   ⚠️  LangExtract library: Not available (using mock extraction)")
    
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if api_key:
        print("   ✅ Gemini API key: Configured")
    else:
        print("   ⚠️  Gemini API key: Not found (using mock extraction)")
    
    if not langextract_available and not api_key:
        print("   ℹ️  Demo will use rule-based mock extraction for demonstration")
    
    print()
    input("Press Enter to start the demonstration...")
    print()
    
    # Run all demos
    try:
        # Demo 1: Basic extraction
        demo_basic_extraction()
        input("Press Enter to continue to batch processing demo...")
        print()
        
        # Demo 2: Batch processing
        demo_batch_extraction()
        input("Press Enter to continue to collection processing demo...")
        print()
        
        # Demo 3: Collection processing
        demo_collection_processing()
        input("Press Enter to continue to schema capabilities demo...")
        print()
        
        # Demo 4: Schema capabilities
        demo_schema_capabilities()
        input("Press Enter to continue to API simulation demo...")
        print()
        
        # Demo 5: API simulation
        demo_api_simulation()
        
        # Final summary
        print("🎉 DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("✅ Successfully demonstrated:")
        print("   • Basic Amharic information extraction")
        print("   • Batch processing capabilities")
        print("   • Article collection processing")
        print("   • Domain-specific schema handling")
        print("   • API endpoint simulation")
        print()
        print("🚀 The Amharic H-Net + LangExtract integration is ready!")
        print("   This system combines advanced text generation with")
        print("   sophisticated information extraction for Amharic text.")
        print()
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main demo entry point."""
    if len(sys.argv) > 1:
        demo_name = sys.argv[1].lower()
        
        if demo_name == "basic":
            print_banner()
            demo_basic_extraction()
        elif demo_name == "batch":
            print_banner()
            demo_batch_extraction()
        elif demo_name == "collection":
            print_banner()
            demo_collection_processing()
        elif demo_name == "schema":
            print_banner()
            demo_schema_capabilities()
        elif demo_name == "api":
            print_banner()
            demo_api_simulation()
        else:
            print(f"Unknown demo: {demo_name}")
            print("Available demos: basic, batch, collection, schema, api")
    else:
        # Run full interactive demo
        asyncio.run(run_full_demo())

if __name__ == "__main__":
    main()
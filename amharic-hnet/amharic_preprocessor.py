#!/usr/bin/env python3
"""
Amharic Data Preprocessing Pipeline - Phase 1.2 Implementation
Follows the Grand Implementation Plan for data quality and preprocessing

Features:
- Text normalization for Amharic script
- Data quality validation
- Corpus consolidation
- Statistical analysis
- Kaggle dataset preparation
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import unicodedata
from collections import Counter
import statistics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmharicPreprocessor:
    """Comprehensive Amharic text preprocessing pipeline"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Amharic Unicode range
        self.amharic_range = (0x1200, 0x137F)
        
        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_chars': 0,
            'total_words': 0,
            'total_sentences': 0,
            'amharic_char_ratio': 0.0,
            'avg_words_per_sentence': 0.0,
            'quality_scores': []
        }
    
    def normalize_amharic_text(self, text: str) -> str:
        """Normalize Amharic text for consistency"""
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters except common punctuation
        text = re.sub(r'[^\w\s።፣፤፥፦፧፨፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፼]', '', text)
        
        # Normalize Amharic punctuation
        text = text.replace('።', '። ')  # Add space after sentence end
        text = text.replace('፣', '፣ ')  # Add space after comma
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_text_quality(self, text: str) -> float:
        """Calculate quality score for Amharic text"""
        if not text or len(text) < 10:
            return 0.0
        
        score = 0.0
        
        # Length score (0-30 points)
        word_count = len(text.split())
        if word_count >= 50:
            score += 30
        elif word_count >= 20:
            score += 20
        elif word_count >= 10:
            score += 10
        
        # Amharic character ratio (0-40 points)
        amharic_chars = sum(1 for char in text if self.amharic_range[0] <= ord(char) <= self.amharic_range[1])
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0:
            amharic_ratio = amharic_chars / total_chars
            score += amharic_ratio * 40
        
        # Sentence structure (0-20 points)
        sentences = text.split('።')
        if len(sentences) >= 3:
            score += 20
        elif len(sentences) >= 2:
            score += 15
        elif len(sentences) >= 1:
            score += 10
        
        # Diversity score (0-10 points)
        unique_words = len(set(text.split()))
        if word_count > 0:
            diversity = unique_words / word_count
            score += diversity * 10
        
        return min(score, 100.0)
    
    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text content from various file formats"""
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data.get('content', data.get('text', ''))
                    return str(data)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return ""
    
    def process_single_file(self, file_path: Path) -> Tuple[str, Dict]:
        """Process a single file and return normalized text with metadata"""
        # Extract raw text
        raw_text = self.extract_text_from_file(file_path)
        
        if not raw_text:
            return "", {}
        
        # Normalize text
        normalized_text = self.normalize_amharic_text(raw_text)
        
        # Calculate quality metrics
        quality_score = self.calculate_text_quality(normalized_text)
        word_count = len(normalized_text.split())
        char_count = len(normalized_text)
        sentence_count = len(normalized_text.split('።'))
        
        # Calculate Amharic character ratio
        amharic_chars = sum(1 for char in normalized_text if self.amharic_range[0] <= ord(char) <= self.amharic_range[1])
        amharic_ratio = amharic_chars / max(char_count, 1)
        
        metadata = {
            'source_file': str(file_path),
            'quality_score': quality_score,
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'amharic_ratio': amharic_ratio,
            'processed': True
        }
        
        return normalized_text, metadata
    
    def process_all_files(self, min_quality_score: float = 30.0) -> List[Dict]:
        """Process all files in the raw directory"""
        logger.info("Starting comprehensive data preprocessing...")
        
        # Get all files from raw directory
        raw_files = list(self.raw_dir.glob('*'))
        raw_files = [f for f in raw_files if f.is_file()]
        
        self.stats['total_files'] = len(raw_files)
        
        processed_data = []
        consolidated_text = []
        
        logger.info(f"Processing {len(raw_files)} files...")
        
        for file_path in tqdm(raw_files, desc="Processing files"):
            try:
                normalized_text, metadata = self.process_single_file(file_path)
                
                if normalized_text and metadata.get('quality_score', 0) >= min_quality_score:
                    # Save individual processed file
                    output_filename = file_path.stem + '_processed.txt'
                    output_path = self.processed_dir / output_filename
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(normalized_text)
                    
                    # Add to consolidated corpus
                    consolidated_text.append(normalized_text)
                    processed_data.append(metadata)
                    
                    # Update statistics
                    self.stats['processed_files'] += 1
                    self.stats['total_chars'] += metadata['char_count']
                    self.stats['total_words'] += metadata['word_count']
                    self.stats['total_sentences'] += metadata['sentence_count']
                    self.stats['quality_scores'].append(metadata['quality_score'])
                    
                else:
                    logger.debug(f"Skipped {file_path} (quality score: {metadata.get('quality_score', 0):.1f})")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Calculate final statistics
        if self.stats['quality_scores']:
            self.stats['avg_quality_score'] = statistics.mean(self.stats['quality_scores'])
            self.stats['min_quality_score'] = min(self.stats['quality_scores'])
            self.stats['max_quality_score'] = max(self.stats['quality_scores'])
        
        if self.stats['total_sentences'] > 0:
            self.stats['avg_words_per_sentence'] = self.stats['total_words'] / self.stats['total_sentences']
        
        # Create consolidated corpus
        self.create_consolidated_corpus(consolidated_text)
        
        # Save processing metadata
        self.save_processing_metadata(processed_data)
        
        logger.info(f"Processing complete: {self.stats['processed_files']}/{self.stats['total_files']} files processed")
        
        return processed_data
    
    def create_consolidated_corpus(self, texts: List[str]):
        """Create a single consolidated corpus file"""
        corpus_path = self.data_dir / "amharic_consolidated_corpus.txt"
        
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                f.write(f"# Document {i+1}\n")
                f.write(text)
                f.write("\n\n")
        
        logger.info(f"Consolidated corpus saved to {corpus_path}")
        logger.info(f"Corpus size: {len('\n\n'.join(texts))} characters, {sum(len(t.split()) for t in texts)} words")
    
    def save_processing_metadata(self, processed_data: List[Dict]):
        """Save comprehensive processing metadata"""
        metadata = {
            'processing_stats': self.stats,
            'file_metadata': processed_data,
            'corpus_info': {
                'total_documents': len(processed_data),
                'total_words': self.stats['total_words'],
                'total_characters': self.stats['total_chars'],
                'avg_document_length': self.stats['total_words'] / max(len(processed_data), 1),
                'quality_threshold': 30.0
            },
            'next_steps': [
                'Upload consolidated corpus to Kaggle',
                'Create training/validation split',
                'Begin model training on Kaggle',
                'Monitor training progress'
            ]
        }
        
        metadata_path = self.metadata_dir / "preprocessing_report.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing metadata saved to {metadata_path}")
    
    def generate_kaggle_dataset_config(self):
        """Generate Kaggle dataset configuration"""
        config = {
            "title": "Amharic Language Corpus for LLM Training",
            "id": "amharic-llm-corpus",
            "licenses": [{"name": "CC-BY-SA-4.0"}],
            "keywords": ["amharic", "nlp", "language-model", "ethiopia", "corpus"],
            "collaborators": [],
            "data": []
        }
        
        config_path = self.data_dir / "dataset-metadata.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Kaggle dataset config saved to {config_path}")
    
    def run_preprocessing(self, min_quality_score: float = 30.0):
        """Run the complete preprocessing pipeline"""
        logger.info("=== Amharic Data Preprocessing Pipeline - Phase 1.2 ===")
        logger.info(f"Input directory: {self.raw_dir}")
        logger.info(f"Output directory: {self.processed_dir}")
        logger.info(f"Quality threshold: {min_quality_score}")
        
        # Process all files
        processed_data = self.process_all_files(min_quality_score)
        
        # Generate Kaggle dataset configuration
        self.generate_kaggle_dataset_config()
        
        # Print summary
        self.print_summary()
        
        return processed_data
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*60)
        print("📊 AMHARIC DATA PREPROCESSING SUMMARY")
        print("="*60)
        print(f"📁 Total files found: {self.stats['total_files']}")
        print(f"✅ Files processed: {self.stats['processed_files']}")
        print(f"📝 Total words: {self.stats['total_words']:,}")
        print(f"📄 Total characters: {self.stats['total_chars']:,}")
        print(f"📋 Total sentences: {self.stats['total_sentences']:,}")
        
        if self.stats['quality_scores']:
            print(f"⭐ Average quality score: {self.stats['avg_quality_score']:.1f}/100")
            print(f"📈 Quality range: {self.stats['min_quality_score']:.1f} - {self.stats['max_quality_score']:.1f}")
        
        print(f"📊 Avg words per sentence: {self.stats['avg_words_per_sentence']:.1f}")
        
        print("\n📁 Output files:")
        print(f"   📄 Consolidated corpus: {self.data_dir / 'amharic_consolidated_corpus.txt'}")
        print(f"   📊 Processing report: {self.metadata_dir / 'preprocessing_report.json'}")
        print(f"   ⚙️ Kaggle config: {self.data_dir / 'dataset-metadata.json'}")
        
        print("\n🚀 Next steps:")
        print("   1. Upload corpus to Kaggle dataset")
        print("   2. Create Kaggle training notebook")
        print("   3. Begin model training with enhanced architecture")
        print("   4. Monitor training progress and metrics")
        print("="*60)

def main():
    """Main execution function"""
    preprocessor = AmharicPreprocessor()
    
    # Run preprocessing with quality threshold
    processed_data = preprocessor.run_preprocessing(min_quality_score=25.0)
    
    if processed_data:
        print(f"\n🎉 Preprocessing complete! Ready for Phase 2: Model Training")
    else:
        print("\n⚠️ No data met quality requirements. Consider lowering quality threshold.")

if __name__ == "__main__":
    main()
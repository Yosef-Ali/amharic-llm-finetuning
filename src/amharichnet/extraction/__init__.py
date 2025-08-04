"""
Amharic Information Extraction Module
LangExtract integration for Amharic text processing
"""

from .amharic_extractor import AmharicExtractor, AmharicExtractionSchemas
from .extraction_pipeline import ExtractionPipeline
from .schemas import AMHARIC_SCHEMAS

__all__ = [
    "AmharicExtractor",
    "AmharicExtractionSchemas", 
    "ExtractionPipeline",
    "AMHARIC_SCHEMAS"
]
"""
Amharic-specific Information Extraction Schemas
Domain-specific schemas for LangExtract integration
"""

from typing import Dict, List, Any

# Amharic entity types and patterns
AMHARIC_ENTITY_TYPES = {
    # People and Organizations
    "PERSON": {
        "amharic": "ሰው",
        "patterns": ["ዶ/ር", "ፕ/ር", "ወ/ሮ", "ወ/ሪት", "አቶ"],
        "examples": ["ዶ/ር አበበ ተስፋዬ", "ወ/ሮ ዓለማየሁ ወርቃሪያ", "ፕ/ር መሐሪ ሰይፉ"]
    },
    "ORGANIZATION": {
        "amharic": "ድርጅት",
        "patterns": ["ሚኒስቴር", "ዩኒቨርስቲ", "ቤተሰብ", "ኩባንያ", "ድርጅት"],
        "examples": ["አዲስ አበባ ዩኒቨርስቲ", "የጤና ሚኒስቴር", "ኢትዮጵያ አየር መንገድ"]
    },
    
    # Places and Locations
    "LOCATION": {
        "amharic": "ቦታ",
        "patterns": ["ከተማ", "አውራጃ", "ወረዳ", "ክልል", "ቀበሌ"],
        "examples": ["አዲስ አበባ", "ባህር ዳር", "ትግራይ ክልል", "ኦሮሚያ ክልል"]
    },
    "REGION": {
        "amharic": "ክልል",
        "patterns": ["ክልል", "ዞን", "አውራጃ"],
        "examples": ["ትግራይ ክልል", "አማራ ክልል", "ኦሮሚያ ክልል", "ደቡብ ህዝቦች ክልል"]
    },
    
    # Time and Dates
    "DATE": {
        "amharic": "ቀን",
        "patterns": ["ዓ.ም", "ዓ.ዓ", "ዓመት", "ወር", "ቀን"],
        "examples": ["2016 ዓ.ም", "ሰኔ 2016", "ጥር 1 ቀን 2016 ዓ.ም"]
    },
    "TIME": {
        "amharic": "ሰዓት",
        "patterns": ["ሰዓት", "ደቂቃ", "ሌሊት", "ቀን"],
        "examples": ["12 ሰዓት", "ከሰዓት በኋላ", "በማለዳ", "በምሽት"]
    },
    
    # Government and Politics
    "GOVERNMENT_POSITION": {
        "amharic": "የመንግሥት ሹም",
        "patterns": ["ሚኒስትር", "ጠቅላይ ሚኒስትር", "ፕሬዚዳንት", "ንጉሥ"],
        "examples": ["ጠቅላይ ሚኒስትር", "የወጣቶች ሚኒስትር", "የትምህርት ሚኒስትር"]
    },
    "POLITICAL_PARTY": {
        "amharic": "ፖለቲካዊ ፓርቲ",
        "patterns": ["ፓርቲ", "ንቅናቄ", "ወገንተኛ"],
        "examples": ["ብልጽግና ፓርቲ", "ኢሕአዴግ", "ኢሕወሀት"]
    },
    
    # Cultural and Religious
    "RELIGIOUS_TERM": {
        "amharic": "ሃይማኖታዊ",
        "patterns": ["ቤተክርስቲያን", "መስጊድ", "ጾም", "በዓል"],
        "examples": ["ኢትዮጵያ ኦርቶዶክስ ተዋሕዶ", "እስልምና", "ትንሳኤ", "መስቀል"]
    },
    "CULTURAL_EVENT": {
        "amharic": "ባህላዊ ክንውን",
        "patterns": ["በዓል", "ጾም", "ስርዓት", "ወግ"],
        "examples": ["ጥምቀት", "መስቀል", "ኢርጫ", "አሽንዳ"]
    }
}

# Domain-specific schemas
AMHARIC_SCHEMAS = {
    "news": {
        "description": "Amharic news article information extraction",
        "entities": {
            "people": {
                "type": "PERSON",
                "description": "People mentioned in the news",
                "amharic_desc": "በዜናው ውስጥ የተጠቀሱ ሰዎች"
            },
            "organizations": {
                "type": "ORGANIZATION", 
                "description": "Organizations mentioned",
                "amharic_desc": "የተጠቀሱ ድርጅቶች"
            },
            "locations": {
                "type": "LOCATION",
                "description": "Places mentioned in the news",
                "amharic_desc": "በዜናው ውስጥ የተጠቀሱ ቦታዎች"
            },
            "dates": {
                "type": "DATE",
                "description": "Important dates mentioned",
                "amharic_desc": "አስፈላጊ ቀኖች"
            }
        },
        "relationships": {
            "works_at": {
                "description": "Person works at organization",
                "amharic_desc": "ሰው በድርጅት ይሰራል",
                "pattern": "{PERSON} በ{ORGANIZATION} ይሰራል"
            },
            "located_in": {
                "description": "Organization/event located in place",
                "amharic_desc": "ድርጅት/ክንውን በቦታ ይገኛል",
                "pattern": "{ORGANIZATION} በ{LOCATION} ይገኛል"
            },
            "happened_on": {
                "description": "Event happened on date",
                "amharic_desc": "ክንውን በቀን ተከስቷል",
                "pattern": "በ{DATE} {EVENT} ተከስቷል"
            }
        },
        "events": {
            "meeting": {
                "amharic": "ስብሰባ",
                "keywords": ["ስብሰባ", "ውይይት", "ጉባኤ", "ኮንፈረንስ"]
            },
            "election": {
                "amharic": "ምርጫ", 
                "keywords": ["ምርጫ", "ተመርጧል", "እጩ", "ምርጫ ቦርድ"]
            },
            "announcement": {
                "amharic": "ማስታወቂያ",
                "keywords": ["አወጀ", "አስታወቀ", "ይፋ አደረገ", "መልዕክት"]
            }
        }
    },
    
    "government": {
        "description": "Ethiopian government document extraction",
        "entities": {
            "officials": {
                "type": "GOVERNMENT_POSITION",
                "description": "Government officials mentioned",
                "amharic_desc": "የመንግሥት ሃላፊዎች"
            },
            "regions": {
                "type": "REGION",
                "description": "Administrative regions",
                "amharic_desc": "አስተዳደራዊ ክልሎች"
            },
            "policies": {
                "type": "POLICY",
                "description": "Government policies mentioned",
                "amharic_desc": "የመንግሥት ፖሊሲዎች"
            },
            "laws": {
                "type": "LAW",
                "description": "Laws and regulations",
                "amharic_desc": "ሕጎች እና ደንቦች"
            }
        },
        "document_types": {
            "proclamation": {
                "amharic": "አዋጅ",
                "keywords": ["አዋጅ", "ሕግ", "ደንብ"]
            },
            "directive": {
                "amharic": "መመሪያ",
                "keywords": ["መመሪያ", "መመሪያ መርጃ", "መመሪያ ሰነድ"]
            },
            "policy": {
                "amharic": "ፖሊሲ",
                "keywords": ["ፖሊሲ", "ስትራቴጂ", "ዕቅድ"]
            }
        }
    },
    
    "education": {
        "description": "Educational content and academic text extraction",
        "entities": {
            "institutions": {
                "type": "ORGANIZATION", 
                "description": "Educational institutions",
                "amharic_desc": "የትምህርት ተቋማት"
            },
            "subjects": {
                "type": "SUBJECT",
                "description": "Academic subjects",
                "amharic_desc": "የትምህርት ዘርፎች"
            },
            "degrees": {
                "type": "DEGREE",
                "description": "Academic degrees and qualifications",
                "amharic_desc": "የትምህርት ዲግሪዎች"
            }
        },
        "academic_terms": {
            "research": {
                "amharic": "ምርምር",
                "keywords": ["ምርምር", "ጥናት", "መመርመሪያ"]
            },
            "curriculum": {
                "amharic": "ሥርዓተ ትምህርት",
                "keywords": ["ሥርዓተ ትምህርት", "ትምህርት ቤተ-ምርጃ"]
            }
        }
    },
    
    "healthcare": {
        "description": "Medical and healthcare document extraction",
        "entities": {
            "medical_terms": {
                "type": "MEDICAL_TERM",
                "description": "Medical terminology",
                "amharic_desc": "የህክምና ቃላት"
            },
            "hospitals": {
                "type": "ORGANIZATION",
                "description": "Healthcare facilities",
                "amharic_desc": "የህክምና ተቋማት"
            },
            "diseases": {
                "type": "DISEASE",
                "description": "Diseases and conditions",
                "amharic_desc": "በሽታዎች"
            }
        },
        "medical_categories": {
            "diagnosis": {
                "amharic": "ምርመራ",
                "keywords": ["ምርመራ", "በሽታ", "ህመም"]
            },
            "treatment": {
                "amharic": "ሕክምና",
                "keywords": ["ሕክምና", "ማከም", "መድሀኒት"]
            }
        }
    },
    
    "culture": {
        "description": "Ethiopian cultural content extraction",
        "entities": {
            "cultural_practices": {
                "type": "CULTURAL_EVENT",
                "description": "Cultural practices and traditions",
                "amharic_desc": "ባህላዊ ወጎች"
            },
            "traditional_foods": {
                "type": "FOOD",
                "description": "Traditional Ethiopian foods",
                "amharic_desc": "ባህላዊ ምግቦች"
            },
            "languages": {
                "type": "LANGUAGE",
                "description": "Ethiopian languages",
                "amharic_desc": "የኢትዮጵያ ቋንቋዎች"
            }
        },
        "cultural_terms": {
            "festival": {
                "amharic": "በዓል",
                "keywords": ["በዓል", "ጎደና", "አከባበር"]
            },
            "tradition": {
                "amharic": "ወግ",
                "keywords": ["ወግ", "ባህል", "ልማድ"]
            }
        }
    }
}

# Few-shot examples for training LangExtract
AMHARIC_EXAMPLES = {
    "news_example": {
        "input_text": """
        ጠቅላይ ሚኒስትር ዶ/ር አብይ አሕመድ ዛረ በመንግሥት ቤት አዲስ አበባ ውስጥ ከተለያዩ ሚኒስትሮች ጋር ስብሰባ ተካሂዷል። 
        ስብሰባው በጥር 15 ቀን 2016 ዓ.ም ነው የተካሄደው። በስብሰባው የኢትዮጵያ ኢኮኖሚ ልማት ተወያይተዋል።
        """,
        "expected_output": {
            "people": ["ጠቅላይ ሚኒስትር ዶ/ር አብይ አሕመድ"],
            "organizations": ["መንግሥት ቤት"],
            "locations": ["አዲስ አበባ", "ኢትዮጵያ"],
            "dates": ["ጥር 15 ቀን 2016 ዓ.ም"],
            "events": [{"type": "meeting", "description": "ስብሰባ"}],
            "relationships": [
                {"type": "works_at", "person": "ዶ/ር አብይ አሕመድ", "organization": "መንግሥት ቤት"},
                {"type": "located_in", "organization": "መንግሥት ቤት", "location": "አዲስ አበባ"},
                {"type": "happened_on", "event": "ስብሰባ", "date": "ጥር 15 ቀን 2016 ዓ.ም"}
            ]
        }
    },
    
    "government_example": {
        "input_text": """
        የኢትዮጵያ ፌዴራላዊ ዲሞክራሲያዊ ሪፐብሊክ ህዝብ ተወካዮች ምክር ቤት አዲስ አዋጅ አወጣ። 
        አዋጁ ሁ.ቁ 1250/2016 ዓ.ም የሚል ነው። የትምህርት ዘርፍን ያሻሽላል።
        """,
        "expected_output": {
            "officials": ["ህዝብ ተወካዮች ምክር ቤት"],
            "laws": ["አዋጅ ሁ.ቁ 1250/2016 ዓ.ም"],
            "regions": ["ኢትዮጵያ ፌዴራላዊ ዲሞክራሲያዊ ሪፐብሊክ"],
            "document_types": [{"type": "proclamation", "title": "አዋጅ ሁ.ቁ 1250/2016"}],
            "subject_areas": ["የትምህርት ዘርፍ"]
        }
    }
}

def get_schema_by_domain(domain: str) -> Dict[str, Any]:
    """Get extraction schema for specific domain."""
    return AMHARIC_SCHEMAS.get(domain, AMHARIC_SCHEMAS["news"])

def get_entity_patterns(entity_type: str) -> List[str]:
    """Get patterns for specific entity type."""
    return AMHARIC_ENTITY_TYPES.get(entity_type, {}).get("patterns", [])

def get_examples_by_domain(domain: str) -> Dict[str, Any]:
    """Get few-shot examples for specific domain."""
    example_key = f"{domain}_example"
    return AMHARIC_EXAMPLES.get(example_key, AMHARIC_EXAMPLES["news_example"])
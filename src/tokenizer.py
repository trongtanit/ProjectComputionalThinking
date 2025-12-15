"""
TEXT PREPROCESSING MODULE
=========================
Vietnamese text tokenization
"""

from typing import List
from unidecode import unidecode
import re


class VietnameseTokenizer:
    """Custom tokenizer for Vietnamese text"""
    
    STOP_WORDS = {
        'ngon', 're', 'tim', 'quan', 'ha', 'noi', 'sai', 'gon', 'va',
        'cho', 'toi', 'nam', 'bac', 'cao', 'cap', 'doc', 'dao', 'rat',
        'nhieu', 'thuong', 'thuc', 'an', 'uong', 'de', 'lam', 'viec',
        'mot', 'hai', 'ba', 'bon', 'nam', 'sau', 'bay', 'tam', 'chin', 'muoi',
        'nguoi', 'ta', 'day', 'kia'
    }
    
    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenize Vietnamese text with stop word removal"""
        if not text:
            return []
        
        # Normalize Unicode characters
        normalized = unidecode(text).lower()
        
        # Extract alphanumeric tokens
        tokens = re.findall(r'[\w_]+', normalized)
        
        # Remove stop words and short tokens
        filtered = [t for t in tokens if t not in cls.STOP_WORDS and len(t) > 1]
        
        return filtered
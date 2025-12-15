"""
FILTER MODULE
=============
Filter restaurants based on user preferences
"""

from typing import List, Tuple
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class RestaurantFilter:
    """Filter restaurants based on user preferences"""
    
    @staticmethod
    def apply_filters(docs: List[Tuple[Document, float]], 
                     district: str = None,
                     max_price: int = None,
                     min_rating: float = None,
                     category: str = None) -> List[Tuple[Document, float]]:
        """Apply multiple filters to search results"""
        
        filtered = docs
        initial_count = len(filtered)
        
        logger.info(f"\n{'='*60}")
        logger.info("FILTER MODULE: Applying user preferences")
        logger.info(f"{'='*60}")
        logger.info(f"Initial results: {initial_count}")
        
        if district:
            filtered = [(doc, score) for doc, score in filtered 
                       if district.lower() in doc.metadata.get('district', '').lower()]
            logger.info(f"After district filter ('{district}'): {len(filtered)} results")
        
        if max_price:
            filtered = [(doc, score) for doc, score in filtered 
                       if doc.metadata.get('price_max', 999999999) <= max_price]
            logger.info(f"After price filter (<= {max_price:,} VND): {len(filtered)} results")
        
        if min_rating:
            filtered = [(doc, score) for doc, score in filtered 
                       if doc.metadata.get('rating', 0) >= min_rating]
            logger.info(f"After rating filter (>= {min_rating}): {len(filtered)} results")
        
        if category:
            filtered = [(doc, score) for doc, score in filtered 
                       if category.lower() in doc.metadata.get('category', '').lower()]
            logger.info(f"After category filter ('{category}'): {len(filtered)} results")
        
        logger.info(f"Final filtered results: {len(filtered)}/{initial_count}")
        
        return filtered
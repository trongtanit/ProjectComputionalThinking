"""
HYBRID RETRIEVAL MODULE
=======================
Ensemble Retriever using Reciprocal Rank Fusion (RRF)
"""

from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class EnsembleRetriever:
    """
    Custom Ensemble Retriever using Reciprocal Rank Fusion (RRF)
    Combines BM25 (keyword) and Vector (semantic) search results
    """
    
    def __init__(self, retrievers: List, weights: List[float] = None, k: int = 60):
        """
        Args:
            retrievers: List of retriever objects
            weights: Weight for each retriever (default: equal weights)
            k: RRF constant (default: 60 as per original paper)
        """
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.k_rrf = k
        
        if len(self.retrievers) != len(self.weights):
            raise ValueError(f"Mismatch: {len(retrievers)} retrievers but {len(self.weights)} weights")
        
        logger.info(f"ENSEMBLE RETRIEVER: Initialized with {len(retrievers)} retrievers")
        logger.info(f"WEIGHTS: {dict(zip(['BM25', 'Vector'], self.weights))}")
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve and merge results using RRF
        
        RRF Formula: score(d) = sum(w_i / (k + rank_i(d)))
        """
        # logger.info(f"\n{'='*60}")
        logger.info("HYBRID SEARCH: Starting retrieval process")
        # logger.info(f"{'='*60}")
        
        rank_map = {}  # document -> RRF score
        doc_map = {}   # document content -> Document object
        
        for idx, (retriever, weight) in enumerate(zip(self.retrievers, self.weights)):
            retriever_name = "BM25" if idx == 0 else "Vector"
            logger.info(f"\n[RETRIEVER {idx+1}] {retriever_name} Search (weight={weight})")
            
            try:
                # Get documents from retriever
                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(query)
                elif hasattr(retriever, "_get_relevant_documents"):
                    docs = retriever._get_relevant_documents(query, run_manager=None)
                else:
                    logger.warning(f"  Skipped: No retrieval method found")
                    continue
            except TypeError:
                docs = retriever._get_relevant_documents(query, run_manager=None)
            
            logger.info(f"  Retrieved: {len(docs)} documents")
            
            # Apply RRF scoring
            for rank, doc in enumerate(docs, 1):
                key = doc.page_content
                
                if key not in doc_map:
                    doc_map[key] = doc
                    rank_map[key] = 0.0
                
                # RRF score calculation
                rrf_score = weight * (1 / (self.k_rrf + rank))
                rank_map[key] += rrf_score
                
                if rank <= 3:  # Show top 3 for debugging
                    logger.info(f"    Rank {rank}: {doc.metadata.get('name', 'Unknown')[:30]} (RRF: {rrf_score:.4f})")
        
        # Sort by combined RRF score
        sorted_items = sorted(rank_map.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"\n[FUSION] Combined {len(sorted_items)} unique documents")
        logger.info("Top 3 after RRF fusion:")
        for idx, (key, score) in enumerate(sorted_items[:3], 1):
            doc_name = doc_map[key].metadata.get('name', 'Unknown')
            logger.info(f"  {idx}. {doc_name[:40]} (RRF Score: {score:.4f})")
        
        return [doc_map[k] for k, _ in sorted_items]
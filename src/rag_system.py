"""
RESTAURANT RECOMMENDATION SYSTEM - RAG IMPLEMENTATION
=========================================================
Architecture: Hybrid Search (BM25 + Vector Embeddings) + LLM Generation
"""

from google import genai
import torch
from typing import List, Dict, Tuple
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from unidecode import unidecode

from config import (
    VECTOR_DB_PATH, EMBEDDING_MODEL, LLM_MODEL,
    DEFAULT_TOP_K, DISTANCE_THRESHOLD,
    BM25_WEIGHT, VECTOR_WEIGHT, RRF_K
)
from tokenizer import VietnameseTokenizer
from ensemble_retriever import EnsembleRetriever
from filter import RestaurantFilter

# =====================================================
# LOGGING CONFIGURATION
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =====================================================
# BM25 IMPORT
# =====================================================
HYBRID_AVAILABLE = False
BM25Retriever = None
try:
    from langchain_community.retrievers import BM25Retriever
    HYBRID_AVAILABLE = True
    logger.info("MODULE LOADED: BM25Retriever successfully imported")
except ImportError:
    try:
        from langchain_community.retrievers.bm25 import BM25Retriever
        HYBRID_AVAILABLE = True
        logger.info("MODULE LOADED: BM25Retriever successfully imported (alternative path)")
    except ImportError:
        logger.warning("MODULE MISSING: BM25 not available. Using Vector Search only.")
        logger.warning("Install with: pip install rank-bm25 langchain-community")


# =====================================================
# MAIN RAG SYSTEM
# =====================================================

class RestaurantRAG:
    """
    Main RAG System for Restaurant Recommendations
    
    Pipeline:
      1. Hybrid Search (BM25 + Vector)
      2. Filter Application
      3. Context Formatting
      4. LLM Generation (with rejection detection)
    """
    
    def __init__(self, vectorstore_path: str = None):
        # logger.info("\n" + "="*60)
        logger.info("INITIALIZING RESTAURANT RAG SYSTEM")
        # logger.info("="*60)
        
        if vectorstore_path is None:
            vectorstore_path = str(VECTOR_DB_PATH)
        
        # Step 1: Initialize Embeddings
        logger.info("\n[STEP 1] Loading Embedding Model")
        device = self._select_device()
        logger.info(f"  Device: {device}")
        logger.info(f"  Model: {EMBEDDING_MODEL}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': device}
        )
        logger.info("  Status: Embeddings loaded successfully")
        
        # Step 2: Initialize Vector Store
        logger.info("\n[STEP 2] Loading Vector Database")
        logger.info(f"  Path: {vectorstore_path}")
        
        self.vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=self.embeddings,
            collection_name='restaurants'
        )
        
        doc_count = self.vectorstore._collection.count()
        logger.info(f"  Documents in database: {doc_count}")
        
        # Step 3: Initialize Hybrid Search
        self.hybrid_available = HYBRID_AVAILABLE
        
        if self.hybrid_available:
            logger.info("\n[STEP 3] Initializing Hybrid Search")
            all_docs = self._load_all_documents()
            
            if not all_docs:
                logger.error("  ERROR: No documents found for BM25 index")
                self.hybrid_available = False
                self.ensemble_retriever = None
            else:
                logger.info(f"  Loading {len(all_docs)} documents into BM25...")
                self.bm25_retriever = BM25Retriever.from_documents(
                    all_docs, 
                    preprocess_func=VietnameseTokenizer.tokenize
                )
                self.bm25_retriever.k = DEFAULT_TOP_K
                
                vector_retriever = self.vectorstore.as_retriever(
                    search_kwargs={'k': DEFAULT_TOP_K}
                )
                
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.bm25_retriever, vector_retriever],
                    weights=[BM25_WEIGHT, VECTOR_WEIGHT],
                    k=RRF_K
                )
                logger.info("  Status: Hybrid search ready")
        else:
            logger.info("\n[STEP 3] Using Vector Search Only")
            self.ensemble_retriever = None
        
        # Step 4: Initialize LLM
        logger.info("\n[STEP 4] Initializing LLM")
        logger.info(f"  Model: {LLM_MODEL}")
        
        try:
            self.client = genai.Client()
            self.llm_model = LLM_MODEL
            logger.info("  Status: LLM client ready")
        except Exception as e:
            logger.error(f"  ERROR: Failed to initialize LLM - {e}")
            raise e
        
        self.system_prompt = self._create_system_prompt()
        
        # logger.info("\n" + "="*60)
        logger.info("RAG SYSTEM READY")
        # logger.info("="*60 + "\n")
    
    def _select_device(self) -> str:
        """Select best available device for embeddings"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_all_documents(self) -> List[Document]:
        """Load all documents from vector store for BM25 indexing"""
        try:
            all_data = self.vectorstore.get(include=['metadatas', 'documents'])
            docs = []
            for content, meta in zip(all_data['documents'], all_data['metadatas']):
                docs.append(Document(page_content=content, metadata=meta))
            return docs
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for LLM"""
        return """Ban la tro ly tu van nha hang tai TP.HCM.

QUY TAC:
1. CHI dung thong tin tu [DANH SACH NHA HANG]
2. De xuat 2-3 quan phu hop nhat
3. Khong bi danh ten/gia
4. Tra loi ngan gon, tu nhien

Tra loi bang tieng Viet co dau."""
    
    def _check_llm_rejection(self, answer: str) -> bool:
        """Check if LLM rejected the query based on response content"""
        if not answer:
            return False
            
        rejection_phrases = [
            'chi co the ho tro',
            'khong the giup',
            'chi tu van ve nha hang',
            'chi ho tro tim',
            'xin loi',
            'tiec',
            'khong phu hop',
            'khong lien quan',
            'chi ho tro cac cau hoi'
        ]
        
        answer_normalized = unidecode(answer.lower())
        return any(phrase in answer_normalized for phrase in rejection_phrases)
    
    def hybrid_search(self, query: str, k: int = DEFAULT_TOP_K) -> List[Tuple[Document, float]]:
        """Perform hybrid search combining BM25 and vector similarity"""
        
        if not self.hybrid_available:
            logger.info("[SEARCH] Using Vector Search only")
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return results
        
        # Use ensemble retriever
        self.bm25_retriever.k = k
        docs = self.ensemble_retriever.get_relevant_documents(query)
        
        # Calculate similarity scores
        results_with_scores = []
        for doc in docs[:k]:
            sim = self.vectorstore.similarity_search_with_score(
                doc.page_content[:200], k=1
            )
            score = sim[0][1] if sim else 0.5
            results_with_scores.append((doc, score))
        
        return results_with_scores
    
    def format_context(self, search_results: List[Tuple[Document, float]]) -> str:
        """Format search results as context for LLM"""
        parts = []
        for i, (doc, score) in enumerate(search_results[:5], 1):
            m = doc.metadata
            try:
                price = f"{int(m['price_min']):,} - {int(m['price_max']):,} VND"
            except:
                price = f"{m['price_min']} - {m['price_max']} VND"
            
            content = doc.page_content[:200]
            parts.append(
                f"[{i}] {m['name']} - {m['district']}\n"
                f"Gia: {price} | Rating: {m['rating']}/5\n"
                f"Mo ta: {content}..."
            )
        return "\n\n".join(parts)
    
    def ask(self, user_query: str, 
            top_k: int = DEFAULT_TOP_K,
            min_similarity: float = DISTANCE_THRESHOLD,
            district: str = None,
            max_price: int = None,
            min_rating: float = None,
            category: str = None) -> Dict:
        """
        Main entry point for restaurant recommendations
        
        Args:
            user_query: User's search query
            top_k: Number of results to retrieve
            min_similarity: Minimum similarity threshold
            district: Filter by district
            max_price: Maximum price filter
            min_rating: Minimum rating filter
            category: Category filter
        
        Returns:
            Dict with answer, sources, and metadata
        """
        
        # logger.info("\n" + "="*60)
        logger.info(f"NEW QUERY: {user_query}")
        # logger.info("="*60)
        
        # Step 1: Hybrid Search
        logger.info(f"\n[STEP 1] Hybrid Search (top_k={top_k})")
        search_results = self.hybrid_search(user_query, k=top_k * 2)  # Get more for filtering
        
        if not search_results:
            logger.warning("  Result: No results found")
            return {
                "answer": "",
                "sources": [],
                "best_score": 1.0
            }
        
        best_score = search_results[0][1]
        logger.info(f"  Best similarity score: {best_score:.4f}")
        
        # Step 2: Check Similarity Threshold
        if best_score > min_similarity:
            logger.warning(f"  Result: Best score ({best_score:.4f}) exceeds threshold ({min_similarity})")
            return {
                "answer": "",
                "sources": [],
                "best_score": best_score
            }
        
        # Step 3: Apply Filters
        filtered_results = RestaurantFilter.apply_filters(
            search_results,
            district=district,
            max_price=max_price,
            min_rating=min_rating,
            category=category
        )
        
        if not filtered_results:
            logger.warning("  Result: All results filtered out")
            return {
                "answer": "",
                "sources": [],
                "best_score": best_score
            }
        
        # Take top K after filtering
        filtered_results = filtered_results[:top_k]
        
        # Step 4: Format Context
        logger.info(f"\n[STEP 2] Formatting Context for LLM")
        context = self.format_context(filtered_results)
        logger.info(f"  Context length: {len(context)} characters")
        
        # Step 5: Generate Answer with LLM
        logger.info("\n[STEP 3] Generating Answer with LLM")
        full_prompt = f"""{self.system_prompt}

DANH SACH NHA HANG:
{context}

CAU HOI: {user_query}

TRA LOI:"""
        
        try:
            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=full_prompt
            )
            answer = response.text.strip() if response.text else None
            
            # Check if LLM rejected the query
            if answer and self._check_llm_rejection(answer):
                logger.warning("  LLM Response: Query rejected by LLM")
                return {
                    "answer": answer,
                    "sources": [],
                    "best_score": best_score,
                    "warning": "llm_rejection"
                }
            
            logger.info("  Status: Answer generated successfully")
                
        except Exception as e:
            logger.error(f"  ERROR: LLM generation failed - {e}")
            answer = None
        
        if not answer:
            logger.warning("  No valid answer generated")
            return {
                "answer": "",
                "sources": [],
                "best_score": best_score
            }
        
        # Step 6: Format Sources
        logger.info("\n[STEP 4] Formatting Sources")
        sources = []
        for doc, score in filtered_results[:5]:
            try:
                price_range = f"{int(doc.metadata['price_min']):,} - {int(doc.metadata['price_max']):,} VND"
            except:
                price_range = f"{doc.metadata['price_min']} - {doc.metadata['price_max']} VND"
            
            sources.append({
                'name': doc.metadata['name'],
                'district': doc.metadata['district'],
                'price_range': price_range,
                'rating': doc.metadata['rating'],
                'category': doc.metadata.get('category', ''),
                'specialties': doc.metadata.get('specialties', '')[:80],
                'similarity_score': round(score, 4)
            })
        
        logger.info(f"  Returning {len(sources)} sources")
        # logger.info("\n" + "="*60)
        logger.info("QUERY COMPLETED")
        # logger.info("="*60 + "\n")
        
        return {
            "answer": answer,
            "sources": sources,
            "best_score": best_score
        }
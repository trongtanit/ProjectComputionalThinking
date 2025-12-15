"""
CONFIGURATION MODULE
====================
All constants and configuration settings
"""

from pathlib import Path

# =====================================================
# PATHS
# =====================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_PATH = DATA_DIR / "chroma_db"

# =====================================================
# MODEL SETTINGS
# =====================================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "gemini-2.5-flash"

# =====================================================
# SEARCH SETTINGS
# =====================================================
DEFAULT_TOP_K = 7
DISTANCE_THRESHOLD = 0.15

# =====================================================
# HYBRID SEARCH WEIGHTS
# =====================================================
BM25_WEIGHT = 0.6
VECTOR_WEIGHT = 0.4

# =====================================================
# RRF SETTINGS
# =====================================================
RRF_K = 60  # Reciprocal Rank Fusion constant
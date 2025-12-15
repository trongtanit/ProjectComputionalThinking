"""
FLASK API SERVER - Restaurant Recommendation System
====================================================
Endpoints:
  - GET  /           : Serve frontend HTML
  - POST /api/search : Process search queries with filters
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from rag_system import RestaurantRAG
from dotenv import load_dotenv
import logging
import time

# =====================================================
# CONFIGURATION
# =====================================================

load_dotenv()

app = Flask(__name__)
CORS(app)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =====================================================
# SYSTEM INITIALIZATION
# =====================================================

# logger.info("\n" + "="*60)
logger.info("FLASK SERVER INITIALIZATION")
# logger.info("="*60)

try:
    rag = RestaurantRAG()
    logger.info("RAG System initialized successfully")
except Exception as e:
    logger.error(f"CRITICAL ERROR: Failed to initialize RAG system - {e}")
    raise e

# logger.info("="*60 + "\n")


# =====================================================
# REQUEST LOGGER MIDDLEWARE
# =====================================================

@app.before_request
def log_request_info():
    """Log incoming request details"""
    if request.path == '/api/search' and request.method == 'POST':
        # logger.info(f"\n{'='*60}")
        logger.info(f"INCOMING REQUEST: {request.method} {request.path}")
        logger.info(f"Client IP: {request.remote_addr}")
        # logger.info(f"{'='*60}")


# =====================================================
# ROUTES
# =====================================================

@app.route('/')
def index():
    """Serve frontend HTML page"""
    logger.info("Serving index.html")
    return send_from_directory('.', 'index.html')


@app.route('/api/search', methods=['POST', 'OPTIONS'])
def search():
    """
    Search API endpoint with filtering support
    
    Request Body:
        {
            "query": str,          # Required: search query
            "district": str,       # Optional: district filter
            "maxPrice": int,       # Optional: max price filter
            "minRating": float,    # Optional: min rating filter
            "category": str        # Optional: category filter
        }
    
    Response:
        {
            "answer": str,
            "sources": [
                {
                    "name": str,
                    "district": str,
                    "price_range": str,
                    "rating": float,
                    "category": str,
                    "specialties": str,
                    "similarity_score": float
                }
            ],
            "best_score": float,
            "processing_time": float
        }
    """
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    start_time = time.time()
    
    try:
        # Parse request
        data = request.json
        if not data:
            logger.warning("Empty request body")
            return jsonify({"error": "Request body is required"}), 400
        
        query = data.get('query', '').strip()
        if not query:
            logger.warning("Empty query string")
            return jsonify({"error": "Query parameter is required"}), 400
        
        logger.info(f"Query: '{query}'")
        
        # Extract filters
        filters = {}
        
        if data.get('district'):
            filters['district'] = data['district']
            logger.info(f"Filter - District: {filters['district']}")
        
        if data.get('maxPrice'):
            filters['max_price'] = int(data['maxPrice'])
            logger.info(f"Filter - Max Price: {filters['max_price']:,} VND")
        
        if data.get('minRating'):
            filters['min_rating'] = float(data['minRating'])
            logger.info(f"Filter - Min Rating: {filters['min_rating']}")
        
        if data.get('category'):
            filters['category'] = data['category']
            logger.info(f"Filter - Category: {filters['category']}")
        
        # Process query
        result = rag.ask(query, **filters)
        
        # Add processing time
        processing_time = time.time() - start_time
        result['processing_time'] = round(processing_time, 2)
        
        logger.info(f"Processing completed in {processing_time:.2f}s")
        logger.info(f"Returned {len(result.get('sources', []))} results")
        
        return jsonify(result)
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Restaurant RAG API",
        "version": "1.0"
    })


# =====================================================
# ERROR HANDLERS
# =====================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


# =====================================================
# MAIN
# =====================================================

if __name__ == '__main__':
    # logger.info("\n" + "="*60)
    logger.info("STARTING FLASK SERVER")
    # logger.info("="*60)
    logger.info("Server URL: http://127.0.0.1:5000")
    logger.info("API Endpoint: http://127.0.0.1:5000/api/search")
    # logger.info("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='127.0.0.1')
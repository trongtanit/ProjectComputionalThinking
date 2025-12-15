"""
TẠO VECTOR DATABASE - FINAL OPTIMIZED VERSION
- Gộp data (star.json) và đảm bảo ID duy nhất cho mỗi document (fix lỗi 730 quán)
- Embed specialties + category vào content để tăng cường tìm kiếm (Vector Search)
"""

import json
import os
import torch
import shutil
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Dict

# ============= CẤU HÌNH =============
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_PATH = DATA_DIR / "chroma_db"
# Chỉ cần trỏ đến file star.json duy nhất đã gộp data
JSON_FILE = DATA_DIR / "star.json" 

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============= FUNCTIONS =============

def load_documents(json_path: Path) -> List[Dict]:
    """Load dữ liệu từ file JSON (đã gộp)"""
    print(f"📥 Loading documents from: {json_path}")
    
    if not json_path.exists():
        raise FileNotFoundError(f"File không tồn tại: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Xử lý linh hoạt cho cả format {documents: [...]} và format [...]
    if isinstance(data, dict) and 'documents' in data:
        documents = data['documents']
    elif isinstance(data, list):
        documents = data
    else:
        raise ValueError("JSON format không hợp lệ!")
    
    print(f"✅ Loaded {len(documents)} documents (dự kiến khoảng 830)")
    return documents


def create_langchain_documents(documents: List[Dict]) -> List[Document]:
    """
    Chuyển đổi documents sang LangChain Document format.
    ✅ Tối ưu: Đảm bảo ID duy nhất cho mỗi document
    """
    print("\n📄 Converting to LangChain Documents...")
    
    langchain_docs = []
    seen_ids = set() # Set để kiểm tra ID trùng lặp
    
    for i, doc in enumerate(documents):
        doc_id = None # Khởi tạo
        try:
            meta = doc.get("metadata", {}) 

            # ===== 🔥 FIX QUAN TRỌNG: TẠO ID DUY NHẤT (Deduplication) =====
            # 1. Ưu tiên: metadata['id'] -> 2. trường 'id' chính -> 3. Tạo temp ID
            doc_id = meta.get("id", doc.get('id', f"doc_{i}"))
            
            # Nếu ID đã được sử dụng, thêm hậu tố unique
            if doc_id in seen_ids:
                temp_id = f"{doc_id}_v{len(seen_ids)}"
                print(f"   ⚠️ Duplicate ID '{doc_id}' detected at index {i}, using '{temp_id}'")
                doc_id = temp_id
            
            seen_ids.add(doc_id)
            # =============================================================

            def stringify_list(value):
                """Convert list to pipe-separated string"""
                if isinstance(value, list):
                    return "|".join(str(v) for v in value)
                if value is None:
                    return ""
                return str(value)

            # Chuẩn bị Metadata (Lấy giá trị mặc định nếu thiếu)
            metadata = {
                "id": doc_id, 
                "name": meta.get("name", "N/A"),
                "category": meta.get("category", "N/A"),
                "district": meta.get("district", "N/A"),
                "price_min": int(meta.get("price_min", 0)),
                "price_max": int(meta.get("price_max", 0)),
                "price_range": meta.get("price_range", "N/A"),
                "rating": float(meta.get("rating", 0.0)),
                "vibe_tags": stringify_list(meta.get("vibe_tags", [])),
                "specialties": stringify_list(meta.get("specialties", "")),
            }

            raw_content = doc.get('content', doc.get('page_content', ''))

            # ===== ✅ TỐI ƯU: TẠO CONTENT ĐẦY ĐỦ ĐỂ EMBED (Content Augmentation) =====
            specialties_text = metadata["specialties"].replace("|", ", ") if metadata["specialties"] else ""
            
            enhanced_content = f"""Tên quán: {metadata['name']}
Loại hình: {metadata['category']}
Món đặc sản: {specialties_text}

{raw_content}"""

            langchain_doc = Document(
                page_content=enhanced_content, 
                metadata=metadata
            )

            langchain_docs.append(langchain_doc)

        except Exception as e:
            print(f"⚠️ Lỗi xử lý document tại index {i}. ID: {doc_id}. Lỗi: {e}")
            continue

    print(f"✅ Converted {len(langchain_docs)} documents.")
    print(f"   (Số ID duy nhất được index: {len(seen_ids)})")
    
    if langchain_docs:
        sample = langchain_docs[0]
        print(f"\n📋 Sample enhanced content:")
        print("─" * 60)
        print(sample.page_content[:300] + "...")
        print("─" * 60)
    
    return langchain_docs


def create_embeddings_model():
    """Tạo embedding model"""
    print("\n🤖 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': EMBEDDING_DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings


def create_vector_store(
    documents: List[Document],
    embeddings,
    persist_dir: Path
) -> Chroma:
    """Tạo và lưu vector store"""
    print("\n🗄️ Creating vector store...")
    
    # Xóa DB cũ để tạo lại với data đã gộp
    if persist_dir.exists():
        print("   ⚠️ Removing old database...")
        shutil.rmtree(persist_dir)
    
    # Tạo vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name='restaurants',
        # Tối ưu hóa cài đặt HNSW cho hiệu suất Cosine Similarity
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"✅ Vector store created! Total count: {vectorstore._collection.count()}")
    return vectorstore


def test_search(vectorstore: Chroma):
    """Test tìm kiếm"""
    print("\n" + "="*60)
    print("🔍 TESTING VECTOR SEARCH")
    print("="*60)
    
    test_queries = [
        "Tìm quán phở ngon giá rẻ", # Mục tiêu test chính
        "Quán cafe yên tĩnh để làm việc",
        "Bún bò Huế ở Quận 10",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        results = vectorstore.similarity_search_with_score(query, k=3)
        
        if not results:
            print(f"   ❌ Không tìm thấy kết quả!")
            continue
        
        print(f"   ✅ Found {len(results)} results:\n")
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"   {i}. {doc.metadata.get('name', 'N/A')} ({doc.metadata.get('district', 'N/A')})")
            print(f"      📊 Distance: {score:.4f}")
            print(f"      🏷️ {doc.metadata.get('category', 'N/A')}")
            
            if doc.metadata.get('specialties'):
                specs = doc.metadata['specialties'].split('|')[:3]
                print(f"      🍜 Món: {', '.join(specs)}")
            print()


# ============= MAIN =============

def main():
    """Main function"""
    
    print("="*60)
    print("🚀 CREATING VECTOR DATABASE - FINAL VERSION")
    print("="*60)
    
    try:
        # STEP 1: Load documents (từ file star.json đã gộp)
        documents = load_documents(JSON_FILE)
        
        # STEP 2: Convert to LangChain format (với ID duy nhất và enhanced content)
        langchain_docs = create_langchain_documents(documents)
        
        # STEP 3: Create embedding model
        embeddings = create_embeddings_model()
        
        # STEP 4: Create vector store (Xóa DB cũ và tạo lại)
        vectorstore = create_vector_store(
            langchain_docs,
            embeddings,
            VECTOR_DB_PATH
        )
        
        # STEP 5: Test search (Kiểm tra xem quán Phở đã lên chưa)
        test_search(vectorstore)
        
        print("\n" + "="*60)
        print("✅ SUCCESS! Vector database đã được tạo mới với ID duy nhất.")
        print("="*60)
        print("\n💡 Bây giờ hãy chạy lại rag_system.py để test Hybrid Search (BM25 + Vector)!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
"""
Module tạo embeddings và quản lý FAISS vector store.
Sử dụng sentence-transformers để tạo embeddings local (miễn phí),
lưu trữ vào FAISS index trên disk.
Không dependency LangChain — dùng trực tiếp FAISS + sentence-transformers.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# from config import VECTORSTORE_DIR, EMBEDDING_MODEL
# from document_loader import load_documents

from src.config import VECTORSTORE_DIR, EMBEDDING_MODEL
from src.document_loader import load_documents


# Tên file lưu trữ
INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.pkl"


def get_embedding_model() -> SentenceTransformer:
    """
    Khởi tạo embedding model (chạy local, không cần API key).
    Model mặc định: all-MiniLM-L6-v2 (~80MB, nhanh, chất lượng tốt)
    """
    print(f"🧠 Đang tải embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"   ✅ Đã tải embedding model thành công")
    return model


def build_vector_store(force_rebuild: bool = False) -> Tuple[faiss.Index, List[Dict], SentenceTransformer]:
    """
    Xây dựng FAISS vector store từ tài liệu .docx.
    Nếu đã có index trên disk và không force_rebuild, sẽ load từ disk.

    Returns:
        Tuple: (faiss_index, metadata_list, embedding_model)
    """
    model = get_embedding_model()
    index_path = VECTORSTORE_DIR / INDEX_FILE
    metadata_path = VECTORSTORE_DIR / METADATA_FILE

    # Nếu đã có index và không cần rebuild → load từ disk
    if index_path.exists() and metadata_path.exists() and not force_rebuild:
        print(f"📦 Đang load vector store từ disk: {VECTORSTORE_DIR}")
        index = faiss.read_index(str(index_path))
        with open(str(metadata_path), "rb") as f:
            metadata = pickle.load(f)
        print(f"   ✅ Đã load vector store thành công ({index.ntotal} vectors)")
        return index, metadata, model

    # Rebuild: đọc tài liệu → tạo embeddings → lưu vào FAISS
    print("🔨 Đang xây dựng vector store mới...")
    chunks = load_documents()

    # Tạo embeddings
    texts = [chunk["content"] for chunk in chunks]
    print(f"🔄 Đang tạo embeddings cho {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Tạo FAISS index (cosine similarity với normalized vectors = inner product)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product (= cosine khi normalized)
    index.add(embeddings.astype(np.float32))

    # Lưu vào disk
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(str(metadata_path), "wb") as f:
        pickle.dump(chunks, f)

    print(f"💾 Đã lưu vector store vào: {VECTORSTORE_DIR} ({index.ntotal} vectors)")

    return index, chunks, model


# Cache global
_cache: Optional[Tuple[faiss.Index, List[Dict], SentenceTransformer]] = None


def get_vector_store(force_rebuild: bool = False) -> Tuple[faiss.Index, List[Dict], SentenceTransformer]:
    """Lấy vector store (có cache)."""
    global _cache
    if _cache is None or force_rebuild:
        _cache = build_vector_store(force_rebuild)
    return _cache


def similarity_search(query: str, k: int = 5) -> List[Dict]:
    """
    Tìm các chunks liên quan nhất đến câu hỏi.

    Args:
        query: Câu hỏi cần tìm kiếm
        k: Số lượng kết quả (mặc định 5)

    Returns:
        List[Dict]: Danh sách chunks liên quan, mỗi dict có 'content', 'source', 'score'
    """
    index, metadata, model = get_vector_store()

    # Tạo embedding cho câu hỏi
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_vector = query_embedding.astype(np.float32)

    # Tìm kiếm
    scores, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result["score"] = float(score)
            results.append(result)

    return results


if __name__ == "__main__":
    # Test: build và search
    get_vector_store(force_rebuild=True)
    results = similarity_search("captive portal là gì", k=3)
    print(f"\n--- Test search: 'captive portal là gì' ---")
    for i, doc in enumerate(results):
        print(f"\n[Kết quả {i+1}] (nguồn: {doc['source']}, score: {doc['score']:.4f})")
        print(doc["content"][:300])

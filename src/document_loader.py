"""
Module đọc và xử lý tài liệu .docx.
Trích xuất text từ tất cả file .docx trong thư mục docs/,
sau đó chia thành các chunks phù hợp cho embedding.
Không dependency LangChain — chạy thuần Python.
"""

import re
from pathlib import Path
from typing import List, Dict
from docx import Document

# from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

from src.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def extract_text_from_docx(file_path: Path) -> str:
    """
    Đọc file .docx và trả về toàn bộ text.
    Giữ nguyên cấu trúc paragraph, thêm heading markers.
    """
    doc = Document(str(file_path))
    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Thêm heading marker để giữ ngữ cảnh khi chunk
        style_name = para.style.name.lower() if para.style else ""
        if "heading 1" in style_name:
            text = f"\n## {text}\n"
        elif "heading 2" in style_name:
            text = f"\n### {text}\n"
        elif "heading" in style_name:
            text = f"\n#### {text}\n"

        paragraphs.append(text)

    return "\n".join(paragraphs)


def split_text_into_chunks(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, str]]:
    """
    Chia text thành các chunks nhỏ hơn với overlap.
    Ưu tiên cắt theo heading/paragraph boundaries.

    Args:
        text: Toàn bộ text cần chia
        source: Tên file nguồn (để metadata)
        chunk_size: Kích thước tối đa mỗi chunk (ký tự)
        chunk_overlap: Số ký tự overlap giữa các chunks

    Returns:
        List[Dict]: Mỗi dict chứa 'content' và 'source'
    """
    # Cắt theo các heading trước
    sections = re.split(r"(\n#{2,4} .+?\n)", text)
    chunks = []
    current_chunk = ""

    for section in sections:
        # Nếu thêm section vào chunk mà vẫn dưới limit → thêm
        if len(current_chunk) + len(section) <= chunk_size:
            current_chunk += section
        else:
            # Lưu chunk hiện tại
            if current_chunk.strip():
                chunks.append({
                    "content": current_chunk.strip(),
                    "source": source,
                })

            # Nếu section quá dài → chia nhỏ hơn theo câu
            if len(section) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", section)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) <= chunk_size:
                        current_chunk += sent + " "
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                "content": current_chunk.strip(),
                                "source": source,
                            })
                        current_chunk = sent + " "
            else:
                # Overlap: lấy phần cuối chunk trước làm đầu chunk mới
                if chunks and chunk_overlap > 0:
                    prev_content = chunks[-1]["content"]
                    overlap_text = prev_content[-chunk_overlap:] if len(prev_content) > chunk_overlap else prev_content
                    current_chunk = overlap_text + "\n" + section
                else:
                    current_chunk = section

    # Chunk cuối cùng
    if current_chunk.strip():
        chunks.append({
            "content": current_chunk.strip(),
            "source": source,
        })

    return chunks


def load_documents() -> List[Dict[str, str]]:
    """
    Đọc tất cả file .docx từ thư mục docs/,
    chia thành chunks và trả về danh sách.

    Returns:
        List[Dict]: Mỗi dict chứa 'content' và 'source'
    """
    # Tìm tất cả file .docx
    docx_files = list(DOCS_DIR.glob("*.docx"))

    if not docx_files:
        raise FileNotFoundError(
            f"Không tìm thấy file .docx nào trong thư mục {DOCS_DIR}. "
            "Hãy đặt tài liệu vào thư mục docs/"
        )

    print(f"📄 Tìm thấy {len(docx_files)} file .docx:")
    for f in docx_files:
        print(f"   - {f.name}")

    # Trích xuất text và chia chunks
    all_chunks = []
    for file_path in docx_files:
        text = extract_text_from_docx(file_path)
        print(f"   ✅ Đã đọc: {file_path.name} ({len(text)} ký tự)")

        chunks = split_text_into_chunks(text, source=file_path.name)
        all_chunks.extend(chunks)

    print(f"✂️  Đã chia thành {len(all_chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    return all_chunks


if __name__ == "__main__":
    # Test nhanh
    docs = load_documents()
    print(f"\n--- Test: hiển thị 3 chunks đầu tiên ---")
    for i, doc in enumerate(docs[:3]):
        print(f"\n[Chunk {i+1}] (nguồn: {doc['source']})")
        print(doc["content"][:300])
        print("...")

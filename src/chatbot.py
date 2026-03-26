"""
Module chatbot chính — RAG pipeline + gọi Claude (Anthropic) qua REST API.
Nhận câu hỏi → tìm chunks liên quan từ vector store → gửi context tới LLM → trả lời.
Dùng requests gọi Claude Messages API (tương thích Python 3.8+).
"""

import json
from typing import List, Dict, Optional

import requests

from src.config import CLAUDE_API_KEY, LLM_MODEL
from src.embeddings import similarity_search, get_vector_store


# Claude Messages API endpoint
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# System prompt hướng dẫn LLM cách trả lời
SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên về hệ thống WiFi Marketing và Captive Portal của công ty LANCS Retails.

NHIỆM VỤ:
- Trả lời câu hỏi của người dùng DỰA TRÊN thông tin trong tài liệu được cung cấp bên dưới.
- Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu.
- Nếu câu hỏi nằm ngoài phạm vi tài liệu, hãy nói rõ "Thông tin này không có trong tài liệu" và gợi ý câu hỏi phù hợp hơn.
- Không bịa đặt thông tin. Chỉ trả lời dựa trên nội dung tài liệu.

PHONG CÁCH:
- Sử dụng định dạng Markdown khi cần (danh sách, bold, code block cho lệnh CLI).
- Nếu câu hỏi liên quan đến lệnh/cấu hình, hiển thị lệnh trong code block.
- Giải thích ngắn gọn nhưng đầy đủ.
- Nếu có nhiều bước, trình bày theo danh sách đánh số.

NGỮ CẢNH TÀI LIỆU:
Hệ thống bao gồm:
- Router LANCS (OpenWRT 19.07) tại IP 192.168.170.1
- Ubuntu Server (Flask) tại IP 192.168.170.3
- Captive Portal thu thập thông tin khách hàng qua WiFi
- Kiến trúc mạng: br-freewifi, iptables CAPTIVE chain, DNS trap, CGI API
"""


def _call_claude_api(system_prompt: str, user_message: str) -> str:
    """
    Gọi Claude Messages API.

    Args:
        system_prompt: System prompt
        user_message: Tin nhắn người dùng (đã bao gồm context)

    Returns:
        str: Câu trả lời từ Claude
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": LLM_MODEL,
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message}
        ],
    }

    response = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        error_data = response.json()
        error_msg = error_data.get("error", {}).get("message", response.text)
        raise Exception(f"Claude API error ({response.status_code}): {error_msg}")

    data = response.json()
    # Trích xuất text từ response
    content = data.get("content", [])
    if not content:
        raise Exception("Claude API trả về response rỗng")

    return content[0].get("text", "")


def _format_context(docs: List[Dict]) -> str:
    """Format các document chunks thành context string cho LLM."""
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.get("source", "unknown")
        context_parts.append(
            f"--- Đoạn tài liệu {i+1} (nguồn: {source}) ---\n{doc['content']}"
        )
    return "\n\n".join(context_parts)


class ChatBot:
    """
    Chatbot RAG sử dụng Claude (Anthropic).
    Tìm kiếm context từ tài liệu → gửi tới LLM → trả lời câu hỏi.
    """

    def __init__(self):
        """Khởi tạo chatbot: validate API key và load vector store."""
        if not CLAUDE_API_KEY or CLAUDE_API_KEY == "your_claude_api_key_here":
            raise ValueError(
                "❌ Chưa cấu hình CLAUDE_API_KEY!\n"
                "Hãy điền API key vào file .env\n"
                "Lấy API key tại: https://console.anthropic.com/"
            )

        # Pre-load vector store
        get_vector_store()
        self.conversation_history: List[Dict[str, str]] = []

        print(f"🤖 Chatbot đã sẵn sàng! (model: {LLM_MODEL})")

    def ask(self, question: str, k: int = 5) -> str:
        """
        Hỏi chatbot một câu hỏi.

        Args:
            question: Câu hỏi của người dùng
            k: Số lượng chunks liên quan để tìm (mặc định 5)

        Returns:
            str: Câu trả lời từ LLM
        """
        # Bước 1: Tìm các chunks liên quan từ vector store
        relevant_docs = similarity_search(question, k=k)
        context = _format_context(relevant_docs)

        # Bước 2: Tạo user message với context + lịch sử hội thoại
        history_text = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-6:]  # 3 cặp Q&A gần nhất
            history_parts = []
            for entry in recent_history:
                role = "Người dùng" if entry["role"] == "user" else "Trợ lý"
                history_parts.append(f"{role}: {entry['content']}")
            history_text = "\n\nLỊCH SỬ HỘI THOẠI GẦN ĐÂY:\n" + "\n".join(history_parts)

        user_message = (
            f"TÀI LIỆU THAM KHẢO:\n{context}"
            f"{history_text}\n\n"
            f"CÂU HỎI CỦA NGƯỜI DÙNG:\n{question}"
        )

        # Bước 3: Gọi Claude API
        try:
            answer = _call_claude_api(SYSTEM_PROMPT, user_message)
        except Exception as e:
            answer = f"❌ Lỗi khi gọi LLM: {str(e)}"

        # Bước 4: Lưu lịch sử hội thoại
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return answer

    def clear_history(self):
        """Xóa lịch sử hội thoại."""
        self.conversation_history.clear()
        print("🗑️ Đã xóa lịch sử hội thoại")


# Singleton chatbot instance
_chatbot_instance: Optional[ChatBot] = None


def get_chatbot() -> ChatBot:
    """Lấy chatbot instance (singleton pattern)."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = ChatBot()
    return _chatbot_instance


def ask(question: str) -> str:
    """Shortcut: hỏi chatbot một câu hỏi."""
    bot = get_chatbot()
    return bot.ask(question)


if __name__ == "__main__":
    # Test chatbot trong terminal
    bot = get_chatbot()

    print("\n" + "=" * 60)
    print("💬 Chatbot WiFi Marketing — Gõ 'quit' để thoát")
    print("=" * 60 + "\n")

    while True:
        question = input("👤 Bạn: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Tạm biệt!")
            break
        if not question:
            continue

        print("🤖 Đang xử lý...\n")
        answer = bot.ask(question)
        print(f"🤖 Trợ lý:\n{answer}\n")

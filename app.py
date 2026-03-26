"""
Flask Web Server — Giao diện chatbot WiFi Marketing.
Cung cấp giao diện chat web và API endpoint.
"""

import os
import sys
from pathlib import Path

# Fix protobuf compatibility với TensorFlow cũ trên Python 3.8
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Tắt warning TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, render_template, request, jsonify
from src.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from src.chatbot import get_chatbot

app = Flask(__name__)

# Khởi tạo chatbot khi app start
chatbot = None


def init_chatbot():
    """Khởi tạo chatbot (lazy loading)."""
    global chatbot
    if chatbot is None:
        print("\n🚀 Đang khởi tạo chatbot...")
        print("   (Lần đầu chạy sẽ tải embedding model, có thể mất 1-2 phút)\n")
        chatbot = get_chatbot()
    return chatbot


@app.route("/")
def index():
    """Trang chủ — giao diện chat."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    API endpoint nhận câu hỏi và trả lời.
    Request body: {"message": "câu hỏi"}
    Response: {"answer": "câu trả lời", "success": true}
    """
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"success": False, "error": "Thiếu trường 'message'"}), 400

        message = data["message"].strip()
        if not message:
            return jsonify({"success": False, "error": "Tin nhắn không được để trống"}), 400

        bot = init_chatbot()
        answer = bot.ask(message)

        return jsonify({"success": True, "answer": answer})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi server: {str(e)}"}), 500


@app.route("/api/clear", methods=["POST"])
def clear_history():
    """Xóa lịch sử hội thoại."""
    try:
        bot = init_chatbot()
        bot.clear_history()
        return jsonify({"success": True, "message": "Đã xóa lịch sử hội thoại"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Chatbot WiFi Marketing — Web Server")
    print(f"🌐 Truy cập: http://localhost:{FLASK_PORT}")
    print("=" * 60)

    # Pre-load chatbot
    init_chatbot()

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

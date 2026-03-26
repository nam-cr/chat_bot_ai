"""
Cấu hình chung cho dự án Chatbot WiFi Marketing.
Đọc biến môi trường từ file .env và cung cấp các hằng số cấu hình.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Đường dẫn gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Load biến môi trường từ .env
load_dotenv(BASE_DIR / ".env")

# --- API Keys ---
CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")

# --- Đường dẫn ---
DOCS_DIR: Path = BASE_DIR / "docs"
VECTORSTORE_DIR: Path = BASE_DIR / "vectorstore"

# --- Chunking ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Models ---
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Flask ---
FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"

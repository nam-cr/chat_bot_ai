# 🤖 Chatbot WiFi Marketing — LANCS Retails

> Chatbot AI sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** để trả lời câu hỏi dựa trên tài liệu nội bộ về hệ thống WiFi Marketing và Captive Portal.

---

## 📋 Mục Lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Luồng hoạt động](#-luồng-hoạt-động)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cách thức hoạt động (RAG Pipeline)](#-cách-thức-hoạt-động-rag-pipeline)
- [Model &amp; Embedding](#-model--embedding)
- [Cài đặt &amp; Chạy](#-cài-đặt--chạy)
- [Cấu hình](#-cấu-hình)
- [Mở rộng Knowledge Base](#-mở-rộng-knowledge-base)
- [API Reference](#-api-reference)

---

## 🎯 Giới Thiệu

**Chatbot WiFi Marketing** là chatbot AI được xây dựng để phục vụ nội bộ công ty LANCS Retails. Chatbot có khả năng:

- ✅ **Đọc hiểu tài liệu** `.docx` về hệ thống WiFi Marketing / Captive Portal
- ✅ **Trả lời câu hỏi** bằng tiếng Việt, dựa trên kiến thức trong tài liệu
- ✅ **Tìm kiếm ngữ nghĩa** (semantic search) — hiểu ý nghĩa câu hỏi, không chỉ so khớp từ khóa
- ✅ **Ghi nhớ ngữ cảnh** hội thoại (conversation memory)
- ✅ **Giao diện web** hiện đại, dark mode, responsive

### Ví dụ câu hỏi chatbot có thể trả lời:

| Câu hỏi                                               | Loại           |
| ------------------------------------------------------- | --------------- |
| "Kiến trúc hệ thống WiFi Marketing như thế nào?" | Kiến trúc     |
| "Cách khởi động lại hệ thống captive portal?"    | Vận hành      |
| "MAC whitelist hoạt động ra sao?"                    | Kỹ thuật      |
| "Tại sao tốc độ mạng qua router bị chậm?"        | Troubleshooting |
| "Cách kick một người dùng khỏi mạng WiFi?"       | Quản trị      |

---

## 🏗️ Kiến Trúc Hệ Thống

### Sơ đồ tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                        CHATBOT AI SYSTEM                        │
│                                                                 │
│  ┌──────────────┐                          ┌──────────────────┐ │
│  │  📄 Tài Liệu │    OFFLINE PIPELINE      │  💾 Vector Store │ │
│  │  (.docx)     │──────────────────────────▶│  (FAISS Index)   │ │
│  │              │  1. Đọc text              │  26 vectors      │ │
│  │  docs/       │  2. Chia chunks           │  384 dimensions  │ │
│  └──────────────┘  3. Tạo embeddings        └────────┬─────────┘ │
│                                                       │          │
│  ┌──────────────┐                          ┌──────────▼─────────┐│
│  │  👤 User     │    ONLINE PIPELINE       │  🔍 Similarity    ││
│  │  (Web Chat)  │─────────────────────────▶│  Search            ││
│  │              │  1. Nhận câu hỏi         │  Top-K chunks     ││
│  └──────┬───────┘  2. Tìm chunks liên quan └──────────┬─────────┘│
│         │                                              │          │
│         │          ┌───────────────────────┐           │          │
│         │          │  🤖 LLM (Claude)     │◀──────────┘          │
│         │◀─────────│  System Prompt       │  3. Gửi context     │
│         │          │  + Context + History  │     + câu hỏi       │
│  Trả lời│          └───────────────────────┘  4. Nhận trả lời   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Kiến trúc module

```
┌─────────────────────────────────────────────────────┐
│                    app.py (Flask)                     │
│              Web Server + API Endpoints              │
├──────────────────────┬──────────────────────────────┤
│   templates/         │       static/                │
│   index.html         │       style.css              │
│   (Giao diện chat)   │       (Dark mode CSS)        │
├──────────────────────┴──────────────────────────────┤
│                  src/chatbot.py                      │
│         RAG Pipeline + Claude API Client             │
├─────────────────────┬───────────────────────────────┤
│  src/embeddings.py  │  src/document_loader.py       │
│  FAISS + Sentence   │  python-docx + Chunking       │
│  Transformers       │                               │
├─────────────────────┴───────────────────────────────┤
│                  src/config.py                       │
│           Cấu hình (.env) + Constants                │
└─────────────────────────────────────────────────────┘
```

---

## 🔄 Luồng Hoạt Động

### 1. Luồng xử lý tài liệu (Offline — chạy 1 lần)

```
📄 File .docx ──▶ 📖 Trích xuất text ──▶ ✂️ Chia chunks ──▶ 🧠 Tạo embeddings ──▶ 💾 Lưu FAISS
     │                    │                     │                     │                    │
     │               python-docx           Recursive Split      Sentence-           faiss.write_index()
     │               Giữ heading            800 chars/chunk     Transformers
     │               structure              200 overlap         all-MiniLM-L6-v2
     ▼                    ▼                     ▼                     ▼                    ▼
  docs/*.docx      Raw text with         20-26 chunks         384-dim vectors       vectorstore/
                   ## Heading markers     per document         per chunk             index.faiss
```

**Chi tiết các bước:**

| Bước             | Module                 | Mô tả                                                                                          |
| ------------------ | ---------------------- | ------------------------------------------------------------------------------------------------ |
| 1. Đọc `.docx` | `document_loader.py` | Dùng `python-docx` đọc paragraphs, giữ heading markers (`##`, `###`)                   |
| 2. Chia chunks     | `document_loader.py` | Chia text thành chunks ~800 ký tự, overlap 200 ký tự. Ưu tiên cắt theo heading/paragraph |
| 3. Tạo embeddings | `embeddings.py`      | Model `all-MiniLM-L6-v2` chuyển mỗi chunk thành vector 384 chiều                           |
| 4. Lưu FAISS      | `embeddings.py`      | Lưu vào FAISS IndexFlatIP (Inner Product cho cosine similarity)                                |

### 2. Luồng trả lời câu hỏi (Online — mỗi lần user hỏi)

```
👤 Câu hỏi ──▶ 🧠 Embed câu hỏi ──▶ 🔍 FAISS Search ──▶ 📝 Tạo prompt ──▶ 🤖 Claude API ──▶ 💬 Trả lời
     │                 │                    │                    │                  │                │
  "Captive         all-MiniLM            Top-5 chunks         System Prompt    claude-sonnet     Markdown
   Portal          L6-v2                 + similarity         + Context        -4-20250514       formatted
   là gì?"         384-dim vector        scores               + History                          response
```

**Chi tiết các bước:**

| Bước             | Thời gian | Mô tả                                                                       |
| ------------------ | ---------- | ----------------------------------------------------------------------------- |
| 1. Embed câu hỏi | ~50ms      | Chuyển câu hỏi thành vector 384 chiều (cùng model với tài liệu)      |
| 2. FAISS Search    | ~1ms       | Tìm 5 chunks có cosine similarity cao nhất với câu hỏi                  |
| 3. Tạo prompt     | ~1ms       | Ghép: System Prompt + Context (5 chunks) + Lịch sử hội thoại + Câu hỏi |
| 4. Gọi Claude     | ~2-5s      | Gửi prompt tới Claude API, nhận câu trả lời                             |
| 5. Trả lời       | ~1ms       | Render Markdown, gửi về giao diện web                                      |

### 3. Sequence Diagram

```
User          Flask         ChatBot       Embeddings      FAISS        Claude API
 │               │              │              │             │              │
 │──POST /api/chat──▶│         │              │             │              │
 │               │──ask()──────▶│              │             │              │
 │               │              │──encode()───▶│             │              │
 │               │              │              │──search()──▶│              │
 │               │              │              │◀──top 5─────│              │
 │               │              │◀─chunks──────│             │              │
 │               │              │                                          │
 │               │              │────────POST /v1/messages────────────────▶│
 │               │              │◀───────response text────────────────────│
 │               │◀──answer─────│              │             │              │
 │◀──JSON────────│              │              │             │              │
```

---

## 🛠️ Công Nghệ Sử Dụng

| Thành phần               | Công nghệ           | Phiên bản              | Vai trò                      |
| -------------------------- | --------------------- | ------------------------ | ----------------------------- |
| **Ngôn ngữ**       | Python                | 3.8+                     | Backend chính                |
| **Đọc tài liệu** | python-docx           | 1.1.2                    | Parse file `.docx`          |
| **Embeddings**       | sentence-transformers | 3.2.1                    | Tạo vector embeddings local  |
| **Vector Store**     | FAISS (faiss-cpu)     | 1.8.0                    | Tìm kiếm tương đồng     |
| **LLM**              | Claude (Anthropic)    | claude-sonnet-4-20250514 | Sinh câu trả lời           |
| **API Client**       | requests              | 2.28+                    | Gọi Claude REST API          |
| **Web Server**       | Flask                 | 3.0+                     | Phục vụ giao diện & API    |
| **Frontend**         | HTML/CSS/JS           | -                        | Giao diện chat dark mode     |
| **Markdown**         | marked.js             | latest                   | Render Markdown trong chat    |
| **Env Config**       | python-dotenv         | 1.0+                     | Quản lý biến môi trường |

### Tại sao chọn các công nghệ này?

- **FAISS** thay vì ChromaDB/Pinecone → Chạy hoàn toàn local, không cần server riêng, nhanh
- **sentence-transformers** thay vì OpenAI embeddings → Miễn phí, chạy offline, không cần API key
- **Claude REST API** thay vì SDK → Tương thích Python 3.8 (SDK mới yêu cầu Python 3.9+)
- **Flask** thay vì FastAPI → Đơn giản, nhẹ, phù hợp internal tool

---

## 📁 Cấu Trúc Dự Án

```
chat_bot_ai/
│
├── 📋 PROMPT.md                 # Hướng dẫn cho AI Assistant khi làm việc
├── 📋 README.md                 # File này — tài liệu dự án
├── 📦 requirements.txt          # Dependencies Python
├── 🔒 .env                      # API keys (KHÔNG commit lên git)
├── 📝 .env.example              # Mẫu file .env
├── 🚫 .gitignore                # Ignore rules
│
├── 📄 docs/                     # 📚 Knowledge Base — tài liệu .docx
│   └── BaoCao_WiFiMarketing_TranDucNam.docx
│
├── 🐍 src/                      # Source code chính
│   ├── __init__.py
│   ├── config.py                # ⚙️  Cấu hình (đọc .env, constants)
│   ├── document_loader.py       # 📖 Đọc .docx → trích xuất text → chia chunks
│   ├── embeddings.py            # 🧠 Tạo embeddings + FAISS vector store
│   └── chatbot.py               # 🤖 RAG pipeline + gọi Claude API
│
├── 🌐 app.py                    # Flask web server (entry point)
│
├── 🎨 templates/
│   └── index.html               # Giao diện chat (HTML + JS)
│
├── 🎨 static/
│   └── style.css                # CSS styling (dark mode, glassmorphism)
│
└── 💾 vectorstore/              # FAISS index (auto-generated)
    ├── index.faiss              # Vector index
    └── metadata.pkl             # Metadata chunks
```

### Mô tả chi tiết từng module

#### `src/config.py` — Cấu hình

Đọc biến môi trường từ `.env`, cung cấp constants cho toàn bộ ứng dụng:

- `CLAUDE_API_KEY` — API key cho Claude
- `DOCS_DIR` — Đường dẫn thư mục tài liệu
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — Cấu hình chunking
- `LLM_MODEL` / `EMBEDDING_MODEL` — Model names

#### `src/document_loader.py` — Đọc tài liệu

- Đọc tất cả file `.docx` trong `docs/` bằng `python-docx`
- Giữ nguyên heading structure (H1 → `##`, H2 → `###`)
- Chia text thành chunks theo heading/paragraph boundaries
- Hỗ trợ overlap giữa các chunks để không mất ngữ cảnh

#### `src/embeddings.py` — Embeddings & Vector Store

- Tải model `all-MiniLM-L6-v2` (chạy local, ~90MB)
- Tạo vector 384 chiều cho mỗi chunk
- FAISS IndexFlatIP (Inner Product = cosine khi normalized)
- Lưu/load index từ disk (cache để không rebuild mỗi lần)
- `similarity_search()` — tìm top-K chunks liên quan

#### `src/chatbot.py` — RAG Pipeline

- Nhận câu hỏi → tìm 5 chunks liên quan → tạo prompt → gọi Claude API
- System prompt tiếng Việt, hướng dẫn trả lời dựa trên tài liệu
- Hỗ trợ conversation history (giữ 3 cặp Q&A gần nhất)
- Singleton pattern để tránh khởi tạo lại

---

## 🧠 Cách Thức Hoạt Động (RAG Pipeline)

### RAG là gì?

**RAG (Retrieval-Augmented Generation)** là kỹ thuật kết hợp tìm kiếm thông tin với khả năng sinh ngôn ngữ của LLM. Thay vì bắt LLM "nhớ" mọi thứ trong tham số (weights), ta cung cấp thông tin liên quan **ngay trong prompt** để LLM đọc và trả lời.

```
                        RAG = Retrieval + Augmented + Generation  
                         
  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
  │  RETRIEVAL   │     │   AUGMENTED      │     │   GENERATION    │
  │              │     │                  │     │                 │
  │ Tìm đoạn tài│────▶│ Ghép vào prompt  │────▶│ LLM đọc context │
  │ liệu liên   │     │ cùng câu hỏi    │     │ và sinh câu     │
  │ quan nhất    │     │                  │     │ trả lời         │
  └─────────────┘     └──────────────────┘     └─────────────────┘
```

**Ý tưởng cốt lõi:** Claude không được "huấn luyện" trên tài liệu của chúng ta. Thay vào đó, mỗi khi có câu hỏi, hệ thống tìm đúng đoạn tài liệu liên quan và đưa cho Claude đọc cùng lúc với câu hỏi → Claude giống như một chuyên gia đọc tài liệu và trả lời ngay.

### Tại sao dùng RAG thay vì fine-tuning?

| Tiêu chí                     | RAG (dự án này)                                              | Fine-tuning                                                     |
| ------------------------------ | --------------------------------------------------------------- | --------------------------------------------------------------- |
| **Cách hoạt động**   | Tìm tài liệu → đưa vào prompt → LLM đọc và trả lời | Train lại LLM trên dữ liệu riêng → LLM "nhớ" kiến thức |
| **Chi phí**             | Thấp (chỉ trả API mỗi lần hỏi)                            | Rất cao (cần GPU mạnh, train nhiều giờ)                    |
| **Cập nhật dữ liệu** | Nhanh (thêm file .docx → rebuild 30s)                         | Chậm (phải train lại từ đầu)                              |
| **Độ chính xác**     | Cao — luôn trích dẫn từ tài liệu gốc                    | Có thể bịa (hallucinate) vì "nhớ mơ hồ"                  |
| **Yêu cầu dữ liệu**  | Ít (vài trang tài liệu là đủ)                            | Nhiều (hàng nghìn mẫu câu hỏi-đáp)                      |
| **Kiểm soát**          | Cao — biết chính xác nguồn thông tin                      | Thấp — không kiểm soát LLM "nhớ" gì                      |
| **Triển khai**          | Đơn giản (vài file Python)                                  | Phức tạp (infrastructure GPU)                                 |

---

## 🎓 Quá Trình "Huấn Luyện" — Chatbot Học Từ Tài Liệu Như Thế Nào?

> ⚠️ **Lưu ý quan trọng:** Dự án này **KHÔNG huấn luyện (train) bất kỳ AI model nào**. Chúng ta sử dụng 2 model đã được huấn luyện sẵn:
>
> - **all-MiniLM-L6-v2** — đã được train bởi Microsoft trên 1 tỷ cặp câu
> - **Claude** — đã được train bởi Anthropic trên hàng nghìn tỷ từ
>
> Thay vì huấn luyện, chúng ta **chuẩn bị kiến thức** cho chatbot bằng quy trình dưới đây.

### Giai đoạn 1: Trích xuất text từ tài liệu

```
📄 BaoCao_WiFiMarketing.docx
         │
         ▼  python-docx library
┌─────────────────────────────────────────────┐
│ Đọc từng paragraph trong file .docx         │
│                                             │
│ Input:  <Heading 1> "PHẦN II: WIFI..."     │
│ Output: "## PHẦN II: WIFI..."              │
│                                             │
│ Input:  <Normal> "Router LANCS tại..."     │
│ Output: "Router LANCS tại..."              │
│                                             │
│ → Giữ cấu trúc heading (## ###) để        │
│   khi chunk vẫn biết đoạn này thuộc        │
│   phần nào của tài liệu                    │
└─────────────────────────────────────────────┘
         │
         ▼
   Raw text: 9,891 ký tự
```

### Giai đoạn 2: Chia text thành Chunks (Chunking)

**Tại sao phải chia chunks?** Vì embedding model có giới hạn 256 tokens (~400 ký tự). Nếu đưa cả tài liệu 10,000 ký tự vào model, nó sẽ bị cắt mất. Ngoài ra, chunk nhỏ hơn cho kết quả tìm kiếm chính xác hơn.

```
Raw text (9,891 ký tự)
│
├── Cắt tại heading (## ###) trước
├── Nếu đoạn > 800 ký tự → cắt tiếp tại dấu chấm câu
├── Mỗi chunk overlap 200 ký tự với chunk trước
│
▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐
│Chunk 1 │ │Chunk 2 │ │Chunk 3 │ │Chunk 4 │ ... │Chunk 20│
│800 ch  │ │800 ch  │ │800 ch  │ │800 ch  │     │~500 ch │
└───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘     └───┬────┘
    │          │          │          │               │
    └──200ch───┘          └──200ch───┘               │
      overlap               overlap                  │
                                            Tổng: 20 chunks
```

**Chiến lược overlap:** 200 ký tự cuối của chunk trước được lặp lại ở đầu chunk sau. Điều này đảm bảo nếu một câu trả lời nằm đúng ranh giới 2 chunk, hệ thống vẫn tìm được.

**Ví dụ thực tế:**

```
Chunk 5 (kết thúc bằng):
  "...Tường lửa trên Router được cấu hình theo nguyên tắc
   chặn tất cả, cho phép từng người sau khi đăng"

Chunk 6 (bắt đầu bằng — overlap):
  "cho phép từng người sau khi đăng nhập. Cụ thể,
   một chuỗi tường lửa tên CAPTIVE được tạo..."
```

### Giai đoạn 3: Tạo Embeddings (Vector hóa ngữ nghĩa)

Đây là bước cốt lõi nhất. Mỗi chunk text được chuyển thành một **vector số 384 chiều** — đại diện cho **ý nghĩa ngữ nghĩa** của đoạn text.

#### Embedding là gì?

Embedding là phép biến đổi text → vector số, sao cho **các text có ý nghĩa tương tự sẽ có vector gần nhau** trong không gian 384 chiều.

```
Ví dụ đơn giản hóa (2D thay vì 384D):

                    ▲ chiều Y (ý nghĩa kỹ thuật)
                    │
                    │   • "iptables chain CAPTIVE"
                    │       • "tường lửa router"
                    │
                    │           • "cấu hình mạng"
                    │
         "số điện  •│
          thoại     │                • "Flask server"
       khách hàng"  │
                    │
                    │    • "form đăng nhập WiFi"
                    │
                    └──────────────────────────────▶ chiều X (ý nghĩa kinh doanh)
```

Khi user hỏi "tường lửa hoạt động thế nào?", câu hỏi được embed thành vector → nằm gần cụm "iptables / tường lửa / CAPTIVE" → hệ thống trả về đúng chunks liên quan.

#### Quá trình embed một chunk text

```
Input: "Tường lửa trên Router được cấu hình theo nguyên tắc chặn tất cả"
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│ BƯỚC 1: TOKENIZATION                                        │
│ Chia text thành sub-word tokens (WordPiece)                  │
│                                                              │
│ "Tường" → ["Tư", "##ờng"]                                   │
│ "lửa"  → ["lửa"]                                            │
│ "Router" → ["Ro", "##uter"]                                 │
│ → Kết quả: [101, 487, 2938, 1829, 384, ...]  (token IDs)   │
│                                                              │
│ Tổng cộng: ~15-20 tokens cho câu này                        │
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│ BƯỚC 2: TRANSFORMER ENCODING (6 layers)                      │
│                                                              │
│ Mỗi token đi qua 6 lớp Transformer:                         │
│                                                              │
│ Layer 1: Token nhìn các token xung quanh (Self-Attention)    │
│          → "lửa" biết nó liên quan đến "Tường" (= tường lửa)│
│                                                              │
│ Layer 2-5: Hiểu ngữ cảnh sâu hơn                            │
│          → "chặn tất cả" hiểu trong context "cấu hình"      │
│                                                              │
│ Layer 6: Mỗi token thành vector 384 chiều chứa ngữ nghĩa   │
│          phong phú từ cả câu                                 │
│                                                              │
│ Kết quả: Ma trận [20 tokens × 384 dimensions]               │
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│ BƯỚC 3: MEAN POOLING                                         │
│                                                              │
│ Lấy trung bình tất cả token vectors → 1 vector duy nhất     │
│                                                              │
│ Token 1: [0.12, -0.45, 0.78, ..., 0.33]  ─┐                │
│ Token 2: [0.08, -0.39, 0.81, ..., 0.29]   ├─▶ Trung bình   │
│ ...                                        │                 │
│ Token 20: [0.15, -0.42, 0.75, ..., 0.31] ─┘                │
│                                                              │
│ Kết quả: [0.11, -0.41, 0.79, ..., 0.31]  (1 × 384)        │
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│ BƯỚC 4: L2 NORMALIZATION                                     │
│                                                              │
│ Chuẩn hóa vector có độ dài = 1 (đơn vị)                     │
│ → Cho phép dùng Inner Product thay vì Cosine Similarity      │
│ → Nhanh hơn rất nhiều khi tìm kiếm                          │
│                                                              │
│ Kết quả: [0.023, -0.087, 0.167, ..., 0.065]  (||v|| = 1)   │
└──────────────────────────────────────────────────────────────┘
```

#### Model `all-MiniLM-L6-v2` được huấn luyện như thế nào?

Model này đã được Microsoft huấn luyện qua **3 giai đoạn** (chúng ta chỉ sử dụng kết quả, không train lại):

```
GIAI ĐOẠN 1: Pre-training (Microsoft thực hiện)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Dữ liệu: Hàng tỷ câu từ Wikipedia, sách, web
• Phương pháp: Masked Language Model (che từ → đoán từ bị che)
• Kết quả: Model hiểu ngữ pháp và ngữ nghĩa cơ bản
                    │
                    ▼
GIAI ĐOẠN 2: Contrastive Learning (Microsoft thực hiện)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Dữ liệu: 1 tỷ cặp câu tương đồng (câu hỏi-đáp, paraphrase)
• Phương pháp: Dạy model để:
  - Cặp câu GIỐNG nhau → vectors GẦN nhau
  - Cặp câu KHÁC nhau → vectors XA nhau
• Ví dụ:
  ✅ "Captive Portal là gì?" ↔ "Cổng đăng nhập WiFi" → vectors gần
  ❌ "Captive Portal là gì?" ↔ "Thời tiết hôm nay" → vectors xa
• Kết quả: Model biết tạo embeddings phản ánh ngữ nghĩa
                    │
                    ▼
GIAI ĐOẠN 3: Knowledge Distillation (Microsoft thực hiện)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• "Chưng cất" kiến thức từ model lớn (12 layers) → model nhỏ (6 layers)
• Giữ 95% chất lượng nhưng nhanh gấp 5 lần
• Kết quả: Model 90MB, chạy trên CPU bình thường
                    │
                    ▼
GIAI ĐOẠN 4: Sử dụng trong dự án (CHÚNG TA thực hiện)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Tải model đã train xong về máy (~90MB)
• Dùng để embed chunks tài liệu + câu hỏi user
• KHÔNG train thêm — chỉ sử dụng (inference)
```

### Giai đoạn 4: Lưu vào FAISS Vector Store

FAISS (Facebook AI Similarity Search) là thư viện tìm kiếm vector do Meta phát triển.

```
26 chunks × 384 dimensions = Ma trận [26 × 384]

┌─────────────────────────────────────────────────────┐
│                FAISS IndexFlatIP                     │
│                                                     │
│  Vector 0:  [0.023, -0.087, 0.167, ..., 0.065]    │ ← Chunk 1
│  Vector 1:  [0.045, -0.012, 0.234, ..., -0.089]   │ ← Chunk 2
│  Vector 2:  [-0.031, 0.098, 0.145, ..., 0.043]    │ ← Chunk 3
│  ...                                               │
│  Vector 25: [0.067, -0.054, 0.189, ..., 0.012]    │ ← Chunk 26
│                                                     │
│  Khi search: Tính Inner Product (= Cosine vì       │
│  vectors đã normalized) giữa query vector và       │
│  TẤT CẢ 26 vectors → trả về Top-5 cao nhất        │
└─────────────────────────────────────────────────────┘

Lưu ra disk:
  vectorstore/index.faiss    → Ma trận vectors
  vectorstore/metadata.pkl   → Nội dung text + nguồn file
```

**Cách tính Cosine Similarity:**

```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)

Vì vectors đã normalized (||A|| = ||B|| = 1):
similarity = A · B = Inner Product

Ví dụ:
  Query:  "captive portal"        → vector Q = [0.15, -0.08, ...]
  Chunk 5: "cổng đăng nhập WiFi"  → vector C = [0.14, -0.07, ...]
  
  similarity = Q · C = 0.15×0.14 + (-0.08)×(-0.07) + ... = 0.54
  
  Score 0.54 = khá tương đồng (max = 1.0, min = -1.0)
```

---

## 🤖 Claude Đưa Ra Câu Trả Lời Như Thế Nào?

### Bước 1: Tìm context liên quan (Retrieval)

Khi user hỏi "Captive Portal là gì?":

```
1. Embed câu hỏi → vector [0.15, -0.08, 0.23, ..., 0.04]

2. FAISS tìm 5 chunks gần nhất:
   ┌────────────────────────────────────────────────────────┐
   │ Rank │ Score  │ Chunk                                  │
   │──────│────────│────────────────────────────────────────│
   │  1   │ 0.538  │ "WiFi Captive Portal — Hướng dẫn..."  │
   │  2   │ 0.498  │ "...hệ điều hành tự động bật cửa sổ   │
   │      │        │  Captive Portal cho người dùng..."     │
   │  3   │ 0.463  │ "...Flask server — hiển thị form..."   │
   │  4   │ 0.412  │ "...chuỗi tường lửa CAPTIVE..."       │
   │  5   │ 0.389  │ "...bridge br-freewifi và card WiFi..."│
   └────────────────────────────────────────────────────────┘
```

### Bước 2: Xây dựng Prompt (Augmented)

Hệ thống ghép các thành phần thành **một prompt hoàn chỉnh** gửi tới Claude:

```
┌─────────────────────────────────────────────────────────────┐
│                    PROMPT GỬI CHO CLAUDE                     │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ SYSTEM PROMPT (vai trò + quy tắc)                      │ │
│ │                                                         │ │
│ │ "Bạn là trợ lý AI chuyên về WiFi Marketing...          │ │
│ │  Trả lời bằng tiếng Việt, dựa trên tài liệu...        │ │
│ │  Không bịa đặt thông tin..."                            │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ TÀI LIỆU THAM KHẢO (5 chunks liên quan nhất)          │ │
│ │                                                         │ │
│ │ --- Đoạn 1 (nguồn: BaoCao_WiFi...) ---                 │ │
│ │ "WiFi Captive Portal — Hướng dẫn triển khai đầy        │ │
│ │  đủ — Thu thập thông tin người dùng trước khi..."       │ │
│ │                                                         │ │
│ │ --- Đoạn 2 (nguồn: BaoCao_WiFi...) ---                 │ │
│ │ "...hệ điều hành tự động bật cửa sổ Captive Portal     │ │
│ │  cho người dùng. Tường lửa trên Router..."              │ │
│ │                                                         │ │
│ │ --- Đoạn 3, 4, 5 ---                                    │ │
│ │ ...                                                     │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ LỊCH SỬ HỘI THOẠI (3 cặp Q&A gần nhất)               │ │
│ │                                                         │ │
│ │ Người dùng: "Kiến trúc hệ thống như thế nào?"          │ │
│ │ Trợ lý: "Hệ thống gồm Router LANCS + Ubuntu..."       │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ CÂU HỎI: "Captive Portal là gì?"                      │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Bước 3: Claude xử lý và sinh câu trả lời (Generation)

Claude (Large Language Model) xử lý prompt qua các bước sau:

```
BƯỚC 3.1: TOKENIZATION
━━━━━━━━━━━━━━━━━━━━━━
Toàn bộ prompt (~2000-3000 ký tự) được chia thành tokens.
Claude dùng BPE (Byte Pair Encoding) tokenizer riêng.

BƯỚC 3.2: SELF-ATTENTION (cốt lõi của Transformer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Claude có kiến trúc Transformer với hàng chục layers.
Mỗi layer, MỌI token "nhìn" tất cả token khác để hiểu ngữ cảnh.

Ví dụ khi xử lý:
  Token "Captive" trong CÂU HỎI
    → attention cao đến "Captive" trong TÀI LIỆU
    → attention cao đến "Portal" trong cả hai
    → attention cao đến "cổng đăng nhập" trong tài liệu
    → hiểu rằng câu hỏi muốn biết về cổng đăng nhập WiFi

BƯỚC 3.3: CONTEXT WINDOW
━━━━━━━━━━━━━━━━━━━━━━━━
Claude có context window ~200K tokens — đủ để "đọc" 
toàn bộ prompt (system + 5 chunks + history + câu hỏi)
trong MỘT LẦN xử lý.

  ┌────────────────────────────────────────────┐
  │ Context Window (~200K tokens)              │
  │                                            │
  │ [System Prompt]  ← Claude hiểu vai trò    │
  │ [5 Chunks]       ← Claude đọc tài liệu   │
  │ [History]        ← Claude nhớ đã nói gì   │
  │ [Câu hỏi]       ← Claude biết cần trả gì │
  │                                            │
  │ Tất cả được xử lý ĐỒNG THỜI qua          │
  │ Self-Attention — không phải đọc tuần tự   │
  └────────────────────────────────────────────┘

BƯỚC 3.4: AUTO-REGRESSIVE DECODING (sinh từng token)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Claude sinh câu trả lời TỪNG TOKEN MỘT, mỗi token mới 
phụ thuộc vào tất cả tokens trước đó:

  Bước 1: P("Captive" | prompt)     → chọn "Captive"
  Bước 2: P("Portal" | prompt + "Captive")  → chọn "Portal"  
  Bước 3: P("là" | prompt + "Captive Portal")   → chọn "là"
  Bước 4: P("cổng" | prompt + "...là")   → chọn "cổng"
  ...
  Bước N: P("[END]" | prompt + toàn bộ câu trả lời)  → dừng

  Mỗi bước, Claude tính xác suất cho ~100K tokens có thể
  và chọn token có xác suất cao nhất (hoặc sampling).
```

### Bước 4: Trả về qua API

```
Claude API Response:
{
  "content": [{
    "text": "**Captive Portal** (cổng đăng nhập WiFi) là hệ thống 
             yêu cầu khách hàng điền thông tin trước khi được 
             truy cập Internet qua WiFi.\n\nTrong hệ thống của 
             LANCS Retails, Captive Portal hoạt động như sau:
             \n1. Khách kết nối WiFi 'FreeWiFi'
             \n2. Thiết bị tự động hiển thị form đăng nhập...
             \n3. Sau khi điền thông tin, MAC address được thêm 
             vào whitelist..."
  }]
}
         │
         ▼
Flask trả về JSON cho frontend
         │
         ▼
JavaScript render Markdown → hiển thị trong chat bubble
```

### Tại sao Claude trả lời ĐÚNG?

Claude trả lời đúng nhờ 3 yếu tố kết hợp:

| Yếu tố                       | Vai trò                                     | Không có thì sao?                       |
| ------------------------------ | -------------------------------------------- | ------------------------------------------ |
| **System Prompt**        | Quy định vai trò, ngôn ngữ, phong cách | Claude trả lời lạc đề, sai ngôn ngữ |
| **Context (5 chunks)**   | Cung cấp kiến thức cụ thể               | Claude bịa hoặc trả lời chung chung    |
| **Conversation History** | Hiểu ngữ cảnh hội thoại                 | Mỗi câu hỏi tách biệt, mất mạch     |

```
Ví dụ nếu KHÔNG có context:
  Q: "Bridge br-freewifi dùng để làm gì?"
  A: "Tôi không có thông tin cụ thể về br-freewifi..."  ❌

Ví dụ CÓ context (RAG):
  Q: "Bridge br-freewifi dùng để làm gì?"
  Context: "...tạo bridge mạng ảo riêng biệt trên Router, 
            mang tên br-freewifi, gắn với card WiFi phụ ath01.
            Vùng mạng mới sử dụng dải IP khác với mạng chính..."
  A: "Bridge **br-freewifi** được tạo để phân tách mạng WiFi
      Marketing khỏi mạng nội bộ chính. Nó được gắn với card
      WiFi phụ (ath01) và sử dụng dải IP riêng biệt..."  ✅
```

---

## 📊 Model & Embedding — Thông Số Kỹ Thuật

### Embedding Model: `all-MiniLM-L6-v2`

| Thuộc tính                | Giá trị                 | Ý nghĩa                                 |
| --------------------------- | ------------------------- | ----------------------------------------- |
| **Kiến trúc**       | MiniLM (Transformer)      | 6 attention layers, 12 heads              |
| **Kích thước**     | ~90 MB (22.7M params)     | Nhỏ gọn, chạy nhanh trên CPU          |
| **Output Dimensions** | 384                       | Mỗi text → vector 384 số thực         |
| **Max Sequence**      | 256 tokens (~400 ký tự) | Giới hạn input — vì sao cần chunking |
| **Similarity Metric** | Cosine Similarity         | Đo góc giữa 2 vectors (0→1)           |
| **Chạy trên**       | CPU (local, offline)      | Không cần GPU, không cần internet     |
| **Chi phí**          | Miễn phí (Apache 2.0)   | Open-source, dùng thương mại OK       |
| **Ngôn ngữ**        | 100+ ngôn ngữ           | Bao gồm tiếng Việt                     |
| **Tốc độ**         | ~2000 câu/giây (CPU)    | Embed 26 chunks mất < 2 giây            |

### LLM: Claude Sonnet 4 (Anthropic)

| Thuộc tính             | Giá trị                         | Ý nghĩa                                  |
| ------------------------ | --------------------------------- | ------------------------------------------ |
| **Kiến trúc**    | Transformer (decoder-only)        | Auto-regressive generation                 |
| **Context Window** | ~200K tokens                      | Đủ đọc ~50 trang tài liệu cùng lúc |
| **Max Output**     | 2048 tokens (cấu hình)          | Giới hạn độ dài câu trả lời        |
| **API**            | REST (Messages API)               | Gọi qua HTTP POST, trả JSON              |
| **Endpoint**       | `api.anthropic.com/v1/messages` | Cần API key trong header                  |
| **Ngôn ngữ**     | Đa ngôn ngữ                    | Tiếng Việt tốt                          |
| **Latency**        | ~2-5 giây                        | Thời gian sinh câu trả lời             |

### Vector Store: FAISS

| Thuộc tính            | Giá trị                   | Ý nghĩa                        |
| ----------------------- | --------------------------- | -------------------------------- |
| **Provider**      | Meta (Facebook AI)          | Open-source, production-ready    |
| **Index Type**    | IndexFlatIP                 | Exact search, Inner Product      |
| **Similarity**    | Cosine (vectors normalized) | IP = Cosine khi                  |
| **Số vectors**   | 26                          | Từ 2 file .docx                 |
| **Search Speed**  | < 1ms                       | Brute-force OK cho < 10K vectors |
| **Lưu trữ**     | Local disk                  | `vectorstore/index.faiss`      |
| **Kích thước** | ~40 KB                      | 26 × 384 × 4 bytes (float32)   |

---

## 🚀 Cài Đặt & Chạy

### Yêu cầu hệ thống

- Python 3.8+
- Ubuntu 20.04+ (hoặc bất kỳ Linux distro)
- RAM: >= 2GB (cho embedding model)
- Internet (để gọi Claude API)

### Các bước cài đặt

```bash
# 1. Clone / di chuyển vào thư mục dự án
cd ~/lancsretails/chat_bot_ai

# 2. Cài đặt dependencies
pip install -r requirements.txt

# 3. Cấu hình API key
cp .env.example .env
nano .env
# Điền CLAUDE_API_KEY=sk-ant-api03-...

# 4. Đặt tài liệu vào docs/
# (File .docx đã có sẵn trong docs/)

# 5. Chạy ứng dụng
python3 app.py

# 6. Mở trình duyệt
# http://localhost:5000
```

### Lần đầu chạy

Lần đầu tiên chạy, hệ thống sẽ:

1. Tải embedding model `all-MiniLM-L6-v2` (~90MB) từ Hugging Face
2. Đọc tất cả file `.docx` trong `docs/`
3. Tạo embeddings và lưu FAISS index vào `vectorstore/`
4. Khởi động Flask server trên port 5000

Các lần sau chỉ load FAISS index từ disk (nhanh hơn nhiều).

### Lưu ý Python 3.8

Nếu gặp lỗi `protobuf` / `TensorFlow`, app đã tự động fix bằng:

```python
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
```

---

## ⚙️ Cấu Hình

Tất cả cấu hình nằm trong file `.env`:

```env
# API Key (bắt buộc)
CLAUDE_API_KEY=sk-ant-api03-...

# Chunking
CHUNK_SIZE=800          # Kích thước mỗi chunk (ký tự)
CHUNK_OVERLAP=200       # Overlap giữa các chunks

# Models
LLM_MODEL=claude-sonnet-4-20250514   # Model Claude
EMBEDDING_MODEL=all-MiniLM-L6-v2     # Model embedding (local)
```

### Tinh chỉnh chất lượng trả lời

| Tham số           | Tăng                                   | Giảm                            |
| ------------------ | --------------------------------------- | -------------------------------- |
| `CHUNK_SIZE`     | Nhiều ngữ cảnh hơn, ít chính xác | Ít ngữ cảnh, chính xác hơn |
| `CHUNK_OVERLAP`  | Ít mất thông tin ranh giới          | Nhanh hơn, ít redundancy       |
| Top-K (trong code) | Nhiều context, chậm hơn              | Ít context, nhanh hơn          |

---

## 📚 Mở Rộng Knowledge Base

### Thêm tài liệu mới

```bash
# 1. Đặt file .docx vào thư mục docs/
cp tai_lieu_moi.docx docs/

# 2. Xóa vector store cũ để rebuild
rm -rf vectorstore/

# 3. Restart app (tự động rebuild)
python3 app.py
```

### Tài liệu hiện có

| File                                     | Nội dung                                            | Kích thước |
| ---------------------------------------- | ---------------------------------------------------- | ------------- |
| `BaoCao_WiFiMarketing_TranDucNam.docx` | Báo cáo kỹ thuật WiFi Marketing & Captive Portal | ~30 KB        |

---

## 📡 API Reference

### `GET /` — Trang chủ

Trả về giao diện chat HTML.

### `POST /api/chat` — Gửi câu hỏi

**Request:**

```json
{
  "message": "Captive Portal là gì?"
}
```

**Response (thành công):**

```json
{
  "success": true,
  "answer": "Captive Portal là cổng đăng nhập WiFi..."
}
```

**Response (lỗi):**

```json
{
  "success": false,
  "error": "Mô tả lỗi"
}
```

### `POST /api/clear` — Xóa lịch sử hội thoại

**Response:**

```json
{
  "success": true,
  "message": "Đã xóa lịch sử hội thoại"
}
```

---

## 👤 Tác Giả

**Trần Đức Nam** — Developer & AI Engineer
LANCS Retails
Ngày tạo: 24/03/2026

# 🤖 Qwen2.5-0.5B RAG Story Generator

A full-stack AI chatbot built from scratch using **Qwen2.5-0.5B** with **Retrieval Augmented Generation (RAG)**.  
Upload any PDF or text document and ask questions grounded in your own content — no hallucination, no API costs, runs 100% locally on CPU.

---

## 🏗️ Architecture

```
Browser (localhost:8501)  ←→  Streamlit UI (app.py)
                               ↕
                          FastAPI Server (api.py) :8000
                          ↙              ↘
                    model.py           rag.py
                  Qwen2.5-0.5B      ChromaDB +
                   Inference      Sentence Transformers
                          ↘              ↙
                         MongoDB   chroma_db/
                      (chat history)  (vectors)
```

---

## ✨ Features

### 💬 Chat
- ChatGPT-style interface with sidebar session history
- **Streaming responses** — words appear one by one in real time
- Smart chunking — handles inputs of ANY length automatically
- Continue story button for creative writing
- Thumbs up / down rating on every AI response
- AI-generated session titles in sidebar

### 📚 RAG (Retrieval Augmented Generation)
- Upload **PDF, TXT, or MD** files
- Documents split into overlapping chunks (1500 chars, 200 overlap)
- Embedded using **Sentence Transformers** (all-MiniLM-L6-v2)
- Stored in **ChromaDB** — persists between restarts
- Similarity search with cosine distance scoring
- **Strict anti-hallucination prompt** — model only answers from document
- **Auto evaluation** after every RAG answer (4 metrics + grade A/B/C/D)
- Full chunk content printed to terminal on every retrieval

### ⚡ Performance
- **int8 quantization** — 2-3x faster, half the RAM
- **torch.compile()** — 20-30% faster after warmup
- **Response caching** — identical questions answered instantly
- History trimmed to last 6 turns — keeps generation fast

### 📊 Dashboard
- Total messages, sessions, tokens, avg response time
- Messages per day chart
- Top words used
- Cache hit stats and ratings summary

### 🔢 Token Tracking
- Live input token counter as you type
- Output token count shown per response
- Total token budget tracker (input + output vs 32,768 limit)
- Visual progress bar with colour zones

---

## 🗂️ Project Structure

```
Qwen/
│
├── app.py              # Streamlit frontend — all UI
├── api.py              # FastAPI backend — all endpoints
├── model.py            # Qwen2.5-0.5B loading + inference
├── rag.py              # RAG engine — chunking, embedding, retrieval
├── requirements.txt    # All Python dependencies
├── README.md           # This file
│
└── chroma_db/          # Auto-created — ChromaDB vector store
    └── ...             # Persists your indexed documents
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit ≥1.32.0 | Chat UI, sidebar, pages |
| Backend | FastAPI + Uvicorn | REST API + SSE streaming |
| AI Model | Qwen2.5-0.5B | Text generation (local, CPU) |
| Embeddings | Sentence Transformers | Text → vectors for RAG |
| Vector DB | ChromaDB | Store + search embeddings |
| Database | MongoDB | Chat history, ratings, cache |
| PDF Reading | pypdf | Extract text from PDFs |
| File Upload | python-multipart | Handle multipart form uploads |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- MongoDB running locally on port 27017
- 4GB RAM minimum (8GB recommended)

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourusername/qwen-rag-story-generator.git
cd qwen-rag-story-generator
```

### Step 2 — Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Install optional speed packages
```bash
# int8 quantization (2-3x faster on CPU)
pip install bitsandbytes

# File upload support (required for RAG)
pip install python-multipart
```

### Step 5 — Start MongoDB
Make sure MongoDB is running on `localhost:27017`

---

## 🚀 Running the App

Open **two terminals**:

**Terminal 1 — FastAPI backend:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Streamlit frontend:**
```bash
streamlit run app.py
```

Open browser: `http://localhost:8501`  
API docs: `http://localhost:8000/docs`

---

## 📖 How to Use

### Normal Chat
1. Open `http://localhost:8501`
2. Type your message and press **Enter**
3. Watch the response stream word by word
4. Use ⚙️ Settings to adjust temperature, tokens etc.

### RAG Mode
1. Click **📚 Documents** in the sidebar
2. Upload a PDF or TXT file
3. Click **"Index This Document"**
4. Go to **💬 Chat**
5. Open **⚙️ Settings** → toggle **🔬 RAG Mode ON**
6. Ask questions about your document

**Best settings for RAG:**
```
Temperature : 0.1 – 0.2   (factual, less hallucination)
Max tokens  : 200 – 300   (complete answers)
Top-k chunks: 5            (more context)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/generate` | Full response generation |
| POST | `/stream` | SSE streaming response |
| POST | `/count-tokens` | Count tokens in any text |
| POST | `/rate` | Save thumbs up/down rating |
| GET | `/stats` | Dashboard statistics |
| GET | `/sessions` | List all sessions |
| GET | `/history/{session_id}` | One session messages |
| DELETE | `/history/{session_id}` | Delete a session |
| POST | `/rag/upload` | Upload + index document |
| POST | `/rag/ask/stream` | RAG question streaming |
| GET | `/rag/documents` | List indexed documents |
| DELETE | `/rag/document/{doc_id}` | Delete a document |
| GET | `/rag/stats` | RAG system statistics |
| POST | `/rag/evaluate` | Evaluate a RAG answer |

---

## 🧠 How RAG Works

```
PHASE 1 — INDEXING (done once per document)
──────────────────────────────────────────────
PDF uploaded → text extracted → split into chunks
→ each chunk embedded (384 numbers) → stored in ChromaDB

PHASE 2 — RETRIEVAL (every question)
──────────────────────────────────────────────
User question → embedded → compared vs all chunks
→ top 3-5 similar chunks retrieved
→ strict prompt: "ONLY use CONTEXT, DO NOT make things up"
→ Qwen reads context → grounded answer
→ 4-metric evaluation runs automatically
```

---

## 📊 RAG Evaluation Metrics

| Metric | What It Measures | Weight |
|---|---|---|
| Context Relevance | Were the right chunks retrieved? | 30% |
| Answer Groundedness | Is the answer from the document? | 35% |
| Answer Completeness | Did the model use all context? | 15% |
| Query Coverage | Did the answer address the question? | 20% |

Grades: **A** (90-100%) · **B** (70-89%) · **C** (50-69%) · **D** (below 50%)

---

## 🔧 Key Configuration

**app.py**
```python
CHUNK_SIZE    = 800     # chars per chunk for long inputs
API_URL       = "http://localhost:8000"
```

**rag.py**
```python
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 1500
CHUNK_OVERLAP = 200
TOP_K         = 3
```

**api.py**
```python
MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "qwen_story_app"
```

---

## 📈 Token Limits

| Setting | Min | Default | Max |
|---|---|---|---|
| Output tokens | 50 | 100 | 500 |
| Model context window | — | — | 32,768 |
| Document chunk size | — | 1,500 chars | — |
| History turns sent | — | Last 6 | — |

---

## 🐛 Troubleshooting

**`Attribute "app" not found in module "api"`**
```bash
python -c "import api; print(dir(api))"
# If shows rag.py contents → api.py saved incorrectly, re-download
```

**`Form data requires python-multipart`**
```bash
pip install python-multipart
```

**`RAG not available`**
```bash
pip install chromadb sentence-transformers pypdf python-multipart
```

**MongoDB not connecting**
```
Make sure MongoDB service is running on localhost:27017
```

---



## 📚 What I Learned Building This

- How LLM tokenization works at the code level
- How RAG pipelines work — chunking, embedding, retrieval
- How cosine similarity measures semantic meaning
- How SSE streaming enables word-by-word responses
- How to evaluate RAG answer quality with custom metrics
- How to prevent hallucination with strict prompting
- How int8 quantization speeds up CPU inference
- How to build a complete full-stack AI app from scratch

---

## 📄 License

MIT License — see LICENSE file for details.  
**Qwen2.5-0.5B** model licensed under Apache 2.0 by Alibaba Cloud.

---

## 🙏 Acknowledgements

- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B) by Alibaba Cloud
- [ChromaDB](https://www.trychroma.com/) — local vector database
- [Sentence Transformers](https://www.sbert.net/) — semantic embeddings
- [FastAPI](https://fastapi.tiangolo.com/) — Python web framework
- [Streamlit](https://streamlit.io/) — Python UI framework

---


Built from scratch — every line of chunking, embedding, retrieval and evaluation written manually to understand RAG deeply before using any framework.


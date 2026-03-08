# ⚖️ Samvidhan AI — Indian Constitution Assistant

A RAG (Retrieval-Augmented Generation) based chatbot that answers questions about the Indian Constitution using actual constitutional text.

---

## 🚀 Tech Stack

- **LangChain** — RAG pipeline
- **Qdrant** — Vector database for storing chunks
- **HuggingFace Embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
- **Groq (LLaMA 3.1)** — LLM for generating answers
- **FastAPI** — Backend API server
- **HTML/CSS/JS** — Static frontend

---

## 📁 Project Structure

```
├── index.html          # Frontend UI
├── server.py           # FastAPI backend
├── build_vector_db.py  # Script to build Qdrant vector DB
├── llm.py              # RAG chain (testing)
├── requirements.txt    # Python dependencies
└── .env                # API keys (not pushed to Git)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/Tarutiwari/bajaj_solve.git
cd bajaj_solve
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup `.env` file
Create a `.env` file in the root directory:
```
API_KEY=your_groq_api_key_here
```
Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Start Qdrant (Docker)
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### 5. Build Vector Database
Add your Constitution PDF files in a `constitution/` folder, then run:
```bash
python build_vector_db.py
```

### 6. Start Backend Server
```bash
uvicorn server:app --reload
```

### 7. Open Frontend
Open `index.html` in your browser — done! 🎉

---

## 💡 How It Works

1. Constitution PDFs are split into chunks and stored in Qdrant
2. User asks a question via the frontend
3. Relevant chunks are retrieved using semantic search
4. Groq's LLaMA 3.1 generates an answer using only those chunks
5. Answer is displayed in the chat UI

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `API_KEY` | Groq API Key |

---

## 📌 Notes

- Constitution PDFs are not included in this repo due to size
- Make sure Qdrant is running before starting the server

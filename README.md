# 📄 PDF Talker

A local **RAG (Retrieval-Augmented Generation)** application that lets you upload any PDF and chat with it — powered by **Cohere LLM**, **FAISS** vector search, and **HuggingFace embeddings**, with full pipeline transparency.

![PDF Talker Demo](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-🦜-1C3C3C)
![Cohere](https://img.shields.io/badge/Cohere-LLM-orange)

---

## ✨ Features

- 📤 Upload any PDF and ask questions in natural language
- 🔍 FAISS vector similarity search with distance scores
- 🤖 Cohere `command-r-plus` LLM for grounded answers
- 📊 Real-time pipeline metrics (retrieval latency, generation latency, token usage)
- 🔬 Under-the-hood trace: see query vectors and retrieved chunks
- 📡 LangSmith tracing integration
- 📏 RAGAS evaluation script (standalone)

---

## 🗂️ Project Structure

```
PDFTalker/
├── app.py              # Main Streamlit UI
├── rag_pipeline.py     # RAG logic (PDF → FAISS → retrieval → generation)
├── evaluation.py       # Standalone RAGAS evaluation script
├── requirements.txt    # Python dependencies
├── .env                # 🔒 Your API keys (NOT committed to GitHub)
├── .env.example        # ✅ Template — safe to commit
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started (Local)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/PDFTalker.git
cd PDFTalker
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
```bash
# Copy the template
copy .env.example .env    # Windows
cp .env.example .env      # macOS/Linux

# Open .env and fill in your keys:
# COHERE_API_KEY=...
# LANGCHAIN_API_KEY=...
```

### 5. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Deploy for Free (Streamlit Community Cloud)

1. Push this repo to GitHub (`.env` is gitignored — your keys are safe)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** → select your repo → set `app.py` as the entry point
4. Add your secrets under **Settings → Secrets**:
   ```toml
   COHERE_API_KEY = "your_cohere_api_key"
   LANGCHAIN_API_KEY = "your_langsmith_api_key"
   LANGCHAIN_TRACING_V2 = "true"
   LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
   LANGCHAIN_PROJECT = "RAG_demo"
   ```
5. Click **Deploy** — done! 🎉

---

## 📊 Run RAGAS Evaluation

Edit the `data` dict in `evaluation.py` with real Q/A pairs from your PDF, then run:

```bash
python evaluation.py
```

This evaluates your RAG pipeline on 4 metrics:
| Metric | What it measures |
|---|---|
| `faithfulness` | Is the answer grounded in the retrieved context? |
| `answer_relevancy` | Is the answer relevant to the question? |
| `context_precision` | Are the retrieved chunks precise? |
| `context_recall` | Did retrieval capture all needed information? |

---

## 🔑 API Keys Needed

| Key | Where to get it |
|---|---|
| `COHERE_API_KEY` | [dashboard.cohere.com](https://dashboard.cohere.com) |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) (optional, for tracing) |

---

## 🛡️ Security Notes

- **`.env` is in `.gitignore`** — your keys will never be committed
- Use `.env.example` as a reference for collaborators
- When deploying to Streamlit Cloud, add keys via their **Secrets UI** — not in code

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io) — UI framework
- [LangChain](https://langchain.com) — RAG orchestration
- [Cohere](https://cohere.com) — LLM (`command-r-plus-08-2024`)
- [FAISS](https://github.com/facebookresearch/faiss) — Vector store
- [HuggingFace](https://huggingface.co) — Embeddings (`all-MiniLM-L6-v2`)
- [RAGAS](https://docs.ragas.io) — RAG evaluation framework
- [LangSmith](https://smith.langchain.com) — Tracing & observability

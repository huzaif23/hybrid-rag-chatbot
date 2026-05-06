# Tax Assistant AI Tool - MVP

A minimal, accurate tax assistant that answers tax-related questions using IRS data with RAG (Retrieval-Augmented Generation).

## 📁 Project Structure

```
.
├── backend/
│   ├── main.py          # FastAPI server with /chat endpoint
│   ├── rag.py           # RAG pipeline (embedding, vector search, retrieval)
│   ├── data.py          # IRS tax dataset (hardcoded)
│   └── models.py        # Pydantic models
├── frontend/
│   ├── src/
│   │   ├── App.jsx      # React UI (input + response)
│   │   └── main.jsx
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
└── README.md
```

## ⚡ Features

- **Minimal UI**: Text input + Ask button + response area
- **RAG Pipeline**: Keyword extraction → embedding → vector search → top 3 chunks
- **Guardrails**: Rejects non-tax questions
- **Source Links**: Returns IRS source URLs
- **Fast**: In-memory FAISS-like index, preloaded vectors
- **Accurate**: Only answers from IRS data

## 🚀 Setup & Run

### 1. Backend Setup

```bash
cd backend
pip install fastapi uvicorn numpy pydantic
```

**Run the server:**

```bash
python main.py
```

Server runs at: `http://127.0.0.1:8000`

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: `http://127.0.0.1:3000`

The frontend proxies `/chat` requests to the backend.

## 🧪 Demo Questions

**These work:**

- "How do I file Form 1040?"
- "What deductions can I claim?"
- "What is the standard deduction?"
- "Tell me about EITC"
- "How do I file an amended return?"

**These are rejected (not tax-related):**

- "Best stocks to invest in?"
- "What's the weather today?"
- "How do I bake a cake?"

## 📋 API Endpoint

**POST /chat**

Request:

```json
{
  "message": "How do I file Form 1040?"
}
```

Response:

```json
{
  "answer": "Here's how to file Form 1040:\n\n1. Gather W-2 and 1099 forms...",
  "sources": ["https://www.irs.gov/instructions/i1040"],
  "confidence": 85.0,
  "is_tax_topic": true
}
```

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python), NumPy, Pydantic
- **Frontend**: React + Vite
- **RAG**: Custom FAISS-like vector index (in-memory)
- **Embeddings**: Mock embeddings (512-dim) — optional upgrade to hosted embeddings
- **LLM**: Chat Completions over RAG context — **xAI Grok** (`XAI_API_KEY` from [console.x.ai](https://console.x.ai)) or **Groq** (`GROK_MODEL` / `GROQ_API_KEY`; keys starting with `gsk_` are sent to [Groq’s API](https://console.groq.com), not xAI)

## 🔧 Customization

### Replace Mock Embeddings

In `backend/rag.py`, replace `MockEmbeddingModel` with actual OpenAI embeddings:

```python
from openai import OpenAI

class OpenAIEmbeddingModel:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def encode(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
```

### Grok (xAI) configuration

In `.env` at the repository root, set `XAI_API_KEY`. Optional: `GROK_MODEL` (default `grok-3-mini`), `XAI_BASE_URL` (default `https://api.x.ai/v1`). Generation is implemented in `@backend/grok_tax.py` using the official OpenAI Python client pointed at xAI’s compatible endpoint.

## 📝 Dataset

The dataset in `backend/data.py` contains:

- Form 1040 filing steps
- Deductions overview (standard & itemized)
- Tax brackets 2024
- Credits (EITC, Child Tax Credit, Roth IRA)
- Filing requirements
- Business expenses
- Home office deduction
- Common tax forms reference
- And more...

Each record includes `content`, `source` (IRS URL), `form`, and `topic`.

## 🎯 Performance

- Preloaded FAISS index in memory
- Top 3 chunks only
- Fast similarity search
- <2s response time target

## ⚠️ Important Notes

- **MVP Simplifications**: No database, no auth, no file uploads
- **Production Ready**: Add proper logging, error handling, rate limiting
- **Security**: Add API key protection, input sanitization
- **Scaling**: Use actual vector DB (FAISS → Chroma/Pinecone)

## 📞 Support

For questions or issues, refer to IRS publications at [IRS.gov](https://www.irs.gov)

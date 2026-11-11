

Uploading MediAsk Demo.mp4…

# MediAsk — AI Assistant for Drug Information

MediAsk is a lightweight, local **RAG (Retrieval-Augmented Generation)** app that lets users ask natural language questions about common medications and receive helpful answers based on trusted medical documents.

🚀 Built  to demonstrate applied NLP, LangChain, and LLM deployment skills, now containerized using Docker for reproducibility and deployment. A Streamlit UI was added for a clean, accessible user experience with a professional medical theme.

---

## 📚 What It Does

”

MediAsk uses a configurable **local or API-backed LLM** + a **vector database (FAISS)** to:
- Load and chunk medical drug files in Markdown, PDF, and DOCX format
- Search for the most relevant information
- Answer user queries using context-grounded responses

---

## 🛠️ Tech Stack

- **LangChain** — Framework for chaining LLMs + retrieval
- **Ollama / OpenAI / Anthropic / Gemini** — Local or API-backed LLM providers
- **FAISS** — Fast vector similarity search
- **HuggingFace Sentence Transformers** — For embedding text
- **FastAPI** — Lightweight API for interaction
- **Streamlit** — UI for asking questions and receiving answers
- **Docker** — Portable app container

### LLM provider configuration

Ollama is used by default:

```text
LLM_PROVIDER=ollama
LLM_MODEL=llama3
```

API-backed providers are also supported:

```text
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-key
```

Or:

```text
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-latest
ANTHROPIC_API_KEY=your-key
```

Or Google Gemini:

```text
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.8-flash
GEMINI_API_KEY=your-key
```

Set `OPENAI_BASE_URL` when using an OpenAI-compatible gateway. Never commit
API keys to source control. See [documentation.md](documentation.md) for setup
details.

The Streamlit sidebar also allows users to select a provider, model, and API
key at runtime. These values are sent with each question request and are not
written to `.env` or persisted by the application.

---

## 🗂️ Project Structure

```
MediAsk/
├── app/
│   ├── main.py                # FastAPI app
│   └── ui.py                  # Streamlit UI
├── data/
│   └── who_drug_data.pdf      # Source PDF
├── notebooks/                 # (Optional) Jupyter notebooks
├── rag/
│   ├── ingest.py              # Markdown, PDF, and DOCX ingestion & indexing
│   ├── rag_chain.py           # RAG chain setup
│   └── faiss_index/
│       ├── index.faiss        # FAISS vector store
│       └── index.pkl
├── tests/                     # Future unit tests
├── .github/workflows/ci.yml  # GitHub Actions workflow
├── .streamlit/config.toml     # Custom dark mode theme
├── Dockerfile                 # For containerizing the app
├── .gitignore
├── README.md
├── documentation.md         # Setup, architecture, API, and testing details
└── requirements.txt
```

---

## ⚙️ How to Run

See [documentation.md](documentation.md) for the architecture, prerequisites,
API behavior, testing commands, and safety limitations.

### 🚀 Run the FastAPI backend with Docker

The current Docker image runs the FastAPI backend only. Build it from the
repository root:

```powershell
docker build -t mediask .
```

Start the API container:

```powershell
docker run --name mediask-api -p 8000:8000 mediask
```

Verify that the API is available:

```powershell
curl.exe http://localhost:8000/
```

The API is now available at `http://localhost:8000`. In another terminal,
run the Streamlit interface locally:

```powershell
streamlit run app\ui.py
```

If needed, set the UI API URL before starting Streamlit:

```powershell
$env:MEDIGUIDE_API_URL = "http://localhost:8000"
streamlit run app\ui.py
```

For local development, mount the source data and FAISS index so uploads and
index changes persist after the container stops:

```powershell
docker run --name mediask-api `
  -p 8000:8000 `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\rag\faiss_index:/app/rag/faiss_index" `
  mediask
```

Useful container commands:

```powershell
docker logs -f mediask-api
docker stop mediask-api
docker start mediask-api
docker rm mediask-api
```

If the container already exists, remove it before running a new container with
the same name. The API upload endpoint writes supported files to `/app/data`
and rebuilds the index in `/app/rag/faiss_index`; the volume mounts map those
directories to the local project.

### 🐍 Run Locally
```bash
pip install -r requirements.txt
python rag/ingest.py
uvicorn app.main:app --reload
```

---

### 🧪 Use the Streamlit UI

Make sure your FastAPI backend is running on port `8000`—either locally or in
the Docker container—then launch the UI:

```bash
streamlit run app/ui.py
```

You’ll get a clean healthcare-themed chatbot interface with:
- A query input
- An uploader for adding Markdown, PDF, and DOCX medical reference files
- Persistent chat history with retrieved source citations
- API health checks and clear upload/query errors
- Upload progress and indexed-file status
- Example questions and a clear-conversation control
- A footer disclaimer

Set `MEDIGUIDE_API_URL` if the FastAPI backend is not running at
`http://localhost:8000`.

---

### 🔎 Query the API directly

```bash
curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the side effects of ibuprofen?"}'
```

### 📊 Evaluate the RAG pipeline

Evaluation cases live in `evaluation\questions.jsonl`, outside `data\`, so
they are not indexed as medical documents. The evaluator accepts either a JSON
array or one JSON object per line. Start the API, ensure the relevant
medical files are indexed, and run:

```powershell
python evaluation\run_evaluation.py
```

The script reports completed cases, failures, answer token overlap against the
reference answers, expected-source matches, average latency, and p95 latency.
The source metric is limited because `/ask` currently returns source text
rather than source filenames; use it only when the expected filename appears
in the returned source text.

---

## 📬 Example Query

**Input:**
```json
{ "query": "Can I take ibuprofen if I have asthma?" }
```

**Output:**
```json
{
  "answer": "Ibuprofen should be used with caution in people with asthma..."
}
```

---

## 🔄 CI/CD Pipeline

A simple GitHub Actions workflow builds the Docker image and runs a smoke test to ensure the app starts cleanly.

---



## 📄 License

MIT License — free to use, adapt, or improve.

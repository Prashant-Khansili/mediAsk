---
title: MediAsk
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

https://github.com/user-attachments/assets/c568f253-67b6-482c-bc1d-8e2e17d25df7

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

The evaluation runs offline from the repository root. It builds its own
in-memory index from `data\` and calls the pipeline code directly, so the API
server doesn't need to be running.

#### The question set

`evaluation\eval_set.jsonl` holds 50 questions written from the indexed
Acetaminophen Injection label:

- **40 answerable.** Each stores a reference answer, plus an exact quote from
  the label (the *evidence span*) with its page and section.
- **10 unanswerable.** These ask about other drugs, prices, or near-misses
  such as dosing for premature newborns, which the label doesn't cover. The
  correct behaviour is to say the answer isn't in the documents.

```json
{"id": "q02", "question": "What is the largest single dose an adult over 50 kg can receive?",
 "answerable": true, "reference_answer": "The maximum single dose is 1,000 mg.",
 "evidence": "maximum single dose of Acetaminophen Injection of 1,000 mg", "page": 3,
 "section": "2.2 Recommended Dosage: Adults and Adolescents", "kind": "dosage"}
```

#### What is measured

There are three scripts in `evaluation\`.

**1. Retrieval comparison** (`retrieval_experiments.py`, no LLM needed). This
changes one search setting at a time (duplicate skipping, chunking, keyword
vs vector vs hybrid search, embedding model, re-ranker). For each answerable
question it checks where the evidence span appears in the search results:

- **hit@5**: the share of questions whose evidence is in the top 5 chunks.
- **MRR@5**: how high it ranks (1st = 1.0, 2nd = 0.5, 3rd ≈ 0.33, missing = 0).

A chunk counts as correct if it contains the evidence span, so the numbers
stay comparable when the chunking changes.

**2. End-to-end** (`evaluate.py`). This runs every question through the
pipeline and the LLM, then makes one judge call per answer:

- **Correctness.** The judge compares the answer with the reference answer
  and scores it 0–3: 3 = correct and complete, 2 = main fact right but a
  detail missing, 1 = partly correct, 0 = wrong or wrongly says "not found".
  A score of 2 or 3 counts as correct.
- **Faithfulness.** The judge checks each sentence of the answer against the
  chunk it cites. A sentence with no citation, or with a claim the chunk
  doesn't state, is unsupported. This catches answers built from the model's
  own knowledge instead of the label.
- **Refusals.** Unanswerable questions should be refused. Answerable ones
  should not be.
- **Latency.** The time per question.

The script also writes `spot_check_<label>.md`, with 10 random answers to
score by hand. Comparing them with the judge shows whether the judge can be
trusted.

**3. Prompt-injection test** (`injection_test.py`). This adds poisoned
documents containing instructions like "ignore previous instructions…" to the
index, then asks questions that retrieve them. An attack succeeds if its
payload shows up in the answer. Each attack is run against both the guarded
prompt and the original prompt.

#### Results

These use `gemini-3.5-flash-lite` for both answering and judging. The
original app is run with `--naive --chunking fixed --no-dedupe --mode vector`.

| Metric | Original app | Current pipeline |
|---|---:|---:|
| Search finds the answer in the top 5 (hit@5) | 45% | **97.5%** |
| Answers judged correct (score ≥ 2 of 3) | 55% | **97.5%** |
| Says "not found" when the answer is in the label | 35% | **2.5%** |
| Refuses questions the label can't answer | 10 / 10 | 10 / 10 |
| Average latency per question | 9.8 s | 5.9 s |

#### Run it

Answering and judging go through an LLM API. The free Gemini tier is enough:
about 2 calls per question. Set `GEMINI_API_KEY` first. Only the small
embedding model runs locally, so a CPU-only laptop is fine.

```powershell
# 1. Retrieval comparison
python -m evaluation.retrieval_experiments

# 2. End-to-end, current pipeline vs the original app
python -m evaluation.evaluate --label final --provider gemini --model gemini-3.5-flash-lite `
  --mode hybrid --embedding e5-small --min-vector-score 0.80
python -m evaluation.evaluate --label baseline --provider gemini --model gemini-3.5-flash-lite `
  --naive --chunking fixed --no-dedupe --mode vector

# 3. Prompt-injection test
python -m evaluation.injection_test --provider gemini --model gemini-3.5-flash-lite
```

Each question is saved as soon as it finishes. If a run stops partway (crash,
Ctrl-C or API quota), re-run the same command and it continues from the next
question (`--fresh` starts over). Results are written to `evaluation\results\`.
The summary is in [RESULTS.md](evaluation/results/RESULTS.md), and all options
are in [documentation.md](documentation.md#offline-evaluation-retrieval-llm-as-judge-guardrails).

**Caveats.**
- The same model writes and judges the answers, which can inflate scores. The
  spot-check sheet is there to test this.
- 40 answerable questions from one label is a small sample: each question is
  worth 2.5 points.
- The questions borrow the label's wording, which favours keyword search. The
  one miss shows the reverse: `q16`, "Who should not receive…?", shares no
  words with the label's "contraindicated", so keyword search ranks the right
  chunk too low.

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

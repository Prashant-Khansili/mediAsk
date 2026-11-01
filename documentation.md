# MediAsk Documentation

## Architecture

MediAsk is a local retrieval-augmented generation application:

1. `rag/ingest.py` loads Markdown, PDF, and DOCX files from `data/`.
2. Documents are split into overlapping 500-character chunks.
3. HuggingFace `all-MiniLM-L6-v2` creates embeddings.
4. FAISS stores the embeddings and source document content in `rag/faiss_index/`.
5. `rag/rag_chain.py` loads the index and retrieves the three most relevant chunks.
6. A configurable local or API-backed chat model generates the answer.
7. FastAPI exposes the `/ask` endpoint; Streamlit provides the web UI.
8. Uploaded medical files are stored in `data/`, then the FAISS index is rebuilt
   from all available Markdown, PDF, and DOCX files.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama separately, then download the configured model:

```bash
ollama pull llama3
```

Ollama is the default provider. To use an API-backed model, set the provider
and credentials before starting FastAPI:

```bash
# Local model (default)
LLM_PROVIDER=ollama
LLM_MODEL=llama3

# OpenAI or an OpenAI-compatible endpoint
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-key
# Optional for compatible gateways:
# OPENAI_BASE_URL=https://your-gateway.example/v1

# Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-latest
ANTHROPIC_API_KEY=your-key

# Google Gemini
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.8-flash
GEMINI_API_KEY=your-key
```

Never commit API keys to source control. The provider client is loaded only
when selected, so local Ollama usage does not require cloud credentials.

Place source Markdown, PDF, or DOCX files in `data/` and build the index:

```bash
python rag/ingest.py
```

Start the API and UI in separate terminals:

```bash
uvicorn app.main:app --reload
streamlit run app/ui.py
```

The Streamlit interface provides persistent chat history, source excerpts,
upload progress, API health checks, example questions, and a session list of
indexed files. Set `MEDIGUIDE_API_URL` to point the UI at a different API
host, for example `MEDIGUIDE_API_URL=http://127.0.0.1:8000`.

The sidebar includes runtime AI provider, model, and API-key fields. The UI
sends these values in request headers to `/ask`; the application does not
write them to `.env` or persist them to disk. Use HTTPS when the UI and API
communicate across an untrusted network.

## Run the API with Docker

The Dockerfile packages the FastAPI backend and exposes port `8000`. It does
not package or start Streamlit, so run the UI locally in a separate terminal.
From the repository root, build the image:

```powershell
docker build -t mediask .
```

Start the API container:

```powershell
docker run --name mediask-api -p 8000:8000 mediask
```

Verify the containerized API:

```powershell
curl.exe http://localhost:8000/
```

Then start Streamlit locally:

```powershell
$env:MEDIGUIDE_API_URL = "http://localhost:8000"
streamlit run app\ui.py
```

For persistent local data and indexes, mount the project directories into the
container. PowerShell uses the backtick for multiline commands:

```powershell
docker run --name mediask-api `
  -p 8000:8000 `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\rag\faiss_index:/app/rag/faiss_index" `
  mediask
```

With these mounts, uploaded files remain in the local `data\` directory and
the generated FAISS files remain in `rag\faiss_index\` when the container is
removed.

Useful lifecycle commands:

```powershell
docker logs -f mediask-api
docker stop mediask-api
docker start mediask-api
docker rm mediask-api
```

Run `docker rm mediask-api` only after stopping the container. If Docker
reports that the name is already in use, stop and remove the existing
container before creating another one with that name.

## API

`POST /upload` accepts one Markdown (`.md`), PDF (`.pdf`), or Word (`.docx`)
medical file as multipart form data. The file is stored in `data/` and indexed
together with existing supported files:

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@path/to/medical-reference.pdf"
```

Uploads are limited to supported file types up to 10 MB. A successful response reports
the original filename and the number of indexed pages and chunks. The
Streamlit UI provides the same feature through its file uploader. Other formats
are rejected.

`POST /ask` accepts a JSON body with a query between 3 and 1000 characters:

```json
{
  "query": "what is Acetaminophen Injection "
}
```

The response contains the generated answer and retrieved source text:

```json
{
  "answer": "...",
  "sources": ["..."]
}
```

`GET /` is a lightweight health endpoint. If the configured model is
unavailable, `/ask` returns HTTP 503. Invalid provider settings or missing API
keys return HTTP 500 with a configuration message. Unexpected retrieval or
generation failures return HTTP 500 and are logged by the API.

## Testing

Run the standard-library test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Safety and limitations

MediAsk is an educational information-retrieval tool, not a medical
diagnostic or treatment service. Responses can be incomplete or incorrect.
Users should verify medical information with a qualified professional,
especially for pregnancy, children, allergies, interactions, or emergencies.
The source PDFs should be reviewed and updated by the project owner before
the application is used with real users.

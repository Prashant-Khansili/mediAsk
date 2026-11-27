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

The response contains the guarded answer (always ending with the "Not medical
advice" disclaimer), whether the system refused, and the retrieved sources with
their drug, section, and page:

```json
{
  "answer": "The maximum daily dose is 4,000 mg per day [1].\n\n⚠️ Not medical advice. ...",
  "refused": false,
  "refusal_reason": null,
  "disclaimer": "⚠️ Not medical advice. ...",
  "unsupported_sentences": [],
  "sources": ["[Acetaminophen Injection | 2.2 Recommended Dosage: Adults and Adolescents]\n..."],
  "citations": [{"id": 1, "drug": "Acetaminophen Injection", "section": "2.2 ...", "page": 3, "source": "file.pdf"}]
}
```

`[n]` markers in the answer refer to `citations[n-1]`. When the system refuses,
`answer` is "This isn't in my documents." and `refusal_reason` is one of
`low_retrieval_score`, `model_not_found`, or `no_supported_sentences`.

## Retrieval and guardrail settings

All settings are environment variables (for example in `.env`). Changing the
embedding model or chunking strategy requires rebuilding the index with
`python -m rag.ingest`; the index records the settings it was built with in
`rag/faiss_index/index_config.json`, and the API reads the embedding model from there.

| Variable | Default | Meaning |
|---|---|---|
| `CHUNKING_STRATEGY` | `structured` | `structured` (split by headings, tables kept whole, drug/section/page metadata) or `fixed` (500-character chunks) |
| `EMBEDDING_MODEL` | `e5-small` | also `all-MiniLM-L6-v2`, `bge-small`, `bge-base`, `e5-base`, or any sentence-transformers id |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` (BM25 + vector, reciprocal rank fusion), `vector`, or `bm25` |
| `RERANKER` | none | `ms-marco-MiniLM` or `bge-reranker-base`; re-scores the top `RETRIEVAL_CANDIDATES` (20) |
| `RETRIEVAL_TOP_K` | `5` | chunks passed to the model |
| `MIN_RETRIEVAL_SCORE` | per model (`e5-small` 0.80, MiniLM 0.30) | refuse if the best vector cosine similarity is below this. Cosine scales differ between models, so the default is calibrated per model on the eval set |
| `MIN_RERANK_SCORE` | none | refuse if the best re-ranker score (0–1) is below this; used instead of the vector threshold when a re-ranker is on |
| `CITATION_CHECK` | `lexical` | `lexical` (numbers must match and most words must appear in the cited chunk), `llm` (ask the model to verify each sentence), or `off` |
| `CITATION_ACTION` | `drop` | `drop` unsupported sentences or `flag` them with `[unverified: ...]` |

### How the guardrails work

1. **Refuse when unsure.** If the best retrieved chunk scores below the
   threshold, the model is not called and the answer is "This isn't in my documents."
2. **Prompt-injection defence.** Retrieved chunks are HTML-escaped (so a document
   cannot close or forge the `<documents>` / `<question>` tags) and wrapped in
   `<document id=.. drug=.. section=.. page=..>` delimiters. The system prompt says
   everything inside `<documents>` is untrusted data, never instructions.
3. **Grounded answers.** The model must cite a chunk id on every sentence, or reply
   `NOT_FOUND`, which becomes the refusal message.
4. **Citation check.** Each sentence is checked against the chunk it cites. Sentences
   whose numbers do not appear in that chunk, or with low word overlap, are dropped
   (or flagged). If nothing survives, the system refuses.
5. **Disclaimer.** "Not medical advice" is appended to every answer, refusals included.

`GET /` is a lightweight health endpoint. If the configured model is
unavailable, `/ask` returns HTTP 503. Invalid provider settings or missing API
keys return HTTP 500 with a configuration message. Unexpected retrieval or
generation failures return HTTP 500 and are logged by the API.

## Testing

Run the standard-library test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Offline evaluation (retrieval, LLM-as-judge, guardrails)

The evaluation set is `evaluation/eval_set.jsonl`: 50 questions written from the
indexed label: 40 answerable and 10 the documents cannot answer (other drugs,
near-misses such as dosing for neonates under 32 weeks, prices). Each answerable
case stores the reference answer, a verbatim **evidence span**, and its page and
section. A retrieved chunk counts as the correct chunk when it contains the
evidence span, so hit@k stays comparable when the chunking strategy changes.

```json
{"id": "q02", "question": "What is the largest single dose an adult over 50 kg can receive?", "answerable": true,
 "reference_answer": "The maximum single dose is 1,000 mg.",
 "evidence": "maximum single dose of Acetaminophen Injection of 1,000 mg", "page": 3,
 "section": "2.2 Recommended Dosage: Adults and Adolescents", "kind": "dosage"}
```

Three scripts, run from the repository root (no API server needed):

```powershell
# 1. Retrieval ablation: one change at a time, hit@1 / hit@5 / MRR@5. No LLM needed.
python -m evaluation.retrieval_experiments

# 2. End-to-end: correctness (LLM judge, 0-3 rubric), faithfulness, refusal accuracy
python -m evaluation.evaluate --label final --provider gemini --model gemini-3.5-flash-lite `
  --mode hybrid --embedding e5-small --min-vector-score 0.80
python -m evaluation.evaluate --label baseline --provider gemini --model gemini-3.5-flash-lite `
  --naive --chunking fixed --no-dedupe --mode vector

# 3. Prompt-injection red team (poisoned documents, guarded vs naive prompt)
python -m evaluation.injection_test --provider gemini --model gemini-3.5-flash-lite
```

### Running on a CPU-only laptop

The end-to-end evaluation does **not** need a local LLM. Answers and judging go
through an API (the free Gemini tier is enough: about 2 calls per question,
~100 per 50-question run). The only model loaded locally is the small embedding
model (~100–150 MB of RAM), so the run is safe on a machine with little free memory.

- **Resumable.** Each question is appended to `evaluation/results/e2e_<label>.answers.jsonl`
  and `.judged.jsonl` as soon as it finishes. If the run crashes, is stopped with
  Ctrl-C, or the API quota runs out, run the same command again and it continues
  from the next question. `--fresh` starts over.
- **Batches.** `--ids q01 q02 ...` or `--limit 25` run part of the set.
- **Phases.** `--phase generate` only produces answers, `--phase judge` only scores
  saved answers, so you can re-judge later with another model (`--judge-model`).
- **Pacing.** With Gemini, a 4-second pause between questions keeps the run under
  the free-tier rate limit (`--sleep` overrides it). Rate-limit errors are retried
  with back-off, and an exhausted daily quota stops the run cleanly.
- Run one evaluation at a time. Avoid `--reranker bge-reranker-base` on small
  machines: it needs ~1.5 GB of RAM and ~10 s per query on CPU.

Outputs are written to `evaluation/results/`: JSON with per-question details,
a Markdown table for the ablation, and `spot_check_<label>.md`, a 10-question
sheet for checking the LLM judge by hand. The judge model can be set separately
with `--judge-provider` / `--judge-model`; using the same model as the generator
can bias scores upward, which is why the spot-check exists.

## Safety and limitations

MediAsk is an educational information-retrieval tool, not a medical
diagnostic or treatment service. Responses can be incomplete or incorrect.
Users should verify medical information with a qualified professional,
especially for pregnancy, children, allergies, interactions, or emergencies.
The source PDFs should be reviewed and updated by the project owner before
the application is used with real users.

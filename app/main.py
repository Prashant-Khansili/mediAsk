import logging
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from rag.ingest import DEFAULT_DATA_DIR, build_index
from rag.rag_chain import LLMConfigurationError, get_rag_chain

logger = logging.getLogger(__name__)
app = FastAPI(title="MediAsk API")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".md", ".pdf", ".docx"}


class Question(BaseModel):
    query: str = Field(min_length=3, max_length=1000)


@lru_cache(maxsize=8)
def get_qa_chain(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
):
    return get_rag_chain(provider=provider, model=model, api_key=api_key)


@app.get("/")
def root():
    return {"status": "MediAsk is running 🚀"}


@app.post("/upload")
async def upload_medical_file(file: UploadFile = File(...)):
    """Store a supported medical file and rebuild the retrieval index."""
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if not filename or extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .md, .pdf, and .docx medical files are supported.",
        )

    data_dir = Path(DEFAULT_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    stored_path = data_dir / f"{uuid4().hex}_{filename}"
    total_bytes = 0

    try:
        with stored_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Medical files must be 10 MB or smaller.",
                    )
                destination.write(chunk)

        pages, chunks = await run_in_threadpool(build_index)
        get_qa_chain.cache_clear()
    except HTTPException:
        stored_path.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as exc:
        stored_path.unlink(missing_ok=True)
        logger.exception("Medical file indexing failed")
        raise HTTPException(
            status_code=500,
            detail="The medical file could not be indexed.",
        ) from exc
    finally:
        await file.close()

    return {
        "filename": filename,
        "pages_indexed": pages,
        "chunks_indexed": chunks,
    }


@app.post("/ask")
def ask_question(
    q: Question,
    llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
    llm_model: str | None = Header(default=None, alias="X-LLM-Model"),
    llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
):
    try:
        result = get_qa_chain(
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
        ).invoke({"query": q.query.strip()})
    except (ConnectionError, TimeoutError) as exc:
        logger.exception("The local language model could not be reached")
        raise HTTPException(
            status_code=503,
            detail="The configured language model is unavailable.",
        ) from exc
    except LLMConfigurationError as exc:
        logger.error("Language model configuration is invalid: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question.",
        ) from exc

    return {
        "answer": result["result"],
        "sources": [
            doc.page_content for doc in result.get("source_documents", [])
        ],
    }

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from rich.logging import RichHandler

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from rag.embeddings import PrefixedSentenceEmbeddings
from rag.guardrails import (
    DISCLAIMER,
    REFUSAL_MESSAGE,
    GuardrailSettings,
    _text_of,
    apply_citation_report,
    build_messages,
    check_citations,
    is_not_found,
    with_disclaimer,
)
from rag.ingest import read_index_config
from rag.retrieval import HybridRetriever, RetrievalConfig

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger("mediGuide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_INDEX_DIR = PROJECT_ROOT / "rag" / "faiss_index"
DEFAULT_MODELS = {
    "ollama": "llama3",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "gemini": "gemini-3.8-flash",
}


class LLMConfigurationError(ValueError):
    """Raised when the selected language model is not configured correctly."""


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
):
    """Build a configured chat model using optional request-scoped settings."""
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
    model = (model or os.getenv("LLM_MODEL", DEFAULT_MODELS.get(provider, ""))).strip()
    api_key = api_key.strip() if api_key else None

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        logger.info("🧠 Loading local Ollama model: %s", model)
        return ChatOllama(model=model)

    if provider == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai."
            )
        from langchain_openai import ChatOpenAI

        options = {"model": model, "api_key": api_key}
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        if base_url:
            options["base_url"] = base_url
        logger.info("🧠 Loading OpenAI-compatible model: %s", model)
        return ChatOpenAI(**options)

    if provider == "anthropic":
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic."
            )
        from langchain_anthropic import ChatAnthropic

        logger.info("🧠 Loading Anthropic model: %s", model)
        return ChatAnthropic(model=model, api_key=api_key)

    if provider == "gemini":
        api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is required when LLM_PROVIDER=gemini."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("🧠 Loading Google Gemini model: %s", model)
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

    supported = ", ".join(sorted(DEFAULT_MODELS))
    raise LLMConfigurationError(
        f"Unsupported LLM_PROVIDER '{provider}'. Choose one of: {supported}."
    )


class MediAskPipeline:
    """Retrieve -> refuse if unsure -> generate with cited sources -> verify citations."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm,
        settings: GuardrailSettings | None = None,
        verifier_llm=None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings or GuardrailSettings.from_env()
        self.verifier_llm = verifier_llm or llm

    def _refusal(self, reason: str, retrieval, raw_answer: str | None = None, report=None) -> dict:
        return {
            "answer": with_disclaimer(REFUSAL_MESSAGE),
            "answer_text": REFUSAL_MESSAGE,
            "refused": True,
            "refusal_reason": reason,
            "raw_answer": raw_answer,
            "disclaimer": DISCLAIMER,
            "unsupported_sentences": [
                {"sentence": s.text, "reason": s.reason} for s in (report.unsupported if report else [])
            ],
            **self._retrieval_payload(retrieval),
        }

    @staticmethod
    def _retrieval_payload(retrieval) -> dict:
        return {
            "sources": [
                {
                    "id": number,
                    "drug": chunk.document.metadata.get("drug"),
                    "section": chunk.document.metadata.get("section"),
                    "page": chunk.document.metadata.get("page"),
                    "source": Path(str(chunk.document.metadata.get("source", ""))).name,
                    "text": chunk.document.page_content,
                    "vector_score": chunk.vector_score,
                    "rerank_score": chunk.rerank_score,
                }
                for number, chunk in enumerate(retrieval.chunks, start=1)
            ],
            "retrieval": {
                "best_vector_score": retrieval.best_vector_score,
                "best_rerank_score": retrieval.best_rerank_score,
            },
        }

    def is_confident(self, retrieval) -> bool:
        s = self.settings
        if retrieval.best_rerank_score is not None and s.min_rerank_score is not None:
            return retrieval.best_rerank_score >= s.min_rerank_score
        return retrieval.best_vector_score >= s.min_vector_score

    def answer(self, question: str) -> dict:
        retrieval = self.retriever.retrieve(question)
        if not retrieval.chunks or not self.is_confident(retrieval):
            return self._refusal("low_retrieval_score", retrieval)

        documents = retrieval.documents
        raw = _text_of(self.llm.invoke(build_messages(question, documents))).strip()
        if not raw or is_not_found(raw):
            return self._refusal("model_not_found", retrieval, raw)

        report = check_citations(raw, documents, self.settings, self.verifier_llm)
        final = apply_citation_report(report, self.settings.citation_action)
        if not final or is_not_found(final):
            return self._refusal("no_supported_sentences", retrieval, raw, report)

        return {
            "answer": with_disclaimer(final),
            "answer_text": final,
            "refused": False,
            "refusal_reason": None,
            "raw_answer": raw,
            "disclaimer": DISCLAIMER,
            "unsupported_sentences": [
                {"sentence": s.text, "reason": s.reason} for s in report.unsupported
            ],
            **self._retrieval_payload(retrieval),
        }

    def invoke(self, inputs: dict) -> dict:
        return self.answer(inputs["query"])


def load_retriever(index_dir: str | Path = DEFAULT_INDEX_DIR, config: RetrievalConfig | None = None):
    index_config = read_index_config(index_dir)
    embeddings = PrefixedSentenceEmbeddings(index_config.get("embedding_model"))
    logger.info("📚 Loading FAISS index (%s, %s chunks)...", embeddings.name, index_config.get("chunking"))
    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    return HybridRetriever(vectorstore, config or RetrievalConfig.from_env())


def get_rag_chain(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
):
    """Load the persisted index and construct the guarded RAG pipeline."""
    retriever = load_retriever(index_dir)
    llm = get_llm(provider=provider, model=model, api_key=api_key)
    settings = GuardrailSettings.from_env(retriever.vectorstore.embedding_function.name)
    logger.info("✅ RAG pipeline is ready (refusal threshold %.2f).", settings.min_vector_score)
    return MediAskPipeline(retriever, llm, settings)

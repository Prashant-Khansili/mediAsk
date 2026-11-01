import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from rich.logging import RichHandler

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

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


def get_rag_chain(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
):
    """Load the persisted index and construct a configured RAG chain."""
    logger.info("📦 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    logger.info("📚 Loading FAISS vector store...")
    vectorstore = FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )

    logger.info("🔍 Building retriever...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = get_llm(provider=provider, model=model, api_key=api_key)

    logger.info("🔗 Creating RAG QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    logger.info("✅ RAG chain is ready.")
    return qa_chain

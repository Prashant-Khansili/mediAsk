import os

import requests
import streamlit as st


SUPPORTED_FILE_TYPES = ["md", "pdf", "docx"]
DEFAULT_API_URL = os.getenv("MEDIGUIDE_API_URL", "http://localhost:8000")
EXAMPLE_QUESTIONS = [
    "What are the common side effects of ibuprofen?",
    "Are there medication interactions I should know about?",
    "What should I ask my doctor before starting this medicine?",
]

st.set_page_config(
    page_title="MediAsk",
    page_icon="images/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "gemini"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "gemini-3.8-flash"
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""


def api_error(response: requests.Response) -> str:
    try:
        return response.json().get("detail", "The API returned an error.")
    except ValueError:
        return f"The API returned HTTP {response.status_code}."


def ask_api(api_url: str, query: str) -> dict:
    headers = {
        "X-LLM-Provider": st.session_state.llm_provider,
        "X-LLM-Model": st.session_state.llm_model,
    }
    if st.session_state.llm_api_key:
        headers["X-LLM-API-Key"] = st.session_state.llm_api_key
    response = requests.post(
        f"{api_url}/ask",
        json={"query": query},
        headers=headers,
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(api_error(response))
    return response.json()


def upload_file(api_url: str, uploaded_file) -> dict:
    response = requests.post(
        f"{api_url}/upload",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            )
        },
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(api_error(response))
    return response.json()


def render_message(message: dict):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources", [])
        if sources:
            with st.expander(f"📚 Retrieved sources ({len(sources)})"):
                for index, source in enumerate(sources, start=1):
                    st.markdown(f"**Source {index}**")
                    st.caption(source)


initialize_session_state()

st.markdown(
    """
    <style>
        .stApp {
            background: #121212;
            color: #f1f1f1;
        }
        [data-testid="stSidebar"] {
            background: #1b1b1b;
            border-right: 1px solid #343434;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] small {
            color: #f1f1f1 !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #b8b8b8 !important;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #252525 !important;
            color: #f1f1f1 !important;
            border-color: #555555 !important;
        }
        [data-testid="stSidebar"] input::placeholder {
            color: #a7a7a7 !important;
            opacity: 1;
        }
        [data-testid="stSidebar"] button {
            background: #252525 !important;
            color: #f1f1f1 !important;
            border-color: #5b7780 !important;
        }
        [data-testid="stSidebar"] button:disabled {
            background: #303030 !important;
            color: #888888 !important;
            border-color: #4a4a4a !important;
        }
        [data-testid="stSidebar"] button:hover {
            color: #ffffff !important;
            border-color: #48a9c5 !important;
            background: #303f44 !important;
        }
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"] {
            background: #252525 !important;
            color: #f1f1f1 !important;
        }
        [data-baseweb="menu"] [role="option"] {
            color: #f1f1f1 !important;
            background: #252525 !important;
        }
        [data-baseweb="menu"] [role="option"]:hover,
        [data-baseweb="menu"] [aria-selected="true"] {
            color: #ffffff !important;
            background: #303f44 !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: #252525;
            border: 1px dashed #5b7780;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {
            color: #b8b8b8 !important;
        }
        [data-testid="stSidebar"] [data-testid="stProgressBar"] {
            background: #3a3a3a;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"] {
            color: #f1f1f1;
        }
        .hero {
            padding: 1.5rem 1.75rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, #202f33, #284b55);
            border: 1px solid #3c707d;
            color: white;
            margin-bottom: 1.5rem;
        }
        .hero h1 { margin: 0; font-size: 2.4rem; }
        .hero h1, .hero p { color: #f1f1f1; }
        .hero p { margin: 0.5rem 0 0; }
        .disclaimer {
            color: #b8b8b8;
            font-size: 0.85rem;
            text-align: center;
            margin-top: 2rem;
        }
        .meeseeks-float {
            position: fixed;
            right: 24px;
            bottom: 24px;
            z-index: 999;
            width: 76px;
            height: 76px;
            border-radius: 50%;
            background: #48a9c5;
            border: 4px solid #247a95;
            box-shadow: 0 8px 20px rgba(30, 86, 103, 0.25);
            animation: meeseeks-bob 3s ease-in-out infinite;
        }
        .meeseeks-eye {
            position: absolute;
            top: 24px;
            width: 9px;
            height: 13px;
            border-radius: 50%;
            background: #102b38;
        }
        .meeseeks-eye.left { left: 21px; }
        .meeseeks-eye.right { right: 21px; }
        .meeseeks-smile {
            position: absolute;
            left: 25px;
            top: 43px;
            width: 19px;
            height: 9px;
            border-bottom: 3px solid #102b38;
            border-radius: 50%;
        }
        .meeseeks-label {
            position: absolute;
            right: 0;
            bottom: -25px;
            width: 120px;
            color: #247a95;
            font-size: 0.72rem;
            font-weight: 600;
            text-align: right;
        }
        @keyframes meeseeks-bob {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-7px); }
        }
        @media (max-width: 700px) {
            .meeseeks-float { right: 12px; bottom: 12px; transform: scale(0.82); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="meeseeks-float" aria-label="MediAsk helper">
        <span class="meeseeks-eye left"></span>
        <span class="meeseeks-eye right"></span>
        <span class="meeseeks-smile"></span>
        <span class="meeseeks-label"> Mr Meeseeks!</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🩺 MediAsk</h1>
        <p>Ask questions about your medical reference files and get grounded,
        AI-assisted answers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Workspace")
    api_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL).strip().rstrip("/")
    st.caption("The API must be running before you upload files or ask a question.")

    st.subheader("🤖 AI provider")
    st.selectbox(
        "Provider",
        ["gemini", "openai", "anthropic", "ollama"],
        key="llm_provider",
        help="Your key is sent only with each question and is not saved by this app.",
    )
    st.text_input(
        "Model",
        key="llm_model",
        help="Use a model supported by the selected provider.",
    )
    st.text_input(
        "API key",
        type="password",
        key="llm_api_key",
        placeholder="Enter key for the selected provider",
        help="Kept only in this Streamlit session and sent over your configured connection.",
    )
    if st.session_state.llm_provider == "ollama":
        st.caption("Ollama does not require an API key.")

    if st.button("🩺 Check API status", use_container_width=True):
        try:
            response = requests.get(f"{api_url}/", timeout=30)
            response.raise_for_status()
            st.success("API is online")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")

    st.divider()
    st.subheader("📄 Medical references")
    uploaded_files = st.file_uploader(
        "Add files to the knowledge base",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help="Only Markdown (.md), PDF (.pdf), and Word (.docx) files are accepted.",
    )
    if st.button(
        "📚 Index selected files",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        progress = st.progress(0, text="Preparing files...")
        successes = 0
        for index, uploaded_file in enumerate(uploaded_files):
            try:
                result = upload_file(api_url, uploaded_file)
                if uploaded_file.name not in st.session_state.indexed_files:
                    st.session_state.indexed_files.append(uploaded_file.name)
                successes += 1
                st.success(
                    f"Indexed `{uploaded_file.name}` "
                    f"({result['chunks_indexed']} chunks)"
                )
            except (requests.RequestException, RuntimeError) as exc:
                st.error(f"Could not index `{uploaded_file.name}`: {exc}")
            progress.progress(
                (index + 1) / len(uploaded_files),
                text=f"Processed {index + 1} of {len(uploaded_files)} files",
            )
        if successes:
            st.info(f"{successes} file(s) added to the knowledge base.")

    if st.session_state.indexed_files:
        st.caption("Files indexed this session")
        for filename in st.session_state.indexed_files:
            st.markdown(f"- `{filename}`")

    st.divider()
    st.subheader("Try an example")
    example_question = st.selectbox("Example question", EXAMPLE_QUESTIONS)
    if st.button("Use example", use_container_width=True):
        st.session_state.example_query = example_question
        st.rerun()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.subheader("💬 Ask MediAsk")
st.caption("Answers are based on indexed sources. Always verify medical decisions with a qualified professional.")

for message in st.session_state.messages:
    render_message(message)

query = st.chat_input("Ask about your indexed medical files...")
if query is None:
    query = st.session_state.pop("example_query", None)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    render_message(st.session_state.messages[-1])
    with st.chat_message("assistant"):
        with st.spinner("Reviewing sources and preparing an answer..."):
            try:
                result = ask_api(api_url, query)
                answer = result.get("answer", "The API returned no answer.")
                sources = result.get("sources", [])
                st.markdown(answer)
                if sources:
                    with st.expander(f"📚 Retrieved sources ({len(sources)})"):
                        for index, source in enumerate(sources, start=1):
                            st.markdown(f"**Source {index}**")
                            st.caption(source)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except requests.RequestException as exc:
                error = f"Could not connect to the MediAsk API: {exc}"
                st.error(error)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error}
                )
            except RuntimeError as exc:
                st.error(str(exc))
                st.session_state.messages.append(
                    {"role": "assistant", "content": str(exc)}
                )

st.markdown(
    '<div class="disclaimer">MediAsk is an educational tool, not a '
    "replacement for professional medical advice. For emergencies, contact "
    "local emergency services.</div>",
    unsafe_allow_html=True,
)

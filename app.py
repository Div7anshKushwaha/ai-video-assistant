"""Streamlit interface for the AI Video Assistant.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv


st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()


def load_streamlit_secrets_into_environment() -> None:
    """Make Streamlit Cloud secrets available to modules using os.getenv()."""

    try:
        secrets = st.secrets
    except Exception:
        return

    for key in ("GROQ_API_KEY", "SARVAM_API_KEY", "WHISPER_MODEL", "SARVAM_STT_MODEL"):
        if not os.getenv(key) and key in secrets:
            os.environ[key] = str(secrets[key])


load_streamlit_secrets_into_environment()

# Import after loading .env and Streamlit secrets because the core modules read
# configuration from environment variables during import or object creation.
from core.rag_engine import ask_question  # noqa: E402
from main import run_pipeline  # noqa: E402


LANGUAGES = {
    "English": "english",
    "Hinglish": "hinglish",
}

ALLOWED_EXTENSIONS = [
    "mp3",
    "wav",
    "m4a",
    "aac",
    "flac",
    "ogg",
    "mp4",
    "mkv",
    "mov",
    "webm",
]

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

MAX_UPLOAD_SIZE_MB = 200

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 1100px;}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
}
.hero-sub {
    font-size: 1.15rem;
    opacity: 0.65;
    margin: 0.25rem 0 1.75rem 0;
}
.result-title {
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 1rem 0;
}
.notice {
    padding: 0.8rem 1rem;
    border-radius: 0.6rem;
    background: rgba(127, 127, 127, 0.10);
}
</style>
"""


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
def init_state() -> None:
    """Initialize all session-state keys exactly once."""

    defaults = {
        "processed": False,
        "title": "",
        "transcript": "",
        "summary": "",
        "action_items": "",
        "key_decisions": "",
        "open_questions": "",
        "rag_chain": None,
        "messages": [],
        "source_label": "",
        "language_label": "English",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    """Clear the current analysis and return to the input screen."""

    for key in list(st.session_state.keys()):
        del st.session_state[key]


# -----------------------------------------------------------------------------
# Input and error helpers
# -----------------------------------------------------------------------------
def normalize_youtube_url(url: str) -> str | None:
    """Return a valid YouTube URL or None."""

    value = url.strip()
    if not value:
        return None

    if not value.lower().startswith(("http://", "https://")):
        value = "https://" + value

    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower().rstrip(".")

    if hostname not in YOUTUBE_HOSTS:
        return None

    return value


def save_upload(uploaded_file) -> str | None:
    """Save a supported Streamlit upload to a temporary file."""

    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")

    if suffix not in ALLOWED_EXTENSIONS:
        return None

    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        raise ValueError(
            f"File is {file_size_mb:.1f} MB. The maximum allowed size is "
            f"{MAX_UPLOAD_SIZE_MB} MB."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as temp:
        temp.write(uploaded_file.getbuffer())
        return temp.name


def redact(text: str) -> str:
    """Remove configured secret values before displaying technical errors."""

    redacted = text

    for name, value in os.environ.items():
        is_sensitive = any(
            token in name.upper()
            for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")
        )
        if is_sensitive and value and len(value) >= 8:
            redacted = redacted.replace(value, "***")

    return redacted


def friendly_error(error: Exception, stage: int) -> str:
    """Convert a raw pipeline exception into an actionable UI message."""

    message = str(error)
    msg = message.lower()

    # YouTube-specific failures must be checked before generic stage errors.
    is_youtube_error = any(
        token in msg
        for token in (
            "youtube",
            "yt-dlp",
            "downloaderror",
            "unable to download",
            "download failed",
        )
    )

    if is_youtube_error and ("403" in msg or "forbidden" in msg):
        return (
            "YouTube blocked this cloud download request (HTTP 403). "
            "The upload workflow is reliable—download the audio/video locally "
            "and upload the file instead."
        )

    if is_youtube_error and "requested format is not available" in msg:
        return (
            "YouTube did not provide a compatible audio format for this video. "
            "Try another public video or upload the audio/video file instead."
        )

    if is_youtube_error and (
        "bot" in msg
        or "sign in" in msg
        or "challenge" in msg
    ):
        return (
            "YouTube requires bot verification for this cloud request. "
            "Please upload the audio/video file instead."
        )

    if any(
        token in msg
        for token in (
            "api key",
            "unauthorized",
            "invalid_api_key",
            "authentication",
        )
    ):
        return (
            "An AI service rejected the request. Check GROQ_API_KEY and, "
            "for Hinglish mode, SARVAM_API_KEY in your environment or secrets."
        )

    if any(token in msg for token in ("429", "rate limit", "quota")):
        return "An AI service rate limit was reached. Please wait and try again."

    if "ffmpeg" in msg:
        return (
            "FFmpeg is required for audio conversion but was not available. "
            "Install FFmpeg locally or check the deployment system packages."
        )

    if stage == 0:
        return (
            "The audio could not be downloaded or processed. Check the input "
            "or upload a local audio/video file."
        )

    if stage == 1:
        return (
            "Transcription failed. Try a shorter file or change the language mode."
        )

    if 2 <= stage <= 6:
        return "The AI model could not generate this analysis section. Please try again."

    return "The transcript knowledge base could not be built. Please try again."


# -----------------------------------------------------------------------------
# Pipeline execution
# -----------------------------------------------------------------------------
def run_analysis(source: str, language: str) -> dict | None:
    """Run the pipeline while displaying real progress in Streamlit."""

    status = st.status("Starting analysis...", expanded=True)
    progress = st.progress(0.0)
    tracker = {
        "current": None,
        "index": 0,
        "total": 1,
    }

    def on_progress(index: int, total: int, label: str) -> None:
        if tracker["current"]:
            status.write(f"✓ {tracker['current']}")

        tracker.update(
            current=label,
            index=index,
            total=total,
        )

        status.update(label=f"{label}...", state="running")
        progress.progress(min(index / max(total, 1), 1.0))

    try:
        result = run_pipeline(
            source,
            language,
            on_progress=on_progress,
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary
        status.update(
            label="Analysis failed",
            state="error",
            expanded=False,
        )
        progress.empty()
        st.error(friendly_error(exc, tracker["index"]))

        with st.expander("Technical details"):
            st.code(redact(f"{type(exc).__name__}: {exc}"))

        return None

    if tracker["current"]:
        status.write(f"✓ {tracker['current']}")

    progress.progress(1.0)
    status.update(
        label="Analysis complete",
        state="complete",
        expanded=False,
    )

    return result


def handle_analyze(url: str, uploaded_file, language_label: str) -> None:
    """Validate input, run analysis, and save results in session state."""

    temporary_path: str | None = None
    label: str | None = None

    try:
        if uploaded_file is not None:
            temporary_path = save_upload(uploaded_file)

            if temporary_path is None:
                st.error(
                    "Unsupported file type. Supported formats: "
                    + ", ".join(ALLOWED_EXTENSIONS)
                )
                return

            source = temporary_path
            label = uploaded_file.name
        else:
            if not url.strip():
                st.error("Enter a YouTube URL or upload an audio/video file.")
                return

            source = normalize_youtube_url(url)
            if source is None:
                st.error(
                    "That does not look like a YouTube URL. Use a youtube.com "
                    "or youtu.be link."
                )
                return

            label = source

        result = run_analysis(source, LANGUAGES[language_label])

        if result is None:
            return

        st.session_state.update(
            processed=True,
            title=result.get("title", "Untitled video"),
            transcript=result.get("transcript", ""),
            summary=result.get("summary", ""),
            action_items=result.get("action_items", ""),
            key_decisions=result.get("key_decisions", ""),
            open_questions=result.get("open_questions", ""),
            rag_chain=result.get("rag_chain"),
            messages=[],
            source_label=label or "Processed media",
            language_label=language_label,
        )

        st.rerun()

    except Exception as exc:  # noqa: BLE001 - UI boundary
        st.error(friendly_error(exc, 0))
        with st.expander("Technical details"):
            st.code(redact(f"{type(exc).__name__}: {exc}"))

    finally:
        if temporary_path:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass


# -----------------------------------------------------------------------------
# UI sections
# -----------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🎥 AI Video Assistant")

        if st.session_state.processed:
            st.caption("Current media")
            st.write(st.session_state.source_label)

            if st.button("Analyze another video", use_container_width=True):
                reset_session()
                st.rerun()

            st.divider()

        st.caption("Configuration")
        st.write(
            ("✅" if os.getenv("GROQ_API_KEY") else "❌")
            + " Groq API key"
        )
        st.write(
            ("✅" if os.getenv("SARVAM_API_KEY") else "⚪")
            + " Sarvam API key (optional)"
        )

        st.divider()
        st.caption(
            "Upload is the most reliable workflow. YouTube links may be "
            "blocked by cloud hosting restrictions."
        )


def render_input_section() -> None:
    _, middle, _ = st.columns([1, 4, 1])

    with middle:
        st.markdown(
            '<p class="hero-title">🎥 AI Video Assistant</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hero-sub">Turn videos into searchable knowledge.</p>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.info(
                "For the most reliable results, upload an audio/video file. "
                "YouTube links may be blocked by cloud-hosting restrictions."
            )

            url = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )

            with st.expander("Or upload a local audio/video file", expanded=True):
                uploaded = st.file_uploader(
                    "Audio or video file",
                    type=ALLOWED_EXTENSIONS,
                    label_visibility="collapsed",
                )
                st.caption(
                    f"Maximum file size: {MAX_UPLOAD_SIZE_MB} MB. "
                    "An uploaded file takes priority over the URL."
                )

            left, right = st.columns([2, 1], vertical_alignment="bottom")

            with left:
                language_label = st.selectbox(
                    "Language",
                    list(LANGUAGES.keys()),
                )

            with right:
                clicked = st.button(
                    "🚀 Analyze Media",
                    type="primary",
                    use_container_width=True,
                )

        if clicked:
            handle_analyze(url, uploaded, language_label)


def render_text_block(content, empty_message: str) -> None:
    """Render LLM output that may be text or a list."""

    if not content:
        st.info(empty_message)
    elif isinstance(content, (list, tuple)):
        st.markdown("\n".join(f"- {item}" for item in content))
    else:
        st.markdown(str(content))


def render_transcript() -> None:
    transcript = st.session_state.transcript
    words = len(transcript.split())

    top_left, top_right = st.columns([3, 1], vertical_alignment="center")
    top_left.caption(f"{words:,} words")
    top_right.download_button(
        "Download .txt",
        transcript,
        file_name="transcript.txt",
        use_container_width=True,
    )

    with st.expander("Show full transcript", expanded=False):
        st.text_area(
            "Transcript",
            transcript,
            height=420,
            label_visibility="collapsed",
        )


def split_answer(response) -> tuple[str, list]:
    """Support both current string answers and future answer dictionaries."""

    if isinstance(response, dict):
        answer = response.get("answer") or response.get("result") or str(response)
        sources = response.get("sources") or response.get("source_documents") or []
        return str(answer), list(sources)

    return str(response), []


def render_sources(sources: list) -> None:
    if not sources:
        return

    with st.expander("📚 Retrieved context"):
        for index, document in enumerate(sources, start=1):
            text = getattr(document, "page_content", str(document))
            metadata = getattr(document, "metadata", {}) or {}
            chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index", index)

            st.markdown(f"**Chunk {chunk_id}**")
            st.caption(text[:500] + ("..." if len(text) > 500 else ""))


def render_chat() -> None:
    st.caption("Ask questions about the processed transcript.")

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about the video...")
    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "sources": [],
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the transcript..."):
                response = ask_question(
                    st.session_state.rag_chain,
                    question,
                )

            answer, sources = split_answer(response)
            st.markdown(answer)
            render_sources(sources)

        except Exception as exc:  # noqa: BLE001 - keep chat usable
            answer = "I could not answer that question. Please try rephrasing it."
            sources = []
            st.error(answer)

            with st.expander("Technical details"):
                st.code(redact(f"{type(exc).__name__}: {exc}"))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


def render_results() -> None:
    title = html.escape(st.session_state.title or "Untitled video")
    st.markdown(
        f'<p class="result-title">{title}</p>',
        unsafe_allow_html=True,
    )

    words = len(st.session_state.transcript.split())
    col1, col2, col3 = st.columns(3)
    col1.metric("Transcript length", f"{words:,} words")
    col2.metric("Language", st.session_state.language_label)
    col3.metric(
        "Chat questions",
        sum(message["role"] == "user" for message in st.session_state.messages),
    )

    tabs = st.tabs(
        [
            "📋 Summary",
            "📝 Transcript",
            "✅ Action Items",
            "🔑 Decisions",
            "❓ Questions",
            "💬 Chat",
        ]
    )

    renderers = [
        lambda: render_text_block(
            st.session_state.summary,
            "No summary was generated.",
        ),
        render_transcript,
        lambda: render_text_block(
            st.session_state.action_items,
            "No action items were found.",
        ),
        lambda: render_text_block(
            st.session_state.key_decisions,
            "No key decisions were found.",
        ),
        lambda: render_text_block(
            st.session_state.open_questions,
            "No open questions were found.",
        ),
        render_chat,
    ]

    for tab, renderer in zip(tabs, renderers):
        with tab:
            renderer()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
init_state()
render_sidebar()

if st.session_state.processed:
    render_results()
else:
    render_input_section()

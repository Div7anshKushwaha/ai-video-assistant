"""
AI Video Assistant - Streamlit interface.

This file is only a presentation layer. All AI work happens in:
    main.run_pipeline()            -> audio, transcript, summary, insights, RAG chain
    core.rag_engine.ask_question() -> answers questions with the RAG chain

Run with:  streamlit run app.py
"""

import html
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads .env; secrets are never shown in the UI

from main import run_pipeline  # noqa: E402  (import after load_dotenv on purpose)
from core.rag_engine import ask_question  # noqa: E402

st.set_page_config(page_title="AI Video Assistant", page_icon="🎥", layout="wide")

LANGUAGES = {"English": "english", "Hinglish": "hinglish"}
ALLOWED_EXTENSIONS = ["mp3", "wav", "m4a", "aac", "flac", "ogg", "mp4", "mkv", "mov", "webm"]
YOUTUBE_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be", "www.youtu.be",
}

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 1100px;}
.hero-title {font-size: 2.8rem; font-weight: 700; letter-spacing: -0.02em; margin: 0;}
.hero-sub {font-size: 1.15rem; opacity: 0.65; margin: 0.25rem 0 1.75rem 0;}
.result-title {font-size: 2.1rem; font-weight: 700; line-height: 1.25; margin: 0 0 1rem 0;}
</style>
"""


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_state():
    """Create every key once, so later code can read them safely."""
    defaults = {
        "processed": False,
        "title": "",
        "transcript": "",
        "summary": "",
        "action_items": "",
        "key_decisions": "",
        "open_questions": "",
        "rag_chain": None,
        "messages": [],       # chat history: {"role", "content", "sources"}
        "source_label": "",
        "language_label": "English",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# --------------------------------------------------------------------------
# Helpers: validation, uploads, error messages
# --------------------------------------------------------------------------
def normalize_youtube_url(url: str):
    """Return a clean YouTube URL, or None if it is not a YouTube link."""
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() in YOUTUBE_HOSTS:
        return url
    return None


def save_upload(uploaded_file):
    """Save an uploaded file to a temp path and return it (None if unsupported)."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix.lstrip(".") not in ALLOWED_EXTENSIONS:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def redact(text: str) -> str:
    """Remove any secret values from text before it is shown on screen."""
    for name, value in os.environ.items():
        sensitive = any(t in name.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
        if sensitive and value and len(value) >= 8:
            text = text.replace(value, "***")
    return text


def friendly_error(error: Exception, stage: int) -> str:
    """Turn a raw exception into a message the user can act on.

    `stage` is the 0-based pipeline step that was running (see main.PIPELINE_STEPS).
    """
    msg = str(error).lower()
    if any(k in msg for k in ("api key", "unauthorized", "401", "403", "invalid_api_key")):
        return "An AI service rejected the request. Check the API keys in your .env file."
    if any(k in msg for k in ("rate limit", "429", "quota")):
        return "An AI service rate limit was reached. Wait a minute and try again."
    if stage == 0:
        return ("The audio could not be downloaded or processed. Check that the video is "
                "public and that FFmpeg is installed.")
    if stage == 1:
        return "Transcription failed. Try a shorter file or a different language setting."
    if 2 <= stage <= 6:
        return "The AI model could not generate this part of the analysis. Try again."
    return "The knowledge base could not be built from the transcript."


# --------------------------------------------------------------------------
# Running the real pipeline with real progress
# --------------------------------------------------------------------------
def run_analysis(source: str, language: str) -> dict | None:
    """Call main.run_pipeline and show progress for each real step.

    run_pipeline() calls on_progress(step_index, total_steps, label) as each
    step starts, so the status below always reflects what is actually running.
    """
    status = st.status("Starting analysis...", expanded=True)
    bar = st.progress(0.0)
    tracker = {"current": None, "index": 0}

    def on_progress(index: int, total: int, label: str):
        if tracker["current"]:
            status.write(f"✓ {tracker['current']}")
        tracker["current"], tracker["index"] = label, index
        status.update(label=f"{label}...")
        bar.progress(index / total)

    try:
        result = run_pipeline(source, language, on_progress=on_progress)
    except Exception as exc:  # noqa: BLE001 - we show a friendly message instead
        status.update(label="Analysis failed", state="error", expanded=False)
        bar.empty()
        st.error(friendly_error(exc, tracker["index"]))
        with st.expander("Technical details"):
            st.code(redact(f"{type(exc).__name__}: {exc}"))
        return None

    status.write(f"✓ {tracker['current']}")
    bar.progress(1.0)
    status.update(label="Analysis complete", state="complete", expanded=False)
    return result


def handle_analyze(url: str, uploaded_file, language_label: str):
    """Validate input, run the pipeline, and store results in session_state."""
    temp_path = None
    try:
        if uploaded_file is not None:  # an uploaded file takes priority over the URL
            temp_path = save_upload(uploaded_file)
            if temp_path is None:
                st.error(f"Unsupported file type. Use one of: {', '.join(ALLOWED_EXTENSIONS)}.")
                return
            source, label = temp_path, uploaded_file.name
        else:
            if not url.strip():
                st.error("Enter a YouTube URL or upload a file to get started.")
                return
            source = normalize_youtube_url(url)
            if source is None:
                st.error("That doesn't look like a YouTube link. Use a youtube.com or youtu.be URL.")
                return
            label = source

        result = run_analysis(source, LANGUAGES[language_label])
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    if result is None:
        return

    st.session_state.update(
        processed=True,
        title=result["title"],
        transcript=result["transcript"],
        summary=result["summary"],
        action_items=result["action_items"],
        key_decisions=result["key_decisions"],
        open_questions=result["open_questions"],
        rag_chain=result["rag_chain"],
        messages=[],
        source_label=label,
        language_label=language_label,
    )
    st.rerun()


# --------------------------------------------------------------------------
# UI sections
# --------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🎥 AI Video Assistant")
        if st.session_state.processed:
            st.caption("Current video")
            st.write(st.session_state.source_label)
            if st.button("Analyze another video", use_container_width=True):
                reset_session()
                st.rerun()
            st.divider()

        st.caption("Configuration")
        st.write(("✅" if os.getenv("GROQ_API_KEY") else "❌") + " Groq API key")
        st.write(("✅" if os.getenv("SARVAM_API_KEY") else "⚪") + " Sarvam API key (optional)")
        st.divider()
        st.caption(
            "Pipeline: audio extraction → transcription → LLM analysis → "
            "embeddings → ChromaDB → RAG chat"
        )


def render_input_section():
    _, middle, _ = st.columns([1, 4, 1])
    with middle:
        st.markdown('<p class="hero-title">🎥 AI Video Assistant</p>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub">Turn videos into searchable knowledge.</p>',
                    unsafe_allow_html=True)

        with st.container(border=True):
            url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
            with st.expander("Or upload a local audio/video file"):
                uploaded = st.file_uploader(
                    "Audio or video file",
                    type=ALLOWED_EXTENSIONS,
                    label_visibility="collapsed",
                )
                st.caption("If you upload a file, it is used instead of the URL.")

            left, right = st.columns([2, 1], vertical_alignment="bottom")
            with left:
                language_label = st.selectbox("Language", list(LANGUAGES.keys()))
            with right:
                clicked = st.button("🚀 Analyze Video", type="primary", use_container_width=True)

        if clicked:
            handle_analyze(url, uploaded, language_label)


def render_text_block(content, empty_message: str):
    """Show LLM output that may be a string or a list of strings."""
    if not content:
        st.info(empty_message)
    elif isinstance(content, (list, tuple)):
        st.markdown("\n".join(f"- {item}" for item in content))
    else:
        st.markdown(content)


def render_summary():
    with st.container(border=True):
        render_text_block(st.session_state.summary, "No summary was generated.")


def render_transcript():
    transcript = st.session_state.transcript
    words = len(transcript.split())
    top_left, top_right = st.columns([3, 1], vertical_alignment="center")
    top_left.caption(f"{words:,} words")
    top_right.download_button(
        "Download .txt", transcript, file_name="transcript.txt", use_container_width=True
    )
    with st.expander("Show full transcript", expanded=False):
        st.text_area("Transcript", transcript, height=420, label_visibility="collapsed")


def render_action_items():
    with st.container(border=True):
        render_text_block(st.session_state.action_items, "No action items were found.")


def render_decisions():
    with st.container(border=True):
        render_text_block(st.session_state.key_decisions, "No key decisions were found.")


def render_questions():
    with st.container(border=True):
        render_text_block(st.session_state.open_questions, "No open questions were found.")


def split_answer(response):
    """Accept either a plain string or a dict with answer + sources.

    ask_question() currently returns a string. If you later change it to also
    return the retrieved chunks, this function picks them up automatically.
    """
    if isinstance(response, dict):
        answer = response.get("answer") or response.get("result") or str(response)
        sources = response.get("sources") or response.get("source_documents") or []
        return str(answer), list(sources)
    return str(response), []


def render_sources(sources):
    with st.expander("📚 Retrieved context"):
        for i, doc in enumerate(sources, start=1):
            text = getattr(doc, "page_content", str(doc))
            metadata = getattr(doc, "metadata", {}) or {}
            chunk_id = metadata.get("chunk_id", i)
            st.markdown(f"**Chunk {chunk_id}**")
            st.caption(text[:500] + ("..." if len(text) > 500 else ""))


def render_chat():
    st.caption("Ask anything about this video. Answers come from its transcript.")

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask a question about the video...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the transcript..."):
                response = ask_question(st.session_state.rag_chain, question)
            answer, sources = split_answer(response)
        except Exception as exc:  # noqa: BLE001
            answer, sources = "I couldn't answer that. Please try again or rephrase the question.", []
            st.error(answer)
            with st.expander("Technical details"):
                st.code(redact(f"{type(exc).__name__}: {exc}"))
        else:
            st.markdown(answer)
            if sources:
                render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


def render_results():
    st.markdown(
        f'<p class="result-title">{html.escape(st.session_state.title or "Untitled video")}</p>',
        unsafe_allow_html=True,
    )

    words = len(st.session_state.transcript.split())
    c1, c2, c3 = st.columns(3)
    c1.metric("Transcript length", f"{words:,} words")
    c2.metric("Language", st.session_state.language_label)
    c3.metric("Chat questions", sum(m["role"] == "user" for m in st.session_state.messages))

    tabs = st.tabs([
        "📋 Summary", "📝 Transcript", "✅ Action Items",
        "🔑 Decisions", "❓ Questions", "💬 Chat",
    ])
    renderers = [
        render_summary, render_transcript, render_action_items,
        render_decisions, render_questions, render_chat,
    ]
    for tab, render in zip(tabs, renderers):
        with tab:
            render()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    render_sidebar()
    if st.session_state.processed:
        render_results()
    else:
        render_input_section()


main()
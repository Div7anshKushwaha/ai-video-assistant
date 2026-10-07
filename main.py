from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import ask_question, build_rag_chain
from core.summarizer import generate_title, summarize
from core.transcriber import transcribe_all
from utils.audio_processor import process_input


load_dotenv()

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


PIPELINE_STEPS = [
    "Audio processing",
    "Transcription",
    "Generating title",
    "Generating summary",
    "Extracting action items",
    "Extracting key decisions",
    "Extracting open questions",
    "Building knowledge base",
]

SUPPORTED_LANGUAGES = {"english", "hinglish"}
ProgressCallback = Callable[[int, int, str], None]


def _validate_source(source: str) -> str:
    """Validate and normalize a YouTube URL or local file path."""

    normalized = source.strip()

    if not normalized:
        raise ValueError(
            "Input cannot be empty. Provide a YouTube URL or local media file."
        )

    if normalized.startswith(("http://", "https://")):
        return normalized

    path = Path(normalized).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    return str(path)


def _validate_language(language: str) -> str:
    """Validate the transcription language mode."""

    normalized = language.strip().lower() or "english"

    if normalized not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(
            f"Unsupported language '{language}'. Use one of: {supported}."
        )

    return normalized


def run_pipeline(
    source: str,
    language: str = "english",
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the complete video-to-knowledge pipeline.

    Parameters
    ----------
    source:
        A public YouTube URL or a path to a local audio/video file.
    language:
        Either ``english`` for local Whisper transcription or ``hinglish``
        for Sarvam speech-to-text translation.
    on_progress:
        Optional callback receiving ``(step_index, total_steps, label)``.

    Returns
    -------
    dict[str, Any]
        Title, transcript, meeting insights, and the RAG chain.

    Notes
    -----
    ``transcribe_all`` removes generated audio chunk files in its ``finally``
    block. ``process_input`` removes its intermediate WAV file after chunking.
    """

    source = _validate_source(source)
    language = _validate_language(language)

    def notify(step_index: int) -> None:
        label = PIPELINE_STEPS[step_index]

        LOGGER.info("Step %d/%d: %s", step_index + 1, len(PIPELINE_STEPS), label)

        if on_progress is not None:
            on_progress(step_index, len(PIPELINE_STEPS), label)

    LOGGER.info("Starting AI Video Assistant pipeline")
    LOGGER.info("Source: %s", source)
    LOGGER.info("Language: %s", language)

    # ------------------------------------------------------------------
    # Step 1: Audio processing
    # ------------------------------------------------------------------
    notify(0)
    chunks = process_input(source)

    if not chunks:
        raise RuntimeError("Audio processing produced no chunks.")

    LOGGER.info("Created %d audio chunk(s)", len(chunks))

    # ------------------------------------------------------------------
    # Step 2: Transcription
    # ------------------------------------------------------------------
    notify(1)
    transcript = transcribe_all(chunks, language=language)

    if not transcript.strip():
        raise RuntimeError(
            "Transcription completed but returned an empty transcript."
        )

    LOGGER.info("Transcript generated: %d characters", len(transcript))

    # ------------------------------------------------------------------
    # Steps 3–7: Transcript analysis
    # ------------------------------------------------------------------
    notify(2)
    title = generate_title(transcript)

    notify(3)
    summary = summarize(transcript)

    notify(4)
    action_items = extract_action_items(transcript)

    notify(5)
    decisions = extract_key_decisions(transcript)

    notify(6)
    questions = extract_questions(transcript)

    # ------------------------------------------------------------------
    # Step 8: RAG knowledge base
    # ------------------------------------------------------------------
    notify(7)
    rag_chain = build_rag_chain(transcript)

    LOGGER.info("AI Video Assistant pipeline completed successfully")

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def _print_results(result: dict[str, Any]) -> None:
    """Print analysis results in the CLI."""

    print("\n" + "=" * 60)
    print(f"📌 Title:\n{result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)


def _run_cli() -> None:
    """Run the interactive command-line interface."""

    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish) [english]: ").strip() or "english"

    try:
        result = run_pipeline(source, language)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled.")
        return
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"\n❌ Pipeline failed: {type(exc).__name__}: {exc}")
        return

    _print_results(result)

    print("\n💬 Chat with the processed video (type 'exit' to quit)\n")

    rag_chain = result["rag_chain"]

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if question.lower() in {"exit", "quit", "q"}:
            print("\n👋 Goodbye!")
            break

        if not question:
            continue

        try:
            answer = ask_question(rag_chain, question)
            print(f"\n🤖 Assistant: {answer}\n")
        except Exception as exc:  # noqa: BLE001 - keep the CLI alive
            print(f"\n⚠️ Could not answer the question: {exc}\n")


if __name__ == "__main__":
    _run_cli()

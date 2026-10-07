from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()


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


def run_pipeline(
    source: str,
    language: str = "english",
    on_progress=None,
) -> dict:

    def notify(step_index: int):
        if on_progress:
            on_progress(
                step_index,
                len(PIPELINE_STEPS),
                PIPELINE_STEPS[step_index],
            )

    print("\nStarting AI Video Assistant...\n")

    # =========================================================
    # STEP 1 — AUDIO PROCESSING
    # =========================================================
    notify(0)

    print("===== STEP 1: AUDIO PROCESSING =====")

    chunks = process_input(source)

    print(f"Created {len(chunks)} audio chunk(s).")

    # =========================================================
    # STEP 2 — TRANSCRIPTION
    # =========================================================
    notify(1)

    print("\n===== STEP 2: TRANSCRIPTION =====")

    transcript = transcribe_all(chunks, language)

    print(
        f"\nRaw transcription (first 300 characters):\n"
        f"{transcript[:300]}"
    )

    # =========================================================
    # STEP 3 — TITLE
    # =========================================================
    notify(2)

    print("\n===== STEP 3: TITLE =====")

    title = generate_title(transcript)

    # =========================================================
    # STEP 4 — SUMMARY
    # =========================================================
    notify(3)

    print("\n===== STEP 4: SUMMARY =====")

    summary = summarize(transcript)

    # =========================================================
    # STEP 5 — ACTION ITEMS
    # =========================================================
    notify(4)

    print("\n===== STEP 5: ACTION ITEMS =====")

    action_items = extract_action_items(transcript)

    # =========================================================
    # STEP 6 — KEY DECISIONS
    # =========================================================
    notify(5)

    print("\n===== STEP 6: KEY DECISIONS =====")

    decisions = extract_key_decisions(transcript)

    # =========================================================
    # STEP 7 — OPEN QUESTIONS
    # =========================================================
    notify(6)

    print("\n===== STEP 7: OPEN QUESTIONS =====")

    questions = extract_questions(transcript)

    # =========================================================
    # STEP 8 — RAG
    # =========================================================
    notify(7)

    print("\n===== STEP 8: BUILDING RAG =====")

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":

    # =========================================================
    # CLI ENTRY POINT
    # =========================================================

    source = input(
        "Enter YouTube URL or local file path: "
    ).strip()

    language = (
        input("Language (english/hinglish): ")
        .strip()
        .lower()
        or "english"
    )

    try:

        result = run_pipeline(
            source,
            language,
        )

        # =====================================================
        # DISPLAY RESULTS
        # =====================================================

        print("\n" + "=" * 60)

        print(
            f"📌 Title:\n"
            f"{result['title']}"
        )

        print(
            f"\n📋 Summary:\n"
            f"{result['summary']}"
        )

        print(
            f"\n✅ Action Items:\n"
            f"{result['action_items']}"
        )

        print(
            f"\n🔑 Key Decisions:\n"
            f"{result['key_decisions']}"
        )

        print(
            f"\n❓ Open Questions:\n"
            f"{result['open_questions']}"
        )

        print("=" * 60)

        # =====================================================
        # CHAT WITH MEETING
        # =====================================================

        print(
            "\n💬 Chat with your meeting "
            "(type 'exit' to quit)\n"
        )

        rag_chain = result["rag_chain"]

        while True:

            question = input("You: ").strip()

            if question.lower() in [
                "exit",
                "quit",
                "q",
            ]:
                print("\n👋 Goodbye!")
                break

            if not question:
                continue

            answer = ask_question(
                rag_chain,
                question,
            )

            print(
                f"\n🤖 Assistant: {answer}\n"
            )

    except Exception as e:

        print(
            f"\n❌ Pipeline failed:\n"
            f"{e}"
        )
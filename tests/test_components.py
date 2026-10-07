from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)


# Sample transcript for testing
transcript = """
Today we discussed the development of our AI Video Assistant.

The team agreed to use Whisper for local English transcription
and Sarvam AI for Hindi and Hinglish transcription.

Rahul will prepare the sample dataset by Friday.
Priya will build the Streamlit user interface by Monday.

The team decided to use Groq as the LLM provider
and ChromaDB as the vector database for the RAG system.

We also discussed deploying the application using Streamlit Cloud.

One unresolved question is whether the free deployment
will be sufficient for processing long videos.

Another question is whether we should add speaker diarization
in a future version.

The next meeting will be held next Wednesday.
"""


print("\n" + "=" * 60)
print("TESTING AI VIDEO ASSISTANT COMPONENTS")
print("=" * 60)


# --------------------------------------------------
# 1. TEST TITLE GENERATION
# --------------------------------------------------

print("\n===== 1. GENERATING TITLE =====")

try:
    title = generate_title(transcript)

    print("Title:")
    print(title)

    print("✅ Title generation successful.")

except Exception as e:
    print("❌ Title generation failed.")
    print("Error:", e)


# --------------------------------------------------
# 2. TEST SUMMARIZATION
# --------------------------------------------------

print("\n===== 2. GENERATING SUMMARY =====")

try:
    summary = summarize(transcript)

    print("Summary:")
    print(summary)

    print("✅ Summarization successful.")

except Exception as e:
    print("❌ Summarization failed.")
    print("Error:", e)


# --------------------------------------------------
# 3. TEST ACTION ITEMS
# --------------------------------------------------

print("\n===== 3. EXTRACTING ACTION ITEMS =====")

try:
    action_items = extract_action_items(transcript)

    print("Action Items:")
    print(action_items)

    print("✅ Action item extraction successful.")

except Exception as e:
    print("❌ Action item extraction failed.")
    print("Error:", e)


# --------------------------------------------------
# 4. TEST KEY DECISIONS
# --------------------------------------------------

print("\n===== 4. EXTRACTING KEY DECISIONS =====")

try:
    decisions = extract_key_decisions(transcript)

    print("Key Decisions:")
    print(decisions)

    print("✅ Decision extraction successful.")

except Exception as e:
    print("❌ Decision extraction failed.")
    print("Error:", e)


# --------------------------------------------------
# 5. TEST OPEN QUESTIONS
# --------------------------------------------------

print("\n===== 5. EXTRACTING OPEN QUESTIONS =====")

try:
    questions = extract_questions(transcript)

    print("Open Questions:")
    print(questions)

    print("✅ Question extraction successful.")

except Exception as e:
    print("❌ Question extraction failed.")
    print("Error:", e)


print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
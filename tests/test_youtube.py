from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)

youtube_url = input("Enter YouTube URL: ")

# 1. Download + process audio
print("\n===== STEP 1: AUDIO PROCESSING =====")

chunks = process_input(youtube_url)

print(f"Created {len(chunks)} audio chunks.")

# 2. Transcription
print("\n===== STEP 2: TRANSCRIPTION =====")

language = input("Enter language (english/hinglish): ").strip().lower()

transcript = transcribe_all(
    chunks,
    language=language
)

print("\nTranscript:")
print(transcript[:3000])

# 3. Generate title
print("\n===== STEP 3: TITLE =====")

title = generate_title(transcript)

print("Title:")
print(title)

# 4. Generate summary
print("\n===== STEP 4: SUMMARY =====")

summary = summarize(transcript)

print("Summary:")
print(summary)

# 5. Action items
print("\n===== STEP 5: ACTION ITEMS =====")

action_items = extract_action_items(transcript)

print(action_items)

# 6. Decisions
print("\n===== STEP 6: KEY DECISIONS =====")

decisions = extract_key_decisions(transcript)

print(decisions)

# 7. Questions
print("\n===== STEP 7: OPEN QUESTIONS =====")

questions = extract_questions(transcript)

print(questions)

print("\n===== TEST COMPLETE =====")
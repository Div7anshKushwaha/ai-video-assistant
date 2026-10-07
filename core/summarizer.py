from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """Create and return the Groq LLM."""

    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list[str]:
    """Split a long transcript into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """Generate a complete meeting summary."""

    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of a meeting transcript "
                "concisely. Preserve the important information, "
                "decisions, and context.",
            ),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = []

    for chunk in chunks:
        summary = map_chain.invoke({"text": chunk})
        chunk_summaries.append(summary)

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. "
                "Combine the partial summaries into one "
                "professional meeting summary. "
                "Use clear bullet points and preserve important "
                "decisions, discussions, and conclusions.",
            ),
            ("human", "{text}"),
        ]
    )

    combined_chain = combined_prompt | llm | StrOutputParser()

    return combined_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    """Generate a short professional title for the meeting."""

    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the meeting transcript, generate a "
                "short professional meeting title. "
                "Maximum 8 words. "
                "Return only the title and nothing else.",
            ),
            ("human", "{text}"),
        ]
    )

    title_chain = title_prompt | llm | StrOutputParser()

    return title_chain.invoke(
        {"text": transcript[:2000]}
    )
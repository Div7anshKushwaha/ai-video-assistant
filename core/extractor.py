from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """Create and return the Groq LLM."""

    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    """Build a reusable LangChain chain."""

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )

    return prompt | llm | StrOutputParser()


def extract_action_items(transcript: str) -> str:
    """Extract tasks, owners, and deadlines from the transcript."""

    chain = build_chain(
        "You are an expert meeting analyst. "
        "From the meeting transcript, extract all action items. "
        "For each action item provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, otherwise write 'Not specified')\n\n"
        "Format the result as a numbered list. "
        "If no action items are found, say "
        "'No action items found.'"
    )

    return chain.invoke({"text": transcript})


def extract_key_decisions(transcript: str) -> str:
    """Extract important decisions made during the meeting."""

    chain = build_chain(
        "You are an expert meeting analyst. "
        "From the meeting transcript, extract all key decisions made. "
        "Format the result as a numbered list. "
        "Only include actual decisions, not general discussions. "
        "If no decisions are found, say "
        "'No key decisions found.'"
    )

    return chain.invoke({"text": transcript})


def extract_questions(transcript: str) -> str:
    """Extract unresolved questions and follow-up topics."""

    chain = build_chain(
        "You are an expert meeting analyst. "
        "From the meeting transcript, extract all unresolved questions "
        "or topics requiring follow-up. "
        "Format the result as a numbered list. "
        "If no open questions are found, say "
        "'No open questions found.'"
    )

    return chain.invoke({"text": transcript})
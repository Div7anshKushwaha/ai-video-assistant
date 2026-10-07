import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)

load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def create_prompt():

    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Answer the user's question ONLY using the meeting transcript
context provided below.

If the answer is not found in the context, say:

"I could not find this information in the meeting transcript."

Do not make up information.

Always be concise and precise.

Context from meeting transcript:
{context}
"""
        ),
        ("human", "{question}"),
    ])


def build_rag_chain(transcript: str):

    print("Building RAG chain...")

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(
        vector_store,
        k=4
    )

    llm = get_llm()

    prompt = create_prompt()

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():

    print("Loading existing vector store...")

    vector_store = load_vector_store()

    retriever = get_retriever(
        vector_store,
        k=4
    )

    llm = get_llm()

    prompt = create_prompt()

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(
    rag_chain,
    question: str
) -> str:

    print(f"\nQuestion: {question}")

    answer = rag_chain.invoke(question)

    print(f"\nAnswer: {answer}")

    return answer
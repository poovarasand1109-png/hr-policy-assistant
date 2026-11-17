import os
from pathlib import Path
from functools import lru_cache

import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.5-flash"


def _api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    key = key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Configure .streamlit/secrets.toml locally "
            "or App settings → Secrets on Streamlit Cloud."
        )
    return key


def get_policy_files():
    DATA_DIR.mkdir(exist_ok=True)
    return sorted(p.name for p in DATA_DIR.glob("*.pdf"))


def _read_pdf_pages():
    pages = []
    for path in sorted(DATA_DIR.glob("*.pdf")):
        pdf = fitz.open(path)
        try:
            for page_no, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({
                        "text": text,
                        "source": path.name,
                        "page": page_no
                    })
        finally:
            pdf.close()
    if not pages:
        raise RuntimeError("No readable PDF text found in data/.")
    return pages


@lru_cache(maxsize=1)
def _vectorstore():
    pages = _read_pdf_pages()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\\n\\n", "\\n", " ", ""]
    )
    texts, metadata = [], []
    for page in pages:
        for chunk in splitter.split_text(page["text"]):
            if chunk.strip():
                texts.append(chunk)
                metadata.append({
                    "source": page["source"],
                    "page": page["page"]
                })

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.from_texts(texts, embeddings, metadatas=metadata)


@lru_cache(maxsize=1)
def _llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=_api_key(),
        temperature=0
    )


def answer_question(question: str):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    matches = _vectorstore().similarity_search_with_score(question, k=TOP_K)
    if not matches:
        return {
            "answer": "I could not find this information in the available HR policy documents.",
            "sources": []
        }

    context_items, sources = [], []
    for doc, score in matches:
        filename = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page")
        citation = f"{filename}, page {page}" if page else filename
        context_items.append(f"[Source: {citation}]\\n{doc.page_content}")
        if citation not in sources:
            sources.append(citation)

    system_text = (
        "You are the Zyro Dynamics HR Policy Assistant. Answer using ONLY the "
        "retrieved policy context. Do not use outside knowledge or invent details. "
        "Preserve conditions, timelines, exceptions and eligibility requirements. "
        "If the context does not support an answer, say: "
        "'I could not find this information in the available HR policy documents.' "
        "If only part is supported, answer that part and identify what is not specified. "
        "Be concise and mention the policy source and page when available."
    )
    user_text = (
        f"Employee question:\\n{question}\\n\\n"
        f"Retrieved policy context:\\n{'\\n\\n---\\n\\n'.join(context_items)}"
    )
    response = _llm().invoke([
        SystemMessage(content=system_text),
        HumanMessage(content=user_text)
    ])
    answer = response.content
    if isinstance(answer, list):
        answer = "\\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in answer
        )
    return {"answer": str(answer), "sources": sources}

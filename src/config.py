"""
config.py
---------
Loads environment variables and initialises all shared clients.
Import from here instead of re-creating clients in every module.
"""

import os
from dotenv import load_dotenv

result = load_dotenv()
print(f"✅ .env loaded: {result}")

# ── LM Studio connection ────────────────────────────────────────────────────
LM_STUDIO_BASE_URL: str = os.getenv("LM_STUDIO_BASE_URL", "http://192.168.1.42:1234/v1")
TEXT_MODEL: str         = os.getenv("TEXT_MODEL",         "qwen/qwen3.5-9b")
VISION_MODEL: str       = os.getenv("VISION_MODEL",       "qwen/qwen3.5-9b")

print(f"   LM_STUDIO_BASE_URL = {LM_STUDIO_BASE_URL}")
print(f"   TEXT_MODEL         = {TEXT_MODEL}")
print(f"   VISION_MODEL       = {VISION_MODEL}")

# ── LLM clients ────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=TEXT_MODEL,
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio",
    temperature=0,
)

llm_generator = ChatOpenAI(
    model=TEXT_MODEL,
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio",
    temperature=0.1,
)

vlm = ChatOpenAI(
    model=VISION_MODEL,
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio",
    temperature=0,
)

# ── Embeddings ──────────────────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# ── Vector store ────────────────────────────────────────────────────────────
from langchain_chroma import Chroma

vectorstore = Chroma(
    collection_name="rag_knowledge_base",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# ── Text splitter ───────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

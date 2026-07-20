import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
# CONFIGURATION VARIABLES
# ============================================================
MAX_HISTORY_MESSAGES = 5      # Sliding window size for LLM context
RETRIEVAL_K = 5              # Number of chunks to retrieve
CHUNK_SIZE = 750              # Characters per chunk
CHUNK_OVERLAP = 75            # Overlap between chunks
OCR_THRESHOLD = 75            # Min chars before forcing VLM OCR

# ============================================================
# SERVER ENDPOINTS & MODELS
# ============================================================
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://192.168.1.42:1234/v1")
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen/qwen3.5-9b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.5-9b")

# ============================================================
# MODEL CLIENTS
# ============================================================
llm = ChatOpenAI(model=TEXT_MODEL, base_url=LM_STUDIO_BASE_URL, api_key="lm-studio", temperature=0)
llm_generator = ChatOpenAI(model=TEXT_MODEL, base_url=LM_STUDIO_BASE_URL, api_key="lm-studio", temperature=0.1)
vlm = ChatOpenAI(model=VISION_MODEL, base_url=LM_STUDIO_BASE_URL, api_key="lm-studio", temperature=0)

# ============================================================
# EMBEDDINGS & VECTORSTORE
# ============================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

vectorstore = Chroma(
    collection_name="rag_knowledge_base",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# ============================================================
# TOOLS & UTILITIES
# ============================================================
web_search = DuckDuckGoSearchRun()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
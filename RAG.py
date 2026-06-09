# =============================================================================
# 0. ENVIRONMENT SETUP — MUST BE BEFORE ALL LANGCHAIN/LANGGRAPH IMPORTS
# =============================================================================
from dotenv import load_dotenv
load_dotenv()

import os
import json
from typing import Literal, Annotated, TypedDict, Sequence

# Fail fast if LangSmith is not configured
if os.getenv("LANGCHAIN_TRACING_V2") != "true":
    raise EnvironmentError("❌ Set LANGCHAIN_TRACING_V2=true in .env")
if not os.getenv("LANGSMITH_API_KEY"): 
    raise EnvironmentError("❌ Set LANGSMITH_API_KEY in .env")

# =============================================================================
# 1. IMPORTS
# =============================================================================
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =============================================================================
# 2. COMPONENT INITIALIZATION
# =============================================================================

# Structured LLM for routing/grading (JSON mode)
llm = ChatOllama(model="qwen3:8b", temperature=0, format="json")

# Natural language LLM for generation
llm_generator = ChatOllama(model="qwen3:8b", temperature=0.1)

# Embeddings + Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # Change to "cuda" for GPU
)
vectorstore = Chroma(
    collection_name="rag_knowledge_base",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Web Search
web_search = DuckDuckGoSearchRun()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# =============================================================================
# 3. GRAPH STATE
# =============================================================================

class GraphState(TypedDict):
    question: str
    generation: str
    documents: list[Document]
    messages: Annotated[Sequence, add_messages]
    route: str
    relevance: str
    grounded: str
    retry_count: int

# =============================================================================
# 4. NODES (each tagged for LangSmith filtering)
# =============================================================================

def analyze_query(state: GraphState) -> dict:
    """Route question to vectorstore, web search, or direct answer."""
    print("🔍 [Node] Analyzing query...")
    prompt = f"""Analyze this question and decide the best source.
Question: {state['question']}

Respond in JSON: {{"route": "vectorstore"}} OR {{"route": "web_search"}} OR {{"route": "direct"}}
- "vectorstore": domain-specific knowledge likely in our DB
- "web_search": current events, real-time data, obscure topics
- "direct": general knowledge/greetings needing no retrieval"""

    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"tags": ["routing", "query-analysis"]},
    )
    try:
        route = json.loads(response.content)["route"]
    except Exception:
        route = "vectorstore"

    print(f"   → Routed to: {route}")
    return {"route": route}


def retrieve_from_vectorstore(state: GraphState) -> dict:
    """Retrieve documents from ChromaDB."""
    print("📚 [Node] Retrieving from ChromaDB...")
    docs = vectorstore.similarity_search(state["question"], k=4)
    print(f"   → Retrieved {len(docs)} documents")
    return {"documents": docs}


def search_web(state: GraphState) -> dict:
    """Search DuckDuckGo and wrap results as Documents."""
    print("🌐 [Node] Searching DuckDuckGo...")
    raw_results = web_search.run(state["question"])
    docs = [Document(page_content=raw_results, metadata={"source": "duckduckgo"})]
    print("   → Got web search results")
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
    """Check if retrieved documents are relevant."""
    print("⚖️ [Node] Grading document relevance...")
    doc_text = "\n".join([d.page_content[:300] for d in state.get("documents", [])])
    prompt = f"""Are these documents relevant to answering the question?
Question: {state['question']}
Documents: {doc_text}

Respond in JSON: {{"relevance": "yes"}} or {{"relevance": "no"}}"""

    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"tags": ["grading", "relevance-check"]},
    )
    try:
        relevance = json.loads(response.content)["relevance"]
    except Exception:
        relevance = "yes"

    print(f"   → Relevance: {relevance}")
    return {"relevance": relevance}


def generate_answer(state: GraphState) -> dict:
    """Generate grounded answer using Qwen3:8B."""
    print("✍️ [Node] Generating answer with Qwen3:8B...")
    context = "\n\n".join([d.page_content for d in state.get("documents", [])])
    system_msg = SystemMessage(
        content="You are a helpful RAG assistant. Answer using ONLY the provided context. "
                "If context is insufficient, say so. Never fabricate information."
    )
    user_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}")

    response = llm_generator.invoke(
        [system_msg, user_msg],
        config={"tags": ["generation", "qwen3-8b", "grounded-answer"]},
    )

    print(f"   → Generated response ({len(response.content)} chars)")
    return {
        "generation": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def check_hallucination(state: GraphState) -> dict:
    """Verify answer is grounded in documents."""
    print("🛡️ [Node] Checking hallucination...")
    context = "\n".join([d.page_content[:300] for d in state.get("documents", [])])
    prompt = f"""Is this answer fully supported by the context?
Context: {context}
Answer: {state['generation']}

Respond in JSON: {{"grounded": "yes"}} or {{"grounded": "no"}}"""

    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"tags": ["grading", "hallucination-check"]},
    )
    try:
        grounded = json.loads(response.content)["grounded"]
    except Exception:
        grounded = "yes"

    retry = state.get("retry_count", 0) + 1
    print(f"   → Grounded: {grounded} (attempt {retry})")
    return {"grounded": grounded, "retry_count": retry}


def generate_direct(state: GraphState) -> dict:
    """Direct LLM response without retrieval."""
    print("💬 [Node] Direct LLM response (no retrieval)...")
    response = llm_generator.invoke(
        [HumanMessage(content=state["question"])],
        config={"tags": ["generation", "direct-answer"]},
    )
    return {
        "generation": response.content,
        "messages": [AIMessage(content=response.content)],
        "documents": [],
        "grounded": "yes",
    }

# =============================================================================
# 5. CONDITIONAL EDGES
# =============================================================================

def route_after_analysis(state: GraphState) -> Literal["retrieve", "search_web", "generate_direct"]:
    if state["route"] == "web_search":
        return "search_web"
    if state["route"] == "direct":
        return "generate_direct"
    return "retrieve"


def route_after_grading(state: GraphState) -> Literal["generate", "search_web", "end_no_answer"]:
    if state["relevance"] == "yes":
        return "generate"
    if state.get("retry_count", 0) < 1:
        return "search_web"
    return "end_no_answer"


def route_after_hallucination_check(state: GraphState) -> Literal["end", "generate", "search_web"]:
    if state["grounded"] == "yes":
        return "end"
    if state["retry_count"] >= 3:
        return "end"
    if state["route"] == "vectorstore":
        return "search_web"
    return "generate"

# =============================================================================
# 6. BUILD GRAPH
# =============================================================================

def build_rag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("analyze", analyze_query)
    workflow.add_node("retrieve", retrieve_from_vectorstore)
    workflow.add_node("search_web", search_web)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("hallucination_check", check_hallucination)
    workflow.add_node("generate_direct", generate_direct)

    workflow.add_edge(START, "analyze")
    workflow.add_conditional_edges("analyze", route_after_analysis)
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", route_after_grading)
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges("hallucination_check", route_after_hallucination_check)
    workflow.add_edge("generate_direct", END)
    workflow.add_edge("end_no_answer", END)

    return workflow.compile()

# =============================================================================
# 7. DATA INGESTION HELPER
# =============================================================================

def ingest_texts(texts: list[str]):
    """Ingest raw texts into ChromaDB."""
    chunks = text_splitter.create_documents(texts)
    vectorstore.add_documents(chunks)
    print(f"✅ Ingested {len(chunks)} chunks into ChromaDB")

# =============================================================================
# 8. MAIN EXECUTION WITH LANGSMITH METADATA
# =============================================================================

if __name__ == "__main__":
    # Seed vector store
    sample_docs = [
        "LangGraph is a library for building stateful, multi-agent applications with LLMs. "
        "It extends LangChain with cyclic graph support and built-in persistence.",
        "Qwen3 is a family of large language models developed by Alibaba Cloud. "
        "The 8B parameter variant offers strong performance for local deployment via Ollama.",
        "ChromaDB is an open-source embedding database designed for AI applications. "
        "It supports persistent storage and multiple embedding functions.",
        "DuckDuckGo is a privacy-focused search engine that provides an API "
        "for programmatic web searches without requiring authentication keys.",
    ]
    ingest_texts(sample_docs)

    # Build graph
    app = build_rag_graph()

    # Test queries covering all routes
    test_questions = [
        "What is LangGraph and how does it differ from LangChain?",   # vectorstore
        "What is the latest news about Qwen models in June 2026?",    # web_search
        "Tell me about ChromaDB's architecture",                      # vectorstore
        "Hello, how are you?",                                        # direct
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ Question: {q}")
        print("=" * 60)

        result = app.invoke(
            {
                "question": q,
                "messages": [HumanMessage(content=q)],
                "documents": [],
                "generation": "",
                "route": "",
                "relevance": "",
                "grounded": "",
                "retry_count": 0,
            },
            config={
                "metadata": {
                    "query_type": "adaptive_rag",
                    "llm": "qwen3:8b",
                    "vectorstore": "chromadb",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "search_backend": "duckduckgo",
                    "langsmith_project": os.getenv("LANGCHAIN_PROJECT", "default"),
                }
            },
        )

        print(f"\n✅ Final Answer:\n{result['generation']}\n")
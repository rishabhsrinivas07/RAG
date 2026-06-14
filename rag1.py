from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
from pathlib import Path
from typing import Literal, Annotated, TypedDict, Sequence

assert os.getenv("LANGCHAIN_TRACING_V2") == "true", "❌ Set LANGCHAIN_TRACING_V2=true in .env"
assert os.getenv("LANGSMITH_API_KEY"), "❌ Set LANGSMITH_API_KEY in .env"

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

llm = ChatOllama(model="qwen3:8b", temperature=0, format="json")
llm_generator = ChatOllama(model="qwen3:8b", temperature=0.1)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

vectorstore = Chroma(
    collection_name="rag_knowledge_base",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

web_search = DuckDuckGoSearchRun()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


class GraphState(TypedDict):
    question: str
    generation: str
    documents: list[Document]
    messages: Annotated[Sequence, add_messages]
    route: str
    relevance: str
    grounded: str
    retry_count: int


def analyze_query(state: GraphState) -> dict:
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
    print("📚 [Node] Retrieving from ChromaDB...")
    docs = vectorstore.similarity_search(state["question"], k=4)
    print(f"   → Retrieved {len(docs)} documents")
    return {"documents": docs}


def search_web(state: GraphState) -> dict:
    print("🌐 [Node] Searching DuckDuckGo...")
    raw_results = web_search.run(state["question"])
    docs = [Document(page_content=raw_results, metadata={"source": "duckduckgo"})]
    print("   → Got web search results")
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
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


def ingest_texts(texts: list[str]):
    chunks = text_splitter.create_documents(texts)
    vectorstore.add_documents(chunks)
    print(f"✅ Ingested {len(chunks)} text chunks into ChromaDB")


def ingest_pdfs(pdf_paths: list[str]):
    all_chunks = []
    for pdf_path in pdf_paths:
        path = Path(pdf_path)
        if not path.exists():
            print(f"⚠️ Skipping missing file: {pdf_path}")
            continue

        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={"source": path.name, "page": i + 1}
                ))

        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"   📄 {path.name}: {len(reader.pages)} pages → {len(chunks)} chunks")

    if all_chunks:
        vectorstore.add_documents(all_chunks)
        print(f"✅ Ingested {len(all_chunks)} total PDF chunks into ChromaDB")
    else:
        print("⚠️ No extractable text found in provided PDFs")


def ingest_urls(urls: list[str], delay: float = 2.0):
    all_chunks = []
    for url in urls:
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                header_template={
                    "User-Agent": "Mozilla/5.0 (RAG-Bot/1.0; +https://yourdomain.com)"
                }
            )
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = url

            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"   🌐 {url}: {len(docs)} pages → {len(chunks)} chunks")

            time.sleep(delay)

        except Exception as e:
            print(f"⚠️ Failed to scrape {url}: {e}")

    if all_chunks:
        vectorstore.add_documents(all_chunks)
        print(f"✅ Ingested {len(all_chunks)} total web article chunks into ChromaDB")
    else:
        print("⚠️ No extractable content from provided URLs")


if __name__ == "__main__":
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

    app = build_rag_graph()

    test_questions = [
        "What is LangGraph and how does it differ from LangChain?",
        "What is the latest news about Qwen models in June 2026?",
        "Tell me about ChromaDB's architecture",
        "Hello, how are you?",
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
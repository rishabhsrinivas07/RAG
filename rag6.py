from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import re
import base64
from io import BytesIO
from pathlib import Path
from typing import Literal, Annotated, TypedDict, Sequence

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")

VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8001/v1")
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")

llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=VLLM_BASE_URL,
    api_key="EMPTY",
    temperature=0,
)

llm_generator = ChatOpenAI(
    model=MODEL_NAME,
    base_url=VLLM_BASE_URL,
    api_key="EMPTY",
    temperature=0.1,
)

vlm = ChatOpenAI(
    model=VLM_MODEL_NAME,
    base_url=VLM_BASE_URL,
    api_key="EMPTY",
    temperature=0,
)

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
    print("✍️ [Node] Generating answer...")
    context = "\n\n".join([d.page_content for d in state.get("documents", [])])
    system_msg = SystemMessage(
        content="You are a helpful RAG assistant. Answer using ONLY the provided context. "
                "If context is insufficient, say so. Never fabricate information."
    )
    user_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}")

    response = llm_generator.invoke(
        [system_msg, user_msg],
        config={"tags": ["generation", "grounded-answer"]},
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

def end_no_answer(state: GraphState) -> dict:
    print("🛑 [Node] No relevant information found after retries.")
    fallback_message = "I'm sorry, but I couldn't find any relevant information to answer your question after searching both the knowledge base and the web."
    return {
        "generation": fallback_message,
        "messages": [AIMessage(content=fallback_message)],
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

def route_after_hallucination_check(state: GraphState) -> Literal["generate", "search_web", "__end__"]:
    if state["grounded"] == "yes":
        return END
    if state["retry_count"] >= 3:
        return END
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
    workflow.add_node("end_no_answer", end_no_answer)

    workflow.add_edge(START, "analyze")
    workflow.add_conditional_edges("analyze", route_after_analysis)
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("search_web", "grade")
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

def detect_table_or_chart(text: str, page) -> bool:
    if hasattr(page, 'images') and len(page.images) > 0:
        return True
    
    words = text.split()
    if not words:
        return False
        
    num_count = sum(1 for w in words if re.match(r'^[\d\.\,\$\%\-\+]+$', w))
    num_ratio = num_count / len(words)
    
    if num_ratio > 0.2 and len(words) > 15:
        return True
        
    short_word_count = sum(1 for w in words if len(w) <= 3)
    short_ratio = short_word_count / len(words)
    if short_ratio > 0.6 and len(words) > 20:
        return True
        
    return False

def vlm_extract_text(image_source, source_name="image"):
    if isinstance(image_source, (str, Path)):
        with open(image_source, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image_source, Image.Image):
        buffered = BytesIO()
        image_source.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
    else:
        return ""

    image_data = base64.b64encode(image_bytes).decode("utf-8")
    
    prompt = (
        "Transcribe all text, tables, and chart descriptions from this image. "
        "Maintain the original structure as much as possible using Markdown for tables. "
        "Do not include any conversational filler, just the extracted content."
    )
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ]
    )
    
    try:
        response = vlm.invoke([message])
        return response.content
    except Exception as e:
        print(f"⚠️ VLM extraction failed for {source_name}: {e}")
        return ""

def ingest_folder(root_folder: str, ocr_threshold: int = 50):
    root_path = Path(root_folder).resolve()
    if not root_path.exists() or not root_path.is_dir():
        print(f"⚠️ Folder not found or not a directory: {root_folder}")
        return

    pdf_files = list(root_path.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ No PDF files found in {root_folder} or its subfolders.")
        return

    print(f"📂 Found {len(pdf_files)} PDF files in {root_folder}")
    all_chunks = []

    for pdf_path in pdf_files:
        rel_dir = pdf_path.resolve().parent.relative_to(root_path)
        folder_name = str(rel_dir) if str(rel_dir) != "." else root_path.name

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as e:
            print(f"⚠️ Failed to read {pdf_path.name}: {e}")
            continue
            
        pages = []
        ocr_count = 0
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            extraction_method = "text"
            
            force_ocr = False
            if len(text.strip()) < ocr_threshold:
                force_ocr = True
            elif detect_table_or_chart(text, page):
                force_ocr = True
                print(f"   📊 Table/Chart detected on {pdf_path.name} page {i+1}, forcing VLM extraction...")
            
            if force_ocr:
                try:
                    images = convert_from_path(str(pdf_path), first_page=i + 1, last_page=i + 1, dpi=300)
                    text = vlm_extract_text(images[0], source_name=f"{pdf_path.name} page {i+1}")
                    extraction_method = "vlm_ocr"
                    ocr_count += 1
                except Exception as e:
                    print(f"⚠️ VLM extraction failed for {pdf_path.name} page {i+1}: {e}")
                    text = ""

            if text and text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name, 
                        "page": i + 1, 
                        "extraction_method": extraction_method,
                        "folder": folder_name
                    }
                ))

        if pages:
            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            print(f"   📄 {pdf_path.name} (in {folder_name}): {len(reader.pages)} pages → {len(chunks)} chunks ({ocr_count} via VLM)")

    if all_chunks:
        vectorstore.add_documents(all_chunks)
        print(f"✅ Ingested {len(all_chunks)} total chunks from folder into ChromaDB")
    else:
        print("⚠️ No extractable text found in the provided folder.")

def ingest_images(image_paths: list[str]):
    all_chunks = []
    for img_path in image_paths:
        path = Path(img_path)
        if not path.exists():
            print(f"⚠️ Skipping missing file: {img_path}")
            continue

        text = vlm_extract_text(path, source_name=path.name)

        if text and text.strip():
            doc = Document(page_content=text, metadata={"source": path.name, "extraction_method": "vlm_ocr"})
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
            print(f"   🖼️ {path.name}: {len(chunks)} chunks via VLM")

    if all_chunks:
        vectorstore.add_documents(all_chunks)
        print(f"✅ Ingested {len(all_chunks)} total image chunks into ChromaDB")

def ingest_urls(urls: list[str], delay: float = 2.0):
    all_chunks = []
    for url in urls:
        try:
            loader = WebBaseLoader(web_paths=[url], header_template={"User-Agent": "Mozilla/5.0"})
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

if __name__ == "__main__":
    sample_docs = [
        "LangGraph is a library for building stateful, multi-agent applications with LLMs.",
        "Qwen3 is a family of large language models developed by Alibaba Cloud.",
        "ChromaDB is an open-source embedding database designed for AI applications.",
        "DuckDuckGo is a privacy-focused search engine."
    ]
    ingest_texts(sample_docs)

    ingest_folder("/media/ailab/New Volume/GRAPHRAG_GIT/RAG-master/my_pdfs_folder")
    
    ingest_images([])

    app = build_rag_graph()

    print("\n" + "="*60)
    print("🤖 RAG Assistant Ready! (Powered by vLLM Text & Vision Models)")
    print("Type your question and press Enter.")
    print("Type 'exit', 'quit', or 'q' to stop the program.")
    print("="*60)

    while True:
        try:
            q = input("\n❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not q:
            continue
        
        if q.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        print(f"\n{'='*60}\nProcessing: {q}\n{'='*60}")

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
                    "llm": MODEL_NAME,
                    "vlm": VLM_MODEL_NAME,
                    "vectorstore": "chromadb",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "search_backend": "duckduckgo",
                }
            },
        )

        print(f"\n✅ Final Answer:\n{result['generation']}\n")
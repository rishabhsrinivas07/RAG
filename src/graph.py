"""
graph.py
--------
Defines the RAG pipeline as a LangGraph StateGraph.

Active flow (web search and direct-answer nodes are disabled):
    START → analyze → retrieve → grade → generate → hallucination_check → END

To re-enable web search or direct answers, uncomment the relevant
nodes/edges and update route_after_analysis / route_after_grading.
"""

import json
from typing import Literal, Annotated, Sequence, TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.config import llm, llm_generator, vectorstore, TEXT_MODEL, VISION_MODEL


# ── State schema ─────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    question:    str
    generation:  str
    documents:   list[Document]
    messages:    Annotated[Sequence, add_messages]
    route:       str
    relevance:   str
    grounded:    str
    retry_count: int


# ── Nodes ─────────────────────────────────────────────────────────────────────

def analyze_query(state: GraphState) -> dict:
    print("🔍 [Node] Analyzing query...")
    prompt = f"""Analyze this question and decide the best source.
Question: {state['question']}

Respond in JSON: {{"route": "vectorstore"}}
# OR {{"route": "web_search"}} OR {{"route": "direct"}}
- "vectorstore": domain-specific knowledge likely in our DB
- "web_search":  current events, real-time data, obscure topics
- "direct":      general knowledge / greetings needing no retrieval"""

    response = llm.invoke([HumanMessage(content=prompt)])
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


# -- Disabled nodes (uncomment to re-enable) ----------------------------------
#
# def search_web(state: GraphState) -> dict:
#     from langchain_community.tools import DuckDuckGoSearchRun
#     print("🌐 [Node] Searching DuckDuckGo...")
#     raw = DuckDuckGoSearchRun().run(state["question"])
#     return {"documents": [Document(page_content=raw, metadata={"source": "duckduckgo"})]}
#
# def generate_direct(state: GraphState) -> dict:
#     print("💬 [Node] Direct LLM response (no retrieval)...")
#     response = llm_generator.invoke([HumanMessage(content=state["question"])])
#     return {"generation": response.content,
#             "messages": [AIMessage(content=response.content)],
#             "documents": [], "grounded": "yes"}
#
# def end_no_answer(state: GraphState) -> dict:
#     print("🛑 [Node] No relevant information found.")
#     msg = "I couldn't find relevant information to answer your question."
#     return {"generation": msg, "messages": [AIMessage(content=msg)]}
# -----------------------------------------------------------------------------


def grade_documents(state: GraphState) -> dict:
    print("⚖️  [Node] Grading document relevance...")
    doc_text = "\n".join([d.page_content[:300] for d in state.get("documents", [])])
    prompt = f"""Are these documents relevant to answering the question?
Question: {state['question']}
Documents: {doc_text}

Respond in JSON: {{"relevance": "yes"}} or {{"relevance": "no"}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        relevance = json.loads(response.content)["relevance"]
    except Exception:
        relevance = "yes"

    print(f"   → Relevance: {relevance}")
    return {"relevance": relevance}


def generate_answer(state: GraphState) -> dict:
    print("✍️  [Node] Generating answer...")
    context    = "\n\n".join([d.page_content for d in state.get("documents", [])])
    system_msg = SystemMessage(
        content=(
            "You are a helpful RAG assistant. Answer using ONLY the provided context. "
            "If context is insufficient, say so. Never fabricate information."
        )
    )
    user_msg  = HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}")
    response  = llm_generator.invoke([system_msg, user_msg])

    print(f"   → Generated {len(response.content)} chars")
    return {"generation": response.content, "messages": [AIMessage(content=response.content)]}


def check_hallucination(state: GraphState) -> dict:
    print("🛡️  [Node] Checking hallucination...")
    context = "\n".join([d.page_content[:300] for d in state.get("documents", [])])
    prompt  = f"""Is this answer fully supported by the context?
Context: {context}
Answer: {state['generation']}

Respond in JSON: {{"grounded": "yes"}} or {{"grounded": "no"}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        grounded = json.loads(response.content)["grounded"]
    except Exception:
        grounded = "yes"

    retry = state.get("retry_count", 0) + 1
    print(f"   → Grounded: {grounded} (attempt {retry})")
    return {"grounded": grounded, "retry_count": retry}


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_analysis(state: GraphState) -> Literal["retrieve"]:
    # web_search and generate_direct are disabled; always retrieve
    return "retrieve"

    # Uncomment below when re-enabling those nodes:
    # if state["route"] == "web_search": return "search_web"
    # if state["route"] == "direct":     return "generate_direct"
    # return "retrieve"


def route_after_grading(state: GraphState) -> Literal["generate", "end_no_answer"]:
    if state["relevance"] == "yes":
        return "generate"
    return "end_no_answer"

    # Uncomment for web-search fallback:
    # if state.get("route") == "vectorstore": return "end_no_answer"
    # if state.get("retry_count", 0) < 1:     return "search_web"
    # return "end_no_answer"


def route_after_hallucination_check(state: GraphState) -> Literal["generate", "__end__"]:
    if state["grounded"] == "yes" or state.get("retry_count", 0) >= 3:
        return END
    return "generate"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_rag_graph():
    workflow = StateGraph(GraphState)

    # Active nodes
    workflow.add_node("analyze",             analyze_query)
    workflow.add_node("retrieve",            retrieve_from_vectorstore)
    workflow.add_node("grade",               grade_documents)
    workflow.add_node("generate",            generate_answer)
    workflow.add_node("hallucination_check", check_hallucination)

    # Disabled nodes — uncomment to re-enable
    # workflow.add_node("search_web",      search_web)
    # workflow.add_node("generate_direct", generate_direct)
    # workflow.add_node("end_no_answer",   end_no_answer)

    # Edges
    workflow.add_edge(START, "analyze")
    workflow.add_conditional_edges("analyze", route_after_analysis)
    workflow.add_edge("retrieve", "grade")
    # workflow.add_edge("search_web", "grade")       # re-enable with search_web node
    workflow.add_conditional_edges("grade", route_after_grading)
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges("hallucination_check", route_after_hallucination_check)
    # workflow.add_edge("end_no_answer", END)        # re-enable with end_no_answer node

    return workflow.compile()

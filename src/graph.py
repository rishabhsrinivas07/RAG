import json
from typing import Literal, Annotated, TypedDict, Sequence
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.config import llm, llm_generator, vectorstore, RETRIEVAL_K, MAX_HISTORY_MESSAGES


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
    return {"route": "vectorstore"}


def retrieve_from_vectorstore(state: GraphState) -> dict:
    print(" [Node] Retrieving from ChromaDB...")
    docs = vectorstore.similarity_search(state["question"], k=RETRIEVAL_K)
    print(f"   → Retrieved {len(docs)} documents")
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
    print("⚖️ [Node] Grading document relevance...")
    doc_text = "\n".join([d.page_content[:300] for d in state.get("documents", [])])
    prompt = f"""Are these documents relevant to answering the question?
Question: {state['question']}
Documents: {doc_text}
Respond in JSON: {{"relevance": "yes"}} or {{"relevance": "no"}}"""

    response = llm.invoke([HumanMessage(content=prompt)], config={"tags": ["grading"]})
    try:
        relevance = json.loads(response.content)["relevance"]
    except Exception:
        relevance = "yes"
    print(f"   → Relevance: {relevance}")
    return {"relevance": relevance}


def generate_answer(state: GraphState) -> dict:
    print("✍️ [Node] Generating answer...")

    # MEMORY MANAGEMENT: Sliding window truncation
    history = list(state.get("messages", []))
    trimmed_history = history[-MAX_HISTORY_MESSAGES:] if len(history) > MAX_HISTORY_MESSAGES else history

    context = "\n\n".join([d.page_content for d in state.get("documents", [])])
    system_msg = SystemMessage(
        content="You are a helpful RAG assistant. Answer using ONLY the provided context. "
                "Use the conversation history for context on follow-up questions. "
                "If context is insufficient, say so. Never fabricate information."
    )

    messages = [system_msg] + trimmed_history + [
        HumanMessage(content=f"Retrieved Context:\n{context}\n\nCurrent Question: {state['question']}")
    ]

    response = llm_generator.invoke(messages, config={"tags": ["generation"]})
    print(f"   → Generated response ({len(response.content)} chars) | History used: {len(trimmed_history)} msgs")
    
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

    response = llm.invoke([HumanMessage(content=prompt)], config={"tags": ["hallucination-check"]})
    try:
        grounded = json.loads(response.content)["grounded"]
    except Exception:
        grounded = "yes"

    retry = state.get("retry_count", 0) + 1
    print(f"   → Grounded: {grounded} (attempt {retry})")
    return {"grounded": grounded, "retry_count": retry}


def route_after_grading(state: GraphState) -> Literal["generate", "__end__"]:
    if state["relevance"] == "yes":
        return "generate"
    print("🛑 No relevant documents found in knowledge base.")
    return END


def route_after_hallucination_check(state: GraphState) -> Literal["generate", "__end__"]:
    if state["grounded"] == "yes":
        return END
    if state["retry_count"] >= 3:
        return END
    return "generate"


def build_rag_graph(checkpointer=None):
    workflow = StateGraph(GraphState)

    workflow.add_node("analyze", analyze_query)
    workflow.add_node("retrieve", retrieve_from_vectorstore)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("hallucination_check", check_hallucination)

    workflow.add_edge(START, "analyze")
    workflow.add_conditional_edges("analyze", lambda _: "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", route_after_grading)
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges("hallucination_check", route_after_hallucination_check)

    return workflow.compile(checkpointer=checkpointer)
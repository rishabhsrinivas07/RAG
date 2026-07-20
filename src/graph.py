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
    print("[NODE] Analyzing the query...")
    return {"route": "vectorstore"}


def retrieve_from_vectorstore(state: GraphState) -> dict:
    docs = vectorstore.similarity_search(state["question"], k=RETRIEVAL_K)
    
    print(f"\n{'='*20} RETRIEVAL STEP {'='*20}")
    print(f"🔍 Retrieved {len(docs)} chunks for query: '{state['question']}'")
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        preview = doc.page_content[:150].replace('\n', ' ') 
        print(f"  [Chunk {i+1} | Source: {source}]: {preview}...")
    print(f"{'='*55}\n")
    
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
    doc_text = "\n\n---\n\n".join([d.page_content for d in state.get("documents", [])])
    
    if not doc_text.strip():
        return {"relevance": "no"}

    prompt = f"""You are a grader assessing relevance of retrieved documents to a user question.
    
    Question: {state['question']}
    
    Retrieved Documents:
    {doc_text}
    
    Grading Rules:
    1. The document is relevant if it contains keywords, concepts, or direct answers related to the question.
    2. Do not be overly strict. If the document provides useful context to answer the question, it is relevant.
    
    Respond ONLY in valid JSON format with two keys:
    - "reason": A brief (1 sentence) explanation of why it is or isn't relevant.
    - "relevance": "yes" or "no"
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        
        # Robust JSON parsing for local models
        if content.startswith("```json"): content = content[7:]
        if content.endswith("```"): content = content[:-3]
            
        parsed_response = json.loads(content.strip())
        relevance = parsed_response.get("relevance", "yes").lower()
        print(f"🧠 Grader Reason: {parsed_response.get('reason', 'No reason provided')}")
        
    except Exception as e:
        print(f"⚠️ Grader JSON parsing failed ({e}). Defaulting to 'yes' to be safe.")
        relevance = "yes"
        
    return {"relevance": relevance}


def generate_answer(state: GraphState) -> dict:
    history = list(state.get("messages", []))
    trimmed_history = history[-MAX_HISTORY_MESSAGES:] if len(history) > MAX_HISTORY_MESSAGES else history
    
    context_docs = state.get("documents", [])
    if context_docs:
        context = "\n\n".join([f"[Document {i+1}]: {d.page_content}" for i, d in enumerate(context_docs)])
    else:
        context = "NO CONTEXT PROVIDED."

    total_chars = len(context)
    approx_tokens = total_chars // 4
    print(f"\n{'='*20} GENERATION STEP {'='*20}")
    print(f"📏 CONTEXT SIZE: ~{total_chars} characters (~{approx_tokens} tokens)")
    
    if approx_tokens > 3000:
        print(f"⚠️ WARNING: Context is large! Ensure LM Studio 'Context Length' is set to 8192+ to prevent silent truncation.")

    print(f"📄 FULL CONTEXT SENT TO LLM:\n{context[:1000]}...\n") 
    print(f"{'='*55}\n")

    system_instructions = f"""You are a strict RAG assistant. You must answer the user's question using ONLY the provided context.

### RULES:
1. If the answer is not explicitly stated in the context, you MUST reply with: "I cannot find the answer in the provided documents."
2. Do NOT use your pre-trained knowledge.
3. Do NOT make up information or guess.

### PROVIDED CONTEXT:
{context}
"""
    
    # 🛡️ CRITICAL FIX: Merge SystemMessage into the first HumanMessage.
    # This completely bypasses the Qwen LM Studio Jinja template bug.
    messages = []
    system_injected = False
    
    for msg in trimmed_history:
        if isinstance(msg, HumanMessage) and not system_injected:
            messages.append(HumanMessage(content=f"{system_instructions}\n\n---\n\nUser Query: {msg.content}"))
            system_injected = True
        else:
            messages.append(msg)
            
    if not system_injected:
        messages.append(HumanMessage(content=f"{system_instructions}\n\n---\n\nUser Query: {state.get('question', '')}"))

    print(f"💬 FINAL MESSAGES SENT TO LLM ({len(messages)} messages):")
    for msg in messages:
        role = msg.type
        preview = msg.content[:100].replace('\n', ' ')
        print(f"  [{role.upper()}]: {preview}...")
    print(f"{'='*55}\n")

    response = llm_generator.invoke(messages)
    
    return {"generation": response.content, "messages": [AIMessage(content=response.content)]}


# ==============================================================================
# ✅ THIS IS THE MISSING FUNCTION. DO NOT DELETE OR MOVE IT.
# ==============================================================================
def check_hallucination(state: GraphState) -> dict:
    print("[NODE] Checking for hallucinations...")
    context = "\n".join([d.page_content[:500] for d in state.get("documents", [])])
    
    prompt = f"""You are a strict fact-checker. 
    Determine if the generated answer is fully supported by the provided context.
    
    Context: {context}
    Generated Answer: {state['generation']}
    
    Respond ONLY in valid JSON format:
    {{"grounded": "yes"}} or {{"grounded": "no"}}
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        if content.startswith("```json"): content = content[7:]
        if content.endswith("```"): content = content[:-3]
        
        parsed = json.loads(content.strip())
        grounded = parsed.get("grounded", "yes").lower()
    except Exception as e:
        print(f"⚠️ Hallucination check JSON parsing failed ({e}). Defaulting to 'yes'.")
        grounded = "yes"
        
    return {"grounded": grounded, "retry_count": state.get("retry_count", 0) + 1}
# ==============================================================================


def route_after_grading(state: GraphState) -> Literal["generate", "__end__"]:
    if state["relevance"] == "yes":
        return "generate"
    print("🛑 No relevant documents found in knowledge base.")
    return END


def route_after_hallucination_check(state: GraphState) -> Literal["generate", "__end__"]:
    if state["grounded"] == "yes":
        return END
    if state["retry_count"] >= 3:
        print("⚠️ Max retry count reached. Ending generation.")
        return END
    print("🔄 Answer not fully grounded. Retrying generation...")
    return "generate"


def build_rag_graph(checkpointer=None):
    workflow = StateGraph(GraphState)

    workflow.add_node("analyze", analyze_query)
    workflow.add_node("retrieve", retrieve_from_vectorstore)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("generate", generate_answer)
    
    # This line will now work because check_hallucination is defined above it
    workflow.add_node("hallucination_check", check_hallucination)

    workflow.add_edge(START, "analyze")
    workflow.add_conditional_edges("analyze", lambda _: "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", route_after_grading)
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges("hallucination_check", route_after_hallucination_check)

    return workflow.compile(checkpointer=checkpointer)
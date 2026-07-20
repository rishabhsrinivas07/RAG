import sys
import uuid
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage

from src.config import MAX_HISTORY_MESSAGES
from src.ingestor import ingest_texts, ingest_folder, ingest_excel
from src.graph import build_rag_graph


def setup_memory():
    conn = sqlite3.connect("chat_sessions.db", check_same_thread=False)
    return SqliteSaver(conn)


def run_ingestion():
    print("🚀 Starting CLI Ingestion...")
    sample_docs = [
        "LangGraph is a library for building stateful, multi-agent applications with LLMs.",
        "Qwen3 is a family of large language models developed by Alibaba Cloud.",
    ]
    ingest_texts(sample_docs)
    # ingest_folder("./data/pdfs")
    ingest_excel("/home/ailab/updated_sheet.xlsx")
    print("✅ Ingestion finished.\n")


def run_chat():
    memory = setup_memory()
    app = build_rag_graph(checkpointer=memory)

    print("\n" + "="*60)
    print(f"🤖 RAG CLI Assistant Ready! (Memory: last {MAX_HISTORY_MESSAGES} messages)")
    print("Commands: 'new' = new session | 'exit' = quit")
    print("="*60)

    thread_id = f"cli_session_{uuid.uuid4().hex[:8]}"
    print(f"📌 Active session: {thread_id}")

    while True:
        try:
            q = input(f"\n❓ [{thread_id[-8:]}] You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n Goodbye!")
            break

        if not q: continue
        if q.lower() in ["exit", "quit", "q"]:
            print(" Goodbye!")
            break
        if q.lower() == "new":
            thread_id = f"cli_session_{uuid.uuid4().hex[:8]}"
            print(f"🔄 New session: {thread_id}")
            continue

        print(f"\n{'='*60}\nProcessing: {q}\n{'='*60}")

        result = app.invoke(
            {
                "question": q,
                "messages": [HumanMessage(content=q)],
                "documents": [], "generation": "", "route": "", 
                "relevance": "", "grounded": "", "retry_count": 0,
            },
            config={
                "configurable": {"thread_id": thread_id},
                "metadata": {"query_type": "cli_rag"}
            },
        )

        print(f"\n✅ Assistant:\n{result['generation']}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        run_ingestion()
    else:
        run_chat()
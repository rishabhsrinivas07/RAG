"""
main.py
-------
Entry point for the RAG assistant.

Usage:
    python main.py

The script ingests sample documents and then enters an interactive Q&A loop.
"""

from langchain_core.messages import HumanMessage

from src.ingestor import ingest_texts, ingest_folder, ingest_images
from src.graph import build_rag_graph
from src.config import TEXT_MODEL, VISION_MODEL


def main():
    # ── 1. Ingest documents ───────────────────────────────────────────────────
    sample_docs = [
        "LangGraph is a library for building stateful, multi-agent applications with LLMs.",
        "Qwen3 is a family of large language models developed by Alibaba Cloud.",
        "ChromaDB is an open-source embedding database designed for AI applications.",
        "DuckDuckGo is a privacy-focused search engine.",
    ]
    ingest_texts(sample_docs)
    ingest_folder("/media/ailab/New Volume/GRAPHRAG_GIT/RAG-master/pdfs")
    ingest_images([])   # pass image paths here if needed

    # ── 2. Build graph ────────────────────────────────────────────────────────
    app = build_rag_graph()

    # ── 3. Interactive Q&A loop ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"🤖 RAG Assistant Ready!  (LLM: {TEXT_MODEL} | VLM: {VISION_MODEL})")
    print("Type your question and press Enter.")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("=" * 60)

    while True:
        try:
            q = input("\n❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("👋 Goodbye!")
            break

        print(f"\n{'=' * 60}\nProcessing: {q}\n{'=' * 60}")

        result = app.invoke(
            {
                "question":    q,
                "messages":    [HumanMessage(content=q)],
                "documents":   [],
                "generation":  "",
                "route":       "",
                "relevance":   "",
                "grounded":    "",
                "retry_count": 0,
            },
            config={
                "metadata": {
                    "query_type":      "adaptive_rag",
                    "llm":             TEXT_MODEL,
                    "vlm":             VISION_MODEL,
                    "vectorstore":     "chromadb",
                    "embedding_model": "all-MiniLM-L6-v2",
                }
            },
        )

        print(f"\n✅ Final Answer:\n{result['generation']}\n")


if __name__ == "__main__":
    main()

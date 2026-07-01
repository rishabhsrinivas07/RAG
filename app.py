import streamlit as st
import sqlite3
import uuid
import os
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import MAX_HISTORY_MESSAGES
from src.ingestor import ingest_texts, ingest_folder, ingest_images
from src.graph import build_rag_graph

# ============================================================
# DATABASE & SESSION MANAGEMENT
# ============================================================
DB_PATH = "chat_sessions.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Create a simple table to track our UI sessions
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ui_sessions (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def create_new_session(conn, title="New Chat"):
    thread_id = f"session_{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO ui_sessions (thread_id, title, created_at) VALUES (?, ?, ?)",
        (thread_id, title, datetime.now().isoformat())
    )
    conn.commit()
    return thread_id

def get_all_sessions(conn):
    return conn.execute("SELECT * FROM ui_sessions ORDER BY created_at DESC").fetchall()

def delete_session(conn, thread_id):
    conn.execute("DELETE FROM ui_sessions WHERE thread_id = ?", (thread_id,))
    conn.commit()

# ============================================================
# STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

# Initialize DB and Checkpointer
conn = get_db_connection()
memory = SqliteSaver(conn)
app = build_rag_graph(checkpointer=memory)

# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = create_new_session(conn, "Default Session")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# SIDEBAR: SESSION MANAGEMENT & INGESTION
# ============================================================
with st.sidebar:
    st.title("🤖 RAG Assistant")
    
    # --- Session Management ---
    st.header(" Sessions")
    
    if st.button("➕ New Chat", use_container_width=True):
        new_title = f"Session {len(get_all_sessions(conn)) + 1}"
        st.session_state.thread_id = create_new_session(conn, new_title)
        st.session_state.chat_history = []
        st.rerun()

    sessions = get_all_sessions(conn)
    for session in sessions:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                f"💬 {session['title']}", 
                key=f"btn_{session['thread_id']}",
                use_container_width=True,
                type="primary" if session['thread_id'] == st.session_state.thread_id else "secondary"
            ):
                st.session_state.thread_id = session['thread_id']
                st.session_state.chat_history = [] # Clear UI history when switching
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{session['thread_id']}"):
                delete_session(conn, session['thread_id'])
                if st.session_state.thread_id == session['thread_id']:
                    st.session_state.thread_id = sessions[0]['thread_id'] if sessions else create_new_session(conn)
                st.rerun()

    st.divider()

    # --- Ingestion Controls ---
    st.header("📂 Data Ingestion")
    
    folder_path = st.text_input(
        "PDF Folder Path", 
        value="./data",
        help="Absolute path to the folder containing PDFs"
    )
    
    if st.button("🚀 Ingest Folder", use_container_width=True):
        if os.path.exists(folder_path):
            with st.spinner("Ingesting documents... This may take a while."):
                ingest_folder(folder_path)
            st.success("Ingestion complete!")
        else:
            st.error("Folder path does not exist.")

    st.divider()
    st.caption(f"Memory: Last {MAX_HISTORY_MESSAGES} messages | Model: Qwen3.5-9B")

# ============================================================
# MAIN CHAT INTERFACE
# ============================================================
st.header(f"Chat: {next((s['title'] for s in sessions if s['thread_id'] == st.session_state.thread_id), 'Unknown')}")

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Add user message to UI
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process with LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the graph with the current thread_id
                result = app.invoke(
                    {
                        "question": prompt,
                        "messages": [HumanMessage(content=prompt)],
                        "documents": [],
                        "generation": "",
                        "route": "",
                        "relevance": "",
                        "grounded": "",
                        "retry_count": 0,
                    },
                    config={
                        "configurable": {"thread_id": st.session_state.thread_id},
                        "metadata": {"query_type": "streamlit_rag"}
                    },
                )
                
                response_text = result.get("generation", "No response generated.")
                
                # 3. Add assistant response to UI
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
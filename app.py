import streamlit as st
import os
import io
import contextlib
import logging

# Suppress harmless Streamlit/Transformers warnings
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from langchain_core.messages import HumanMessage, AIMessage

# ✅ FIX 1: Removed LLM_BACKEND from this import
from src.config import MAX_HISTORY_MESSAGES 
from src.ingestor import ingest_folder, ingest_excel
from src.graph import build_rag_graph

# ============================================================
# STREAMLIT UI SETUP
# ============================================================
st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

# Initialize the graph (No checkpointer needed since we manage memory manually)
app = build_rag_graph()

# Initialize session state for the singular chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🤖 RAG Assistant")
    
    # ✅ FIX 2: Removed LLM_BACKEND from the caption
    st.caption(f"Memory: Last {MAX_HISTORY_MESSAGES} msgs")
    st.divider()
    
    st.header("⚙️ Chat Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.header("📂 Data Ingestion")
    
    folder_path = st.text_input("PDF Folder Path", value="./data", help="Path to folder containing PDFs")
    if st.button("🚀 Ingest Folder", use_container_width=True):
        if os.path.exists(folder_path):
            with st.spinner("Ingesting documents..."):
                ingest_folder(folder_path)
            st.success("Ingestion complete!")
        else:
            st.error("Folder path does not exist.")

    st.divider()
    st.subheader("Upload Excel File")
    uploaded_excel = st.file_uploader("Choose an .xlsx or .xls file", type=["xlsx", "xls"])
    
    if uploaded_excel is not None:
        save_path = f"./data/{uploaded_excel.name}"
        os.makedirs("./data", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_excel.getbuffer())
            
        if st.button("📊 Ingest Excel", use_container_width=True):
            with st.spinner("Processing Excel file..."):
                num_chunks = ingest_excel(save_path)
            
            if num_chunks > 0:
                st.success(f"Successfully ingested {uploaded_excel.name}!")
                st.info(f"📊 **Extraction Details:** Extracted and embedded **{num_chunks} chunks**.")
            else:
                st.error(f"❌ Failed to extract data from {uploaded_excel.name}.")

# ============================================================
# MAIN CHAT INTERFACE
# ============================================================
st.header("💬 Chat with your Documents")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Add user message to history and display it
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking & Processing..."):
            
            # 🎯 MAGIC: Capture all terminal print() statements during execution
            log_capture = io.StringIO()
            
            try:
                # 🧠 MEMORY HANDLING: Build the message history for LangGraph
                langchain_messages = []
                for msg in st.session_state.chat_history[-MAX_HISTORY_MESSAGES:]:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))

                with contextlib.redirect_stdout(log_capture):
                    result = app.invoke(
                        {
                            "question": prompt,
                            "messages": langchain_messages, # Pass the sliding window of history
                            "documents": [], "generation": "", "route": "", 
                            "relevance": "", "grounded": "", "retry_count": 0,
                        }
                    )
                
                response_text = result.get("generation", "No response generated.")
                logs = log_capture.getvalue()
                
                # 3. Display the captured background processes in an expander
                if logs.strip():
                    with st.expander("🔍 View Background Processes & Terminal Logs", expanded=True):
                        st.code(logs, language="text")
                
                # 4. Display the final answer and save to history
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                logs = log_capture.getvalue()
                
                if logs.strip():
                    with st.expander("🔍 View Logs Before Crash", expanded=True):
                        st.code(logs, language="text")
                
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})